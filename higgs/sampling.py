"""
Sampling functions for HiGGs, for NeurIPS 23, by anonymous authors
"""


from tqdm import tqdm
from collections import Counter

from higgs_utils import *

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

# Need DiGress utils - liable to change in later version
pwd = os.getcwd()
dgd_dir = os.path.join(pwd, "dgd")
sys.path.insert(0, dgd_dir)
import utils
from diffusion.beta_diffusion_utils import fixed_h2_condition
try:
    from .interface_ablation import (
        adapt_h2_to_glue_class,
        adapt_h2_to_h1_conditions,
        apply_component_policy,
        audit_glue_training_cache,
        class_histogram,
        inference_context,
        interface_options,
        model_sampling_state,
        one_hot_histogram,
        tensor_graph,
        unique_undirected_pairs,
        validate_discrete_graph,
    )
except ImportError:
    from interface_ablation import (
        adapt_h2_to_glue_class,
        adapt_h2_to_h1_conditions,
        apply_component_policy,
        audit_glue_training_cache,
        class_histogram,
        inference_context,
        interface_options,
        model_sampling_state,
        one_hot_histogram,
        tensor_graph,
        unique_undirected_pairs,
        validate_discrete_graph,
    )


def get_h2_graph(cfg, intermediate_figures=False, graph_number=0,
                 return_audit=False):
    """
    Samples an h2 graph using a pre-trained model (HiGGs Stage One)

    params:
    param: cfg: omegadict.DictConfig, config keys for model loading

    returns:
    param: h2_graph: [[h1_graph_categories, h2_adjacency]] as torch.Tensors
    """
    cfg_h2 = cfg.copy()
    cfg_h2["dataset"]["h"] = 2
    cfg_h2["model"]["extra_features"] = 'all'

    # Load h2 model and updated config, then move to GPU if available
    cfg_h2, model_h2 = get_samplers(cfg_h2)
    device = 'cuda' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu'
    model_h2 = model_h2.to(device)
    # H2 already used eval/inference mode before the interface ablation was
    # introduced.  Keep that behavior in C0 as well as force_eval mode.
    model_h2.eval()
    model_audit = model_sampling_state(
        model_h2, omegaconf.OmegaConf.select(
            cfg, "general.h2_model", default=""), stage="H2")

    # Sample the graph, and a networkx.Graph version
    h2_sampling_kwargs = dict(
        batch_id=0,
        batch_size=1,
        keep_chain=0,
        number_chain_steps=40,
        num_nodes=None,
        figures=intermediate_figures,
    )
    if isinstance(model_h2, ConditionalGraphBetaDiffusionH2):
        # Only Beta-H2 uses this fixed pseudo-condition. Original discrete
        # H1 and Glue keep the upstream HiGGs interfaces unchanged.
        h2_sampling_kwargs["y"] = fixed_h2_condition(
            model_h2.ydim_output).to(device)
        h2_sampling_kwargs["graph_number"] = graph_number
    with torch.inference_mode():
        h2_graph, h2_networkx = model_h2.sampling_batch_h2(
            **h2_sampling_kwargs)

    options = interface_options(cfg)
    full_nodes, full_adjacency = h2_graph[0][0], h2_graph[0][1]
    used_nodes, used_adjacency, component_audit = apply_component_policy(
        full_nodes, full_adjacency, options["h2_component_policy"],
        label="H2")
    h2_graph = [[used_nodes, used_adjacency]]
    full_networkx = tensor_graph(full_nodes, full_adjacency)
    used_networkx = tensor_graph(used_nodes, used_adjacency)
    audit = {
        **component_audit,
        "raw_class_histogram": class_histogram(full_nodes),
        "used_class_histogram": class_histogram(used_nodes),
        "model": model_audit,
    }

    print(
        f"Sampled h2 n_nodes: {full_nodes.shape[0]} -> "
        f"{used_nodes.shape[0]} "
        f"({options['h2_component_policy']})")

    # print(f"h2 X: {h2_graph[0][0]}")
    # print(f"Unique h2: {np.unique(h2_graph[0][1])}")
    # print(f"h2 E: {h2_graph[0][1]}")
    # print(f"Sampled inter-community {h2_networkx[0]}")

    if return_audit:
        return h2_graph, full_networkx, used_networkx, audit
    return h2_graph

def get_h1_graphs(cfg, h1_graph_types, h1_batch_size,
                  return_audit=False):
    """
    Sample h1 graphs (HiGGs Stage Two)

    params:
    param: cfg: omegadict.DictConfig, config keys for model loading
    param: h1_graph_types: torch.Tensor categories for conditioning h1 graph sampling
    param: h1_batch_size: Size of batches to use during sampling

    returns:
    param: h1_adjs:     torch.Tensor adjacency matrices for each sampled h1 graph
    param: h1_node_types:   torch.Tensor of node classes for each sampled h1 graph
    param: h1_label_mapping:    dictionary mapping to keep track of node ids
    param: node_to_class:   dictionary mapping of node ids and sampled class for that node
    param: n_nodes_counter:     list of graph sizes as each h1 graph is added
    """

    # Use config to load h1 sampling model and send to gpu if available
    cfg_h1 = cfg.copy()
    cfg_h1["dataset"]["h"] = 1
    cfg_h1, model_h1 = get_samplers(cfg_h1)#, model_kwargs_h1)
    device = 'cuda' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu'
    model_h1 = model_h1.to(device)
    options = interface_options(cfg)
    if options["force_eval"]:
        model_h1.eval()
    model_audit = model_sampling_state(
        model_h1, omegaconf.OmegaConf.select(
            cfg, "general.h1_model", default=""), stage="H1")
    if options["force_eval"] and model_audit["dropout_training_count"] != 0:
        raise RuntimeError(
            "force_eval=true but an H1 Dropout module is still training")

    # Get dimensions of model outputs
    x_out, y_out, e_out = model_h1.Xdim_output, model_h1.ydim_output, model_h1.Edim_output

    # Throws errors currently for out_dim==1
    if x_out == 1:
        x_out += 1
    if y_out == 1:
        y_out += 1
    if e_out == 1:
        e_out += 1

    # Get One-hot representation of graph categories to condition h1 sampling
    temp_batch_size = h1_batch_size
    h1_y = adapt_h2_to_h1_conditions(
        h1_graph_types,
        y_out,
        options["h2_to_h1_condition_mode"],
        dataset_name=str(cfg.general.name),
    )

    # Need to instantiate some aggregators, and determine batch sizes
    h1_node_types = []
    h1_adjs = []
    n_nodes_counter = []
    h1_label_mapping = []
    h1_full_graphs = []
    h1_used_graphs = []
    h1_component_audits = []
    if h1_graph_types.shape[0] % temp_batch_size == 0:

        print(f"Data is {h1_graph_types.shape[0]} long, using {h1_graph_types.shape[0] / temp_batch_size} batches (no remainder)")

        n_batches = int(h1_graph_types.shape[0] / temp_batch_size)
        batch_sizes = np.full(n_batches, temp_batch_size)
    else:

        print(f"Data is {h1_graph_types.shape[0]} long, using {int(h1_graph_types.shape[0] / temp_batch_size) + 1} batches")

        n_batches = int(h1_graph_types.shape[0] / temp_batch_size) + 1
        batch_sizes = np.array([temp_batch_size]*(n_batches - 1) + [h1_graph_types.shape[0] % temp_batch_size])

    # Iterate over batches and conditioning classes, sample h1 graphs
    node_agg = 0
    pbar = tqdm(range(n_batches))
    node_to_class = {}
    for n in pbar:
        batch_ind_low, batch_ind_high = temp_batch_size * n, temp_batch_size*(n+1)
        try:
            y_batch = h1_y[batch_ind_low:batch_ind_high, :]
        except:
            y_batch = h1_y[batch_ind_low:,:]
        y_batch = y_batch.to(device)

        figures = bool(omegaconf.OmegaConf.select(
            cfg, "general.intermediate_figures", default=False)) and n % 10 == 0

        with inference_context(options["force_eval"]):
            graphs, networkx_graphs = model_h1.sampling_batch_h1(
                n, batch_sizes[n], 0, 40, y=y_batch, figures=figures)
        for i in range(len(graphs)):
            full_adj, full_nodes = graphs[i][1], graphs[i][0]
            nodes, adj, component_audit = apply_component_policy(
                full_nodes, full_adj, options["h1_component_policy"],
                label=f"H1 community {len(h1_adjs)}")
            component_audit["community_id"] = len(h1_adjs)
            h1_component_audits.append(component_audit)
            h1_full_graphs.append(tensor_graph(full_nodes, full_adj))
            h1_used_graphs.append(tensor_graph(nodes, adj))

            h1_node_types.append(nodes.to("cpu"))
            h1_adjs.append(adj.to("cpu"))
            print(torch.sum(adj[-1,:]))
            n_nodes = adj.shape[0]
            n_nodes_counter += [node_agg]
            h1_label_mapping += [{"start_ind":node_agg, "n_nodes":n_nodes}]

            for i, n in enumerate(nodes.tolist()):
                node_to_class[i+node_agg] = n

            node_agg += n_nodes
            h1_label_mapping[-1]["end_ind"] = node_agg

        pbar.set_description(f"N nodes added: {node_agg}")

    audit = {
        "model": model_audit,
        "h2_raw_class_histogram": class_histogram(h1_graph_types),
        "condition_mode": options["h2_to_h1_condition_mode"],
        "condition_histogram_after_adapter": one_hot_histogram(h1_y),
        "communities": h1_component_audits,
    }
    model_h1.to("cpu")
    del h1_y
    del model_h1
    if return_audit:
        return (
            h1_adjs, h1_node_types, h1_label_mapping, node_to_class,
            n_nodes_counter, h1_full_graphs, h1_used_graphs, audit)
    return h1_adjs, h1_node_types, h1_label_mapping, node_to_class, n_nodes_counter

def size_classing(type_idx, mean_size, dev_sizes, num_classes_out = 5):
    """
    Returns the binning of graph size given mu, sigma
    Currently mu, sigma are calculated during initial partitioning
    NB: Not used in HiGGs saved models

    type_idx: list(node class for each node)
    """

    num_nodes = len(type_idx)

    position = (num_nodes - mean_size) / dev_sizes
    shift_for_classes = int((num_classes_out - 1)/2)
    try:
        if position < - shift_for_classes:
            graph_type = 0
        elif position > shift_for_classes:
            graph_type = int(num_classes_out)
        else:
            graph_type = int(np.around(position, decimals=0)) + shift_for_classes
    except:
        print(f"\n\ntype idx {type_idx}\n"
              f"Community size {num_nodes}\n"
              f"Classes shift {shift_for_classes}\n"
              f"And position {position}\n"
              f"Mu, sigma {mean_size} {dev_sizes}\n")
        quit()

    return graph_type

def prepare_glue_dataset(
        cfg, model_glue, h1_adjs, h1_node_types, h2_edge_index,
        h2_edge_type, glue_batch_size, return_audit=False):
    """
    Function to prepare the dataset for HiGGs Stage Three

    params:
    param: cfg: omegaconf.DictConfig
    param: model_glue: PyTorch model used for Stage Three
    param: h1_adjs: torch.Tensor adjacency matrices for sampled h1 graphs
    param: h1_node_types: torch.Tensor(s) of node categories for sampled h1 graphs
    param: h2_edge_index: torch.Tensor of edges in sampled h2 graph
    param: h2_edge_type: torch.Tensor of edge categories in sampled h2 graph (for conditioning)
    param: glue_batch_size: The batch size for edge-sampling

    returns:
    param: datalist: torch geometric dataset
    param: loader: dataloader for that dataset
    param: batch_size: batch size - slightly redundant to pass this back
    param: n_batches: number of batches
    param: remainder: final batch size
    param: pair_ids: list of (h1_id1, h1_id2) tuples to identify the graphs in the dataset
    """

    device = 'cuda' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu'

    options = interface_options(cfg)
    training_cache_audit = None
    if options["glue_training_cache"] is not None:
        training_cache_audit = audit_glue_training_cache(
            str(options["glue_training_cache"]))
        print(
            "[HiGGs interface] Glue training y cache: "
            f"{training_cache_audit['absolute_path']} "
            f"sha256={training_cache_audit['sha256']} "
            f"histogram={training_cache_audit['one_hot_histogram']} "
            "verified_present_class_index="
            f"{training_cache_audit['verified_present_class_index']}")
    if (options["h2_to_glue_condition_mode"] == "sbm_semantic"
            and training_cache_audit is None):
        raise ValueError(
            "A Glue training cache is required for sbm_semantic adaptation")

    # The H2 adjacency is symmetric, so dense_to_sparse contains both edge
    # orientations. Keep one stable record per unordered present pair.
    pair_records = list(
        unique_undirected_pairs(h2_edge_index, h2_edge_type)
    )

    if not pair_records:
        raise ValueError(
            "Glue received no H2 community pairs"
        )

    # DYNAMIC_GLUE_PADDING_FIX
    # All pos tensors in the Glue dataset must have one common size.
    # The original fixed size 700 fails when a generated pair has
    # more than 700 nodes.
    pair_node_totals = [
        int(h1_node_types[int(left)].shape[0])
        + int(h1_node_types[int(right)].shape[0])
        for left, right, _ in pair_records
    ]

    max_pair_nodes = max(pair_node_totals)
    max_nodes = max(700, max_pair_nodes)

    oversize_pairs = sum(
        total > 700
        for total in pair_node_totals
    )

    print(
        "[HiGGs interface] Glue dynamic padding: "
        f"base=700 "
        f"max_pair_nodes={max_pair_nodes} "
        f"max_nodes={max_nodes} "
        f"oversize_pairs={oversize_pairs}"
    )

    pbar = tqdm(pair_records)
    datalist = []
    pair_ids = []
    raw_edge_classes = []
    adapted_glue_conditions = []

    x_out, y_out, e_out = model_glue.Xdim_output, model_glue.ydim_output, model_glue.Edim_output
    # Avoid errors thrown when out_dim==1
    if x_out == 1:
        x_out += 1
    if y_out == 1:
        y_out += 1
    if e_out == 1:
        e_out += 1

    # Lots of code also found in dgd/datasets/..., better documented there
    # Prepares datasets of pairs of h1 graphs
    dataset_name = str(
        cfg.dataset.get("name", cfg.general.name)
    ).strip().lower()

    is_cora = (
        dataset_name == "cora"
        or dataset_name.startswith("cora_")
    )

    print(
        "[HiGGs interface] Glue condition dataset: "
        f"dataset_name={dataset_name} is_cora={is_cora}"
    )

    for left, right, raw_edge_class in pbar:
        pair_ids.append((left, right))
        raw_edge_classes.append(raw_edge_class)

        com1_adj, com2_adj = h1_adjs[left], h1_adjs[right]
        com1_nodes, com2_nodes = h1_node_types[left], h1_node_types[right]
        com1_nodes, com2_nodes = com1_nodes, com2_nodes

        com1_edgelist, com1_attr = dense_to_sparse(com1_adj)
        com2_edgelist, com2_attr = dense_to_sparse(com2_adj)

        pair_nodes = torch.cat((com1_nodes, com2_nodes))
        N = pair_nodes.shape[0]
        com2_edgelist = com2_edgelist + com1_nodes.shape[0]
        pair_edges = torch.cat((com1_edgelist, com2_edgelist), axis = -1)

        pos = torch.Tensor([True]*com1_nodes.shape[0] + [False]*com2_nodes.shape[0]).to(torch.int)
        pos = torch.Tensor(utils.get_intra_matrix(pos))
        pad_nodes = max_nodes - int(N)

        if pad_nodes < 0:
            raise ValueError(
                "Glue padding became negative despite dynamic sizing: "
                f"N={N}, max_nodes={max_nodes}"
            )

        pos = F.pad(
            input=pos,
            pad=(0, pad_nodes, 0, pad_nodes),
            value=False,
        )

        x = F.one_hot(pair_nodes, num_classes=x_out)
        if x_out == 2:
            x = x[:,0].reshape(-1,1)

        if is_cora:
            # CORA_GLUE_PAIR_MAJORITY_FIX
            # Glue training uses the majority node class of the joined
            # community pair. Reproduce exactly the same condition here.
            pair_node_classes = (
                pair_nodes.detach().long().reshape(-1).cpu()
            )

            if pair_node_classes.numel() == 0:
                raise ValueError(
                    "Cora Glue pair contains no node classes"
                )

            min_class = int(pair_node_classes.min().item())
            max_class = int(pair_node_classes.max().item())

            if min_class < 0 or max_class >= y_out:
                raise ValueError(
                    "Cora Glue pair node class is outside the "
                    f"checkpoint y range: min={min_class}, "
                    f"max={max_class}, y_out={y_out}"
                )

            class_counts = torch.bincount(
                pair_node_classes,
                minlength=y_out,
            )

            # torch.argmax and numpy.argmax both select the first class
            # during a tie, matching majority_classing().
            graph_type = int(
                torch.argmax(class_counts).item()
            )
        else:
            graph_type = adapt_h2_to_glue_class(
                raw_edge_class,
                options["h2_to_glue_condition_mode"],
                configured_present_class_index=options[
                    "glue_present_class_index"],
                training_cache_audit=training_cache_audit,
                dataset_name=str(cfg.general.name),
            )
        if graph_type < 0 or graph_type >= y_out:
            raise ValueError(
                f"Glue condition class {graph_type} is outside {y_out} classes")
        y = F.one_hot(
            torch.tensor([graph_type]), num_classes=y_out).float()
        adapted_glue_conditions.append(y.squeeze(0))

        edge_attr = torch.cat((com1_attr, com2_attr))
        edge_attr = F.one_hot(edge_attr, num_classes=e_out).to(torch.float)

        data = Data(x = x,
                    edge_index=pair_edges,
                    edge_attr = edge_attr,
                    pos = pos,
                    y = y,
                    pair_order=torch.tensor(
                        [len(datalist)], dtype=torch.long)).to(device)
        datalist.append(data)


    datalist = datalist
    batch_size = glue_batch_size
    remainder = len(datalist) % batch_size
    n_batches = len(datalist) / batch_size if remainder == 0 else int(len(datalist) / batch_size) + 1
    print(f"Batch size of {batch_size}, leaves remainder {remainder} as final batch")



    loader = DataLoader(datalist, batch_size=batch_size, shuffle=False)#.to(device)
    print(f"Pair-pair community dataset is size {sys.getsizeof(loader)} in memory")

    raw_histogram = {
        str(key): int(value)
        for key, value in sorted(Counter(raw_edge_classes).items())
    }
    if adapted_glue_conditions:
        condition_tensor = torch.stack(adapted_glue_conditions)
        condition_histogram = one_hot_histogram(condition_tensor)
    else:
        condition_histogram = {}
    audit = {
        "condition_mode": (
            "pair_majority"
            if is_cora
            else options["h2_to_glue_condition_mode"]
        ),
        "raw_h2_edge_type_histogram": raw_histogram,
        "glue_condition_histogram_after_adapter": condition_histogram,
        "verified_glue_training_y_histogram": (
            None if training_cache_audit is None
            else training_cache_audit["one_hot_histogram"]),
        "glue_training_cache": training_cache_audit,
        "pair_ids": [list(pair) for pair in pair_ids],
    }
    if return_audit:
        return (
            datalist, loader, batch_size, n_batches, remainder, pair_ids,
            audit)
    return datalist, loader, batch_size, n_batches, remainder, pair_ids

def get_glue_model(cfg):
    # Load model for HiGGs Stage Three
    cfg_glue = cfg.copy()
    cfg_glue["dataset"]["h"] = 1.5
    cfg_glue, model_glue = get_samplers(cfg_glue)#, model_kwargs_glue)

    return cfg_glue, model_glue

def get_inter_edges(cfg, model_glue, loader, n_batches, batch_size, remainder,
                    pair_ids=None, return_audit=False):
    """
    Sample inter-h1 edges (HiGGs Stage Three) using a loaded model and prepared dataset

    params:
    param: cfg: omegaconf.DictConfig
    param: model_glue: torch model to sample edges
    param: loader: dataloader of un-connected graphs
    param: n_batches: number of batches
    param: batch_size: the size of each batch
    param: remainder: the size of the final batch

    returns:
    param: out_nx: list of networkx.Graphs, which are connected h1 graphs

    """
    # Send model to GPU if available
    device = 'cuda' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu'
    model_glue = model_glue.to(device)
    options = interface_options(cfg)
    if options["force_eval"]:
        model_glue.eval()
    model_audit = model_sampling_state(
        model_glue, omegaconf.OmegaConf.select(
            cfg, "general.glue_model", default=""), stage="Glue")
    if options["force_eval"] and model_audit["dropout_training_count"] != 0:
        raise RuntimeError(
            "force_eval=true but a Glue Dropout module is still training")
    out_nx = []
    observed_pair_order = []

    pbar_glue = tqdm(loader)
    for n, batch in enumerate(pbar_glue):
        pbar_glue.set_description(f"Batch {n}")
        if hasattr(batch, "pair_order"):
            observed_pair_order.extend(
                int(item) for item in batch.pair_order.detach().cpu().tolist())
        # Determine the size of this batch
        if n != n_batches - 1:
            bs = batch_size
        else:
            bs = remainder if remainder != 0 else batch_size
            print(f"Doing final batch with remainder {remainder}")

        # Often too many batches to visualise all
        figures = bool(omegaconf.OmegaConf.select(
            cfg, "general.intermediate_figures", default=False)) and n % 50 == 0

        # Predict edges
        with inference_context(options["force_eval"]):
            pair_graphs, pair_nx = model_glue.sampling_batch_glue(
                n, bs, 0, 40, data=batch, figures=figures)

        # Send predicted graphs to aggregator
        for i in range(bs):
            out_nx.append(pair_nx[i])
        del pair_graphs, pair_nx

    expected_order = list(range(len(out_nx)))
    if observed_pair_order and observed_pair_order != expected_order:
        raise RuntimeError(
            "Glue DataLoader output order differs from datalist/pair_ids: "
            f"expected={expected_order}, observed={observed_pair_order}")
    if pair_ids is not None and len(pair_ids) != len(out_nx):
        raise RuntimeError(
            "Glue output count does not match pair_ids: "
            f"{len(out_nx)} != {len(pair_ids)}")
    audit = {
        "model": model_audit,
        "observed_pair_order": observed_pair_order,
        "output_count": len(out_nx),
    }
    if return_audit:
        return out_nx, audit
    return out_nx


def inter_edge_specific(h1_nx, glue_nx, pair_ids, return_details=False):
    """
    Extract new inter-h1 edges from graphs of connected h1-subgraphs

    params:
    param: h1_nx: list of sampled h1 networkx.Graph(s)
    param: glue_nx: list of connected h1 pairs as networkx.Graph(s)
    param: pair_ids: identifiers for which h1 graphs are in which glue_nx connected pair

    returns:
    param: all_edges: list of added edges
    """

    # Aggregate all inter-h1 edges
    all_edges = []
    pair_details = []

    pbar_pairs = tqdm(range(len(pair_ids)))
    pbar_pairs.set_description("Extracting inter h1 edges")

    for i in pbar_pairs:
        # Get identifiers
        com1, com2 = pair_ids[i]

        # Get corresponding h1 graphs
        g1, g2 = h1_nx[com1], h1_nx[com2]

        # Get un-connected graphs and compose into single networkx.Graph object
        gX = nx.compose(g1, g2)
        # Get linked pair of same graphs
        gY = glue_nx[i]

        # Relabel nodes in connected pair to match un-connected original h1 graphs
        relabel_mapping = {}
        for i, n in enumerate(gX.nodes()):
            relabel_mapping[i] = n
        gY = nx.relabel_nodes(gY, relabel_mapping)

        # Add edges if they aren't already in one of the h1 graphs
        x_edges = list(gX.edges())
        y_edges = list(gY.edges())
        pair_new_edges = []
        for e in y_edges:
            if e not in x_edges and (e[1], e[0]) not in x_edges:
                canonical = (min(int(e[0]), int(e[1])),
                             max(int(e[0]), int(e[1])))
                all_edges.append(canonical)
                pair_new_edges.append(canonical)
        pair_details.append({
            "pair_id": [int(com1), int(com2)],
            "new_edges": [list(edge) for edge in pair_new_edges],
        })

    if return_details:
        return all_edges, pair_details
    return all_edges

def build_from_components(
        h1_node_types, h1_adjs, out_nx, h1_label_mapping, node_to_class,
        pair_ids, strict_assertions=True, return_audit=False):
    """
    Combines graphs from each previous stage into the final sampled graph

    params:
    param: h1_node_types: torch.Tensor of node categories from Stage Two
    param: h1_adjs: Adjacency matrices for each sampled graph from Stage Two
    param: out_nx: connected networkx.Graph(s) from Stage Three
    param: h1_label_mapping: Dictionary mapping to indicate start and end node ids in each pair
    param: node_to_class: dictionary mapping, node_id:node_category
    param: pair_ids: List of h1 graph identifiers for each edge-sampling pair

    returns:
    param: G: Final HiGGs sampled networkx.Graph
    param: cg: Largest component from the HiGGs sampled networkx.Graph

    """
    # Instantiate empty graph
    G = nx.Graph()
    expected_node_count = sum(
        int(nodes.shape[0]) for nodes in h1_node_types)
    node_to_community = {}

    # Relabel each h1 graph and compose it into the final sampled graph
    h1_relabels = []
    pbar_h1s = tqdm(range(len(h1_node_types)))
    for i in pbar_h1s:
        mapping = h1_label_mapping[i]
        shift = mapping["start_ind"]

        adj = h1_adjs[i]
        validate_discrete_graph(
            h1_node_types[i], adj, label=f"Final assembly H1 {i}")

        g = nx.from_numpy_array(adj.cpu().numpy())
        g = nx.convert_node_labels_to_integers(g, first_label=shift)
        G = nx.compose(G, g)
        for node_id in g.nodes():
            node_to_community[int(node_id)] = int(i)

        h1_relabels.append(g)

    # Get edges from Stage Three (inter-h1 edges)
    all_edges, pair_details = inter_edge_specific(
        h1_relabels, out_nx, pair_ids, return_details=True)
    base_edges = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in G.edges()
    }
    for pair_detail in pair_details:
        expected_pair = set(pair_detail["pair_id"])
        for left, right in pair_detail["new_edges"]:
            actual_pair = {
                node_to_community[int(left)],
                node_to_community[int(right)],
            }
            if actual_pair != expected_pair:
                raise RuntimeError(
                    "A Glue pair output was routed to another community pair: "
                    f"expected={sorted(expected_pair)}, "
                    f"actual={sorted(actual_pair)}, edge={(left, right)}")

    # Add these edges to the final graph
    print(f"Before adding IC edges G is a {G}")
    G.add_edges_from(all_edges)
    print(f"After adding IC edges G is a {G}")

    # Try and add a node category for each sampled node (-1. if none is found due to some bug, this is v rare)
    for n in tqdm(list(G.nodes())):
        try:
            G.nodes[n]["target"] = node_to_class[n]
        except:
            G.nodes[n]["target"] = -1.

    # Relabel G
    G = nx.convert_node_labels_to_integers(G)
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    cg = CGs[0]
    cg = nx.relabel_nodes(cg, mapping=lambda x: int(x))

    print(f"HiGGs sampled {G}, largest component is a {cg}")

    final_edges = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in G.edges()
    }
    missing_intra_edges = sorted(base_edges - final_edges)
    added_edges = sorted(final_edges - base_edges)
    invalid_added_edges = [
        edge for edge in added_edges
        if node_to_community[edge[0]] == node_to_community[edge[1]]
    ]
    if strict_assertions:
        if not isinstance(G, nx.Graph) or isinstance(G, nx.MultiGraph):
            raise RuntimeError("Final HiGGs output must be a networkx.Graph")
        if nx.number_of_selfloops(G) != 0:
            raise RuntimeError("Final HiGGs output contains a self-loop")
        if list(G.nodes()) != list(range(G.number_of_nodes())):
            raise RuntimeError("Final HiGGs node IDs are not contiguous")
        if G.number_of_nodes() != expected_node_count:
            raise RuntimeError(
                "Final node count differs from the sum of used H1 nodes")
        if missing_intra_edges:
            raise RuntimeError(
                f"Glue removed H1 intra edges: {missing_intra_edges[:10]}")
        if invalid_added_edges:
            raise RuntimeError(
                "Glue introduced an edge inside one H1 community: "
                f"{invalid_added_edges[:10]}")
        if len({tuple(sorted((int(a), int(b)))) for a, b in pair_ids}) != len(
                pair_ids):
            raise RuntimeError("Glue pair IDs contain duplicates")
    audit = {
        "glue_nodes_before": expected_node_count,
        "glue_nodes_after": G.number_of_nodes(),
        "glue_added_edge_count": len(added_edges),
        "intra_h1_edges_changed": len(missing_intra_edges),
        "invalid_intra_added_edge_count": len(invalid_added_edges),
        "pair_details": pair_details,
        "self_loop_count": nx.number_of_selfloops(G),
        "final_nodes": G.number_of_nodes(),
        "final_edges": G.number_of_edges(),
        "final_components": nx.number_connected_components(G),
        "full_graph_nodes": G.number_of_nodes(),
        "final_lcc_nodes": cg.number_of_nodes(),
        "all_assertions_passed": True,
    }
    if return_audit:
        return G, cg, audit
    return G, cg
