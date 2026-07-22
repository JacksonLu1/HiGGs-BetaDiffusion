"""
Implementation of HiGGs (Hierarchical Generation of Graphs) for NeurIPS 23 by anonymous authors
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)

import warnings
import torch
import networkx as nx
torch.multiprocessing.set_sharing_strategy('file_system')

import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from visualisation import *
from sampling import *
from higgs_utils import *

pwd = os.getcwd()
dgd_dir = os.path.join(pwd, "dgd")
sys.path.insert(0, dgd_dir)
import utils

# Combination of libraries throws various user warnings, silenced here
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.chdir("higgs")

def sample_large_graph(cfg, dataset, h1_batch_size=1, glue_batch_size=1,
                       graph_number=0, visualize=True, sampling_metrics=True,
                       save_tensors=False, save_extras=False,
                       intermediate_figures=False):
    """
    Sample a large graph using HiGGs

    params:
    param: cfg: omegaconf.DictConfig, config dictionary
    param: dataset: name of dataset being replicated (currently sbm, cora, fb_hierarchies)
    param: h1_batch_size: Batch size to use when sampling h1 graphs (HiGGs Stage One)
    param: glue_batch_size: What batch size to use during inter-h1 edge sampling (HiGGs Stage Three)
    param: graph_number: identifier for which graph is being generated, used in file saving
    param: visualize: Whether to visualize the result (on large graphs this is a comparatively small overhead)
    param: sampling_metrics: Whether to compute sampling metrics (on large graphs comparatively costly as we compute eccentricities)
    param: save_tensors: Whether to save pytorch tensors of nodes and edges
    param: save_extras: Whether to also save files for each stage of HiGGs, eg h1 graphs

    returns:
    param: G: sampled networkx.Graph. Node and edge categories should be under "color" and "weight" attributes.
    param: cg: Largest component from previous.
    """

    # Get the h2 template graph (Stage One)
    h2_graph = get_h2_graph(
        cfg, intermediate_figures=intermediate_figures,
        graph_number=graph_number)

    # Store the node categories and the edges between them
    h1_graph_types = h2_graph[0][0]
    h2_graph_adj   = h2_graph[0][1]

    # Get h1 graphs for each node in h2 (Stage Two)
    # h1_label_mapping gives keys for relabelling later - in order that they were generated
    h1_adjs, h1_node_types, h1_label_mapping, node_to_class, n_nodes_counter = get_h1_graphs(
        cfg, h1_graph_types, h1_batch_size)

    sampled_h2_node_count = int(h1_graph_types.shape[0])
    sampled_h1_count = len(h1_adjs)
    if sampled_h1_count != sampled_h2_node_count:
        raise RuntimeError(
            "H2-to-H1 mapping mismatch: "
            f"sampled_h1_count={sampled_h1_count}, "
            f"sampled_h2_node_count={sampled_h2_node_count}")
    if len(h1_node_types) != sampled_h1_count:
        raise RuntimeError("H1 adjacency/node-type counts do not match")
    if len(h1_label_mapping) != sampled_h1_count:
        raise RuntimeError("H1 community mapping count does not match H2 nodes")
    expected_community_ids = set(range(sampled_h2_node_count))
    actual_community_ids = set(range(len(h1_label_mapping)))
    if actual_community_ids != expected_community_ids:
        raise RuntimeError("H2-to-H1 community IDs are missing or duplicated")
    assembled_h1_nodes = sum(int(nodes.shape[0]) for nodes in h1_node_types)
    if assembled_h1_nodes != len(node_to_class):
        raise RuntimeError("H1 local/global node mapping is incomplete")


    # Get edges from h2 into a more useful [n1, n2] form
    # Sparse representation, (2 * n_nodes), E[:,i] is an edge [id1, id2]
    h2_edge_index, h2_edge_type = dense_to_sparse(h2_graph_adj)

    # Sample inter-h1 edges (Stage Three), this outputs a big set of networkx.Graphs
    glue_cfg, glue_model = get_glue_model(cfg)
    datalist, loader, batch_size, n_batches, remainder, pair_ids = prepare_glue_dataset(cfg, glue_model, h1_adjs, h1_node_types, h2_edge_index, h2_edge_type, glue_batch_size)
    routed_pairs = [tuple(sorted((int(left), int(right))))
                    for left, right in pair_ids]
    if len(routed_pairs) != len(set(routed_pairs)):
        raise RuntimeError("Original Glue received a duplicate community pair")
    expected_pairs = {
        (left, right)
        for left in range(sampled_h2_node_count)
        for right in range(left + 1, sampled_h2_node_count)
        if int(h2_graph_adj[left, right]) != 0
    }
    if set(routed_pairs) != expected_pairs:
        raise RuntimeError(
            "Original Glue pair routing differs from H2 present meta-edges: "
            f"expected={sorted(expected_pairs)}, actual={sorted(routed_pairs)}")
    out_nx = get_inter_edges(
        cfg, glue_model, loader, n_batches, batch_size, remainder)


    # Use the pairs of connected graphs to produce one large sampled networkx.Graph
    G, cg = build_from_components(h1_node_types, h1_adjs, out_nx, h1_label_mapping, node_to_class, pair_ids)
    if G.number_of_nodes() != assembled_h1_nodes:
        raise RuntimeError(
            "Glue changed the assembled node count: "
            f"before={assembled_h1_nodes}, after={G.number_of_nodes()}")
    if nx.number_of_selfloops(G) != 0:
        raise RuntimeError("Final HiGGs graph contains self-loops")
    for community_id, adjacency in enumerate(h1_adjs):
        start = h1_label_mapping[community_id]["start_ind"]
        edge_index = torch.triu(adjacency, diagonal=1).nonzero()
        for local_left, local_right in edge_index.tolist():
            if not G.has_edge(start + local_left, start + local_right):
                raise RuntimeError(
                    "Original Glue modified an H1 intra-community edge")

    # Dump graph into .pkl - loading back for your own purposes just pickle.load("path/to/filename.pkl")
    with open(f"sampled_graph_{graph_number}.pkl", "wb") as handle:
        pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if save_tensors:
        # Save pytorch files - networkx representations are lighter, and more useable, but might be useful for downstream tasks
        edges_with_attr = list(G.edges(data=True))
        all_edges = torch.Tensor([[e[0], e[1]] for e in edges_with_attr])
        all_nodes = torch.Tensor([node[1]["target"] for node in G.nodes(data=True)])

        torch.save(torch.Tensor(all_edges).t(), f"sampled_edgelist_{graph_number}.pt")
        torch.save(all_nodes, f"sampled_nodes_{graph_number}.pt")

        edge_keys = list(edges_with_attr[0][2].keys())
        print(f"Possible edge category labels {edge_keys}")

        if len(edge_keys) == 0:
            pass
        else:
            try:
                if "weight" in edge_keys:
                    all_edge_attr = torch.Tensor([e[2]["weight"] for e in edges_with_attr])
                elif "color" in edge_keys:
                    all_edge_attr = torch.Tensor([e[2]["color"] for e in edges_with_attr])
                else:
                    arbitrary_choice = edge_keys[0]
                    all_edge_attr = torch.Tensor([e[2][arbitrary_choice] for e in edges_with_attr])

                torch.save(all_edge_attr, f"sampled_edge_attr_{graph_number}.pt")
            except:
                pass

    # Save each h1 graph and edge-sampled pair if specified
    if save_extras:
        h1_joined = vis_h1_group(h1_adjs, graph_number=graph_number)
        with open(f"sampled_h1_graphs_{graph_number}.pkl", "wb") as handle:
            pickle.dump(h1_joined, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del h1_joined

        glue_joined = vis_glue_group(out_nx, graph_number=graph_number)
        with open(f"sampled_glue_graphs_{graph_number}.pkl", "wb") as handle:
            pickle.dump(glue_joined, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del glue_joined

    # Need original graph for visualisations and sampling metrics
    if visualize or sampling_metrics:
        original_G = load_original(root_path=cfg.general.root_path, name=dataset).copy()

    if visualize:
        vis_big_graphs(original_G, G, cg, graph_number=graph_number)

    if sampling_metrics:
        sampling_quality(original_G, cg)

    return G, cg

@hydra.main(version_base='1.1', config_path='../configs', config_name='sample') #
def main(cfg: DictConfig):
    """
    Script to produce graphs using HiGGs.
    Parameters are principally loaded from config file, given with general=config_file, for example
    `python higgs/sample_main.py general=sample_cora`
    Outputs are under higgs/outputs/...

    params:
    param: cfg: omegaconf.DictConfig, config dictionary for sampling run
    """

    print("\nEntered sampling main")
    # Loading models can mess with config keys, so store some config parameters here to pass on each iteration
    dataset = cfg.general.name
    n_graphs = cfg.general.samples_to_generate
    h1_batch_size = cfg.general.h1_batch_size
    glue_batch_size = cfg.general.glue_batch_size
    visualize = bool(omegaconf.OmegaConf.select(cfg, "general.visualize", default=False))
    intermediate_figures = bool(omegaconf.OmegaConf.select(
        cfg, "general.intermediate_figures", default=False))

    print(f"Dataset: {dataset}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    print(f"Intermediate figures: {intermediate_figures}")
    print(f"Final visualization: {visualize}")

    audit_checkpoint_resolutions(
        cfg.general.h2_model,
        cfg.general.h1_model,
        cfg.general.glue_model,
    )

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    os.mkdir("sampling")
    os.chdir("sampling")

    for i in range(n_graphs):
        G_sampled, cg = sample_large_graph(cfg, dataset, h1_batch_size = h1_batch_size, glue_batch_size=glue_batch_size,
                                           graph_number=i, visualize=visualize, sampling_metrics=False,
                                           save_tensors=False, save_extras=False,
                                           intermediate_figures=intermediate_figures)




if __name__ == '__main__':
    print("Get past imports")
    main()
