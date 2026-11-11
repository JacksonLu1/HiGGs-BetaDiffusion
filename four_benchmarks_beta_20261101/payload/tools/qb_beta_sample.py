"""Full-chain sampling for four frozen structure-only benchmarks.

Only this process is adapted. Existing HiGGs model, diffusion and sampling
sources are not rewritten. Full H2/H1 components and every requested sample
are retained, matching the original-HiGGs arm for these new datasets.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

from qb_beta_common import ROOT, SPECS, STAGES, baseline, check_hashes, get_item, sha


DATASETS = tuple(SPECS)
CONDITION_MODE = "four_benchmark_structure_only"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_reports(training, name, digest, report_mode='formal'):
    """Accept only complete formal training from the same frozen collection."""
    reports = {}
    for stage, (h, _, monitor) in STAGES.items():
        path = training / stage / (report_mode + "_result.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        require(report["stage"] == stage and report["dataset"] == name, "Wrong dataset/stage in " + str(path))
        require(report["train_seed"] == 0 and report["batch_size"] == 32,
                "Wrong training seed or batch size")
        require(report["epoch"] == (SPECS[name]["epochs"] - 1 if report_mode == "formal" else 0) and report["global_step"] > 0,
                "Requested training budget has not completed")
        require(report["monitor"] == monitor, "Wrong validation monitor")
        require(report["manifest_sha256"] == digest, "Training manifest differs")
        require(report["strict_reload"] and report["strict_best_reload"],
                "Training did not pass strict best/last reload")
        best = Path(report["best"]).resolve()
        require(training.resolve() in best.parents and best.is_file(),
                "Checkpoint is outside the specified training experiment")
        require(sha(best) == report["best_sha256"], "Best checkpoint changed")
        last = Path(report["last"]).resolve()
        require((training / stage).resolve() in last.parents and sha(last) == report["last_sha256"],
                "Last checkpoint changed or is outside its stage")
        reports[stage] = report
    return reports


def fixed_h1_conditions(node_types, num_classes, mode, *, dataset_name):
    """Dummy structure-only H2 node categories always condition H1 on class 0."""
    import torch

    require(dataset_name in DATASETS and mode == CONDITION_MODE,
            "The structure-only condition adapter is scoped to these four datasets")
    require(node_types.ndim == 1 and node_types.dtype == torch.long,
            "Expected one-dimensional H2 node categories")
    require(num_classes == 2, "Frozen H1 cache requires two y channels")
    result = torch.zeros((len(node_types), num_classes), dtype=torch.long,
                         device=node_types.device)
    result[:, 0] = 1
    return result


def prepare_pairs(cfg, model, h1_adjs, h1_nodes, edge_index, edge_types,
                  batch_size, return_audit=False, *, interface, cache_audit):
    """Create ordinary Glue inputs with actual-size membership padding.

    Both comparison arms use actual maximum pair size (not 700). This changes
    allocation only, not masks, y classes, valid nodes/edges or the model.
    A generated edgeless H2 is assembled without Glue rather than rejected.
    """
    from collections import Counter
    import torch
    from torch.nn import functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import dense_to_sparse

    require(cfg.dataset.name in DATASETS, "Unexpected collection")
    require(batch_size == 1, "Use the original comparison Glue batch size 1")
    require((model.Xdim_output, model.Edim_output, model.ydim_output) == (1, 2, 2),
            "Unexpected structure-only Glue output dimensions")
    records = list(interface.unique_undirected_pairs(edge_index, edge_types))
    require(all(category == 1 for _, _, category in records),
            "H2 present edges must be binary class 1")
    require(all(0 <= left < len(h1_adjs) and 0 <= right < len(h1_adjs)
                for left, right, _ in records), "H2 pair index out of bounds")
    maximum = max((len(h1_nodes[left]) + len(h1_nodes[right])
                   for left, right, _ in records), default=0)
    device = "cuda" if torch.cuda.is_available() and cfg.general.gpus > 0 else "cpu"
    pairs, samples = [], []
    for left, right, category in records:
        a_nodes, b_nodes = h1_nodes[left], h1_nodes[right]
        require(torch.all(a_nodes == 0).item() and torch.all(b_nodes == 0).item(),
                "Structure-only H1 must have dummy node category 0")
        a_edges, a_attr = dense_to_sparse(h1_adjs[left])
        b_edges, b_attr = dense_to_sparse(h1_adjs[right])
        n_left, n_right = len(a_nodes), len(b_nodes)
        n = n_left + n_right
        require(n_left > 0 and n_right > 0, "Empty H1 community")
        side = torch.arange(n) < n_left
        pos = side[:, None] == side[None, :]
        pos = F.pad(pos, (0, maximum - n, 0, maximum - n), value=False)
        attr = torch.cat((a_attr, b_attr)).long()
        require(not attr.numel() or torch.all(attr == 1).item(),
                "H1 present edges must be class 1")
        sample = Data(
            x=torch.ones((n, 1), dtype=torch.float),
            edge_index=torch.cat((a_edges, b_edges + n_left), dim=-1),
            edge_attr=F.one_hot(attr, num_classes=2).float(),
            pos=pos.bool(),
            y=torch.tensor([[0.0, 1.0]]),
            pair_order=torch.tensor([len(samples)], dtype=torch.long),
        ).to(device)
        samples.append(sample)
        pairs.append((left, right))
    print("Small collection Glue: pairs={}, membership_padding={} (actual maximum); "
          "empty H2 is retained without resampling".format(len(pairs), maximum), flush=True)
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False) if samples else []
    raw_histogram = {str(k): int(v) for k, v in sorted(Counter(
        category for _, _, category in records).items())}
    audit = {
        "condition_mode": "direct",
        "raw_h2_edge_type_histogram": raw_histogram,
        "glue_condition_histogram_after_adapter": {"0,1": len(pairs)} if pairs else {},
        "verified_glue_training_y_histogram": cache_audit["one_hot_histogram"],
        "glue_training_cache": cache_audit,
        "pair_ids": [list(pair) for pair in pairs],
        "membership_padding": maximum,
        "empty_h2_policy": "assemble all H1; no resampling or filtering",
    }
    result = (samples, loader, batch_size, len(samples), 0, pairs)
    return result + (audit,) if return_audit else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--report-mode", choices=("smoke", "formal"), default="formal")
    args = parser.parse_args()
    root = args.root.resolve()
    require(root == ROOT, "Only the modified Linux experiment root is allowed")
    training, out = args.training_output.resolve(), args.output.resolve()
    require(root / "outputs" in training.parents, "Wrong training experiment directory")
    require(out.name in ("sample", "smoke_sample") and (root / "outputs" in out.parents or root / "higgs/outputs" in out.parents) and not out.exists(),
            "Use a new sample subdirectory under the modified sampling experiment")
    item = get_item(root, args.dataset)
    before = baseline(root, args.dataset)
    frozen = json.loads(Path(item["manifest"]).read_text(encoding="utf-8"))
    count = len(frozen["splits"]["test"]) if args.count is None else args.count
    require(count in (1, len(frozen["splits"]["test"])),
            "Choose one exploratory graph or the complete frozen test-set count")
    audit_path = training / "input_hash_check.json"
    training_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(training_audit["unchanged"] is True, "Training has not passed final input audit")
    check_hashes(training_audit["sha256"])
    before.update(training_audit["sha256"])
    before[str(audit_path)] = sha(audit_path)
    reports = read_reports(training, args.dataset, item["manifest_sha256"], args.report_mode)
    for stage, report in reports.items():
        report_path = training / stage / (args.report_mode + "_result.json")
        before[str(report_path)] = sha(report_path)
        before[str(Path(report["best"]).resolve())] = report["best_sha256"]

    import fcntl
    lock_path = root / "logs/sampling" / (args.dataset + "_mean_hog_fullchain.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    import os
    os.chdir(root)
    sys.path[:0] = [str(root / "dgd"), str(root / "higgs"), str(root)]
    import torch
    import networkx as nx
    from omegaconf import OmegaConf
    from four_benchmark_models import model_class
    import sample_main as sampling_entry
    import sampling
    import higgs_utils
    import interface_ablation as interface

    require(torch.cuda.is_available(), "CUDA unavailable; CPU sampling is not launched")
    torch.cuda.set_device(0)
    require(torch.cuda.mem_get_info()[0] >= 8 * 1024**3, "Less than 8 GiB free memory")
    print("Physical GPU:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("Logical GPU: cuda:0", torch.cuda.get_device_name(0), flush=True)
    classes = {stage: model_class(stage) for stage in STAGES}
    stage_keys = {float(value[0]): key for key, value in STAGES.items()}
    checkpoint_cfgs = {}
    for stage, report in reports.items():
        state = higgs_utils._load_checkpoint_cpu(report["best"])
        saved = state["hyper_parameters"]["cfg"]
        cfg = saved if OmegaConf.is_config(saved) else OmegaConf.create(saved)
        require(cfg.dataset.name == args.dataset, "Wrong checkpoint dataset")
        require(float(cfg.dataset.h) == float(STAGES[stage][0]), "Wrong checkpoint stage")
        require(float(cfg.dataset.resolution) == item["resolution"], "Wrong checkpoint resolution")
        require(Path(cfg.dataset.manifest).resolve() == Path(item["manifest"]).resolve(),
                "Checkpoint does not use the frozen shared manifest")
        require(int(cfg.train.n_epochs) == (SPECS[args.dataset]["epochs"] if args.report_mode == "formal" else 1) and int(cfg.train.batch_size) == 32,
                "Formal comparison requires the frozen dataset-specific epochs and batch32")
        require(int(cfg.train.seed) == 0, "Training seed differs from original arm")
        require(cfg.general.resume is None and cfg.general.test_only is None,
                "Expected fresh training rather than resumed/finetuned weights")
        family = {"h2": "beta_h2", "h1": "beta_h1", "glue": "original_glue"}[stage]
        higgs_utils._require_checkpoint_identity(
            report["best"], family, float(STAGES[stage][0]), stage.upper(),
            allowed_families={family, "digress"} if stage == "glue" else None)
        if stage == "h1":
            require(state["h1_two_hop"]["normalization"] == "mean", "Wrong two-hop mode")
            require(state["h1_two_hop"]["warm_start"] is None, "Unexpected H1 warm start")
            require(state["h1_hog_rw"]["rw_depth"] == 4 and state["h1_hog_rw"]["two_hop"] is True,
                    "Wrong HOG feature configuration")
        checkpoint_cfgs[stage] = copy.deepcopy(cfg)
        del state

    glue_cache = item["cache_paths"]["1.5"]["train"]
    cache_audit = interface.audit_glue_training_cache(str(glue_cache))
    require(cache_audit["verified_present_class_index"] == 1,
            "Frozen Glue cache must condition on class1")
    interface._CONDITION_MODES.add(CONDITION_MODE)
    sampling.adapt_h2_to_h1_conditions = fixed_h1_conditions

    def own_checkpoint_sampler(request):
        stage = stage_keys[float(request.dataset.h)]
        cfg = copy.deepcopy(checkpoint_cfgs[stage])
        model = classes[stage].load_from_checkpoint(
            reports[stage]["best"], map_location="cpu", cfg=cfg, strict=True)
        require(model.dataset_info.manifest_sha256 == item["manifest_sha256"],
                "Checkpoint dataset-info manifest differs")
        require(list(model.dataset_info.training_graph_ids) == frozen["splits"]["train"],
                "Checkpoint training graph IDs differ")
        require(type(model.node_dist).__module__ == "datasets.four_benchmark_beta",
                "Unexpected node-count prior class")
        if stage == "h1":
            widths = {key: getattr(model.model, "mlp_in_" + key)[0].in_features
                      for key in ("X", "E", "y")}
            require(widths == {"X": 12, "E": 6, "y": 14}, "HOG input widths differ")
            require(model.cfg.model.two_hop.normalization == "mean" and model.cfg.model.hog_rw.enabled,
                    "H1 architecture differs after strict loading")
        model.eval()
        model.freeze()
        runtime = copy.deepcopy(model.cfg)
        OmegaConf.set_struct(runtime, False)
        runtime.general = copy.deepcopy(request.general)
        runtime.general.test_only = reports[stage]["best"]
        return runtime, model

    sampling.get_samplers = own_checkpoint_sampler

    def own_pairs(cfg, model, adjs, nodes, edge_index, edge_types,
                  batch_size, return_audit=False):
        return prepare_pairs(cfg, model, adjs, nodes, edge_index, edge_types,
                             batch_size, return_audit, interface=interface,
                             cache_audit=cache_audit)

    sampling_entry.prepare_glue_dataset = own_pairs
    cfg = OmegaConf.create({
        "dataset": OmegaConf.to_container(checkpoint_cfgs["h2"].dataset, resolve=True),
        "model": OmegaConf.to_container(checkpoint_cfgs["h2"].model, resolve=True),
        "train": OmegaConf.to_container(checkpoint_cfgs["h2"].train, resolve=True),
        "general": {
            "name": args.dataset, "gpus": 1, "wandb": "disabled",
            "root_path": str(root / "data"),
            "h2_model": reports["h2"]["best"], "h2_model_family": "beta_h2",
            "h1_model": reports["h1"]["best"], "h1_model_family": "beta_h1",
            "glue_model": reports["glue"]["best"], "glue_model_family": "original_glue",
            "samples_to_generate": count, "sampling_seed": 1234,
            "h1_batch_size": 4, "glue_batch_size": 1,
            "visualize": False, "intermediate_figures": False,
            "interface": {
                "h2_component_policy": "full", "h1_component_policy": "full",
                "h2_to_h1_condition_mode": CONDITION_MODE,
                "h2_to_glue_condition_mode": "direct", "glue_present_class_index": 1,
                "glue_training_cache": str(glue_cache), "force_eval": True,
                "save_intermediates": True, "strict_assertions": True,
            },
        },
    })
    interface.interface_options(cfg)
    check_hashes(before)
    out.mkdir(parents=True)
    (out / ".hydra").mkdir()
    OmegaConf.save(cfg, out / ".hydra/config.yaml")
    (out / "input_manifest.json").write_text(json.dumps({
        "dataset": args.dataset, "training_run": str(training),
        "manifest_sha256": item["manifest_sha256"], "checkpoints": reports,
        "source_sha256": before, "requested_samples": count,
        "real_test_graphs": len(frozen["splits"]["test"]),
        "exploratory": count != len(frozen["splits"]["test"]),
        "protocol": "seed1234 once; strict saved-config loading; full H2/H1 retained",
        "h1_condition": "fixed frozen training class0",
        "glue_condition": "fixed frozen training class1",
        "glue_padding": "actual maximum pair size, as original new-dataset arm",
        "empty_h2": "assemble all H1; no filtering/resampling",
    }, indent=2), encoding="utf-8")
    samples_dir = out / "sampling"
    samples_dir.mkdir()
    os.chdir(samples_dir)
    interface.seed_sampling(1234)
    rows = []
    for index in range(count):
        print("\nSTART {} GRAPH {}/{}".format(args.dataset.upper(), index + 1, count), flush=True)
        graph, largest = sampling_entry.sample_large_graph(
            cfg, args.dataset, h1_batch_size=4, glue_batch_size=1,
            graph_number=index, visualize=False, sampling_metrics=False,
            save_tensors=False, save_extras=False, intermediate_figures=False)
        require(isinstance(graph, nx.Graph) and not graph.is_directed() and not graph.is_multigraph(),
                "Expected simple undirected generated graph")
        require(len(graph) > 0 and nx.number_of_selfloops(graph) == 0,
                "Generated graph is empty or has loops")
        path = samples_dir / ("sampled_graph_{}.pkl".format(index))
        require(path.is_file(), "Sampler did not save the expected graph")
        rows.append({"index": index, "nodes": len(graph), "edges": graph.number_of_edges(),
                     "components": nx.number_connected_components(graph),
                     "lcc_nodes": len(largest), "lcc_edges": largest.number_of_edges(),
                     "sha256": sha(path)})
        (out / "sampling_progress.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("SAVED:", rows[-1], flush=True)
    require(len(list(samples_dir.glob("sampled_graph_*.pkl"))) == count,
            "Saved graph count differs from the requested protocol")
    check_hashes(before)
    with (out / "sampling_structure.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "input_hash_check.json").write_text(json.dumps(
        {"unchanged": True, "sha256": before}, indent=2), encoding="utf-8")
    print("PASS: {}_MEAN_HOG_FULLCHAIN_{}_COMPLETE".format(args.dataset.upper(), count), flush=True)
    print("OUTPUT:", out, flush=True)
    print("Sampling completed; generation quality has not yet been evaluated.", flush=True)


if __name__ == "__main__":
    main()




