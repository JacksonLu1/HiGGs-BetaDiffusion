
import csv
import fcntl
import gc
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path
from statistics import mean

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()

os.chdir(root)
sys.path[:0] = [str(root / "dgd"), str(root)]

import networkx as nx
import numpy as np
import torch
from omegaconf import OmegaConf
from datasets.sbm_hierarchies import SBMDataset
from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1
from higgs.higgs_utils import _load_checkpoint_cpu
from higgs.interface_ablation import (
    apply_upstream_lcc,
    validate_discrete_graph,
)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)

def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def graph_stats(adj):
    graph = nx.from_numpy_array(adj.cpu().numpy())
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    components = nx.number_connected_components(graph)
    return {
        "nodes": n,
        "edges": m,
        "density": nx.density(graph),
        "mean_degree": 2 * m / n,
        "max_degree": max(dict(graph.degree()).values()),
        "clustering": nx.average_clustering(graph),
        "components": components,
        "connected": int(components == 1),
    }

require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Expected physical GPU 0")
require(torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback")
require(not out.exists(), "Output already exists: " + str(out))

lock_handle = (root / "logs/evaluation/sbm_h1_threeway_mean_gpu0.lock").open("a+")
try:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError("Another SBM three-way evaluation is already running")

torch.cuda.set_device(0)
print("Physical GPU 0 -> cuda:0:", torch.cuda.get_device_name(0), flush=True)

sqrt_run = (
    root / "outputs"
    / "sbm_beta_h1_twohop_r1_e500_scratch_seed0_20261016_205246_cacheonly"
)
mean_run = (
    root / "outputs"
    / "sbm_beta_h1_twohop_mean_r1_e500_scratch_seed0_20261017_202904"
)

mean_path = (
    mean_run / "checkpoints" / mean_run.name / "epoch=149.ckpt"
)
training_report = mean_run / "formal_training_check.json"
require(training_report.is_file(), "Missing mean training completion report")

with training_report.open(encoding="utf-8") as handle:
    report = json.load(handle)

require(report.get("passed") is True, "Mean training report did not pass")
require(
    Path(report["best_checkpoint"]).resolve() == mean_path,
    "Mean checkpoint does not match the recorded validation-best checkpoint",
)

specs = [
    {
        "label": "old_beta_h1",
        "mode": "disabled",
        "path": root / "saved_models/sbm/sbm_beta_h1_r1_best.ckpt",
        "sha256": "c69446b3b1b928351ad18a4e8847a429924356c2f6be6c5145af277cb540df99",
        "epoch": 359,
    },
    {
        "label": "sqrt_twohop_h1",
        "mode": "sqrt",
        "path": sqrt_run / "checkpoints" / sqrt_run.name / "epoch=359.ckpt",
        "sha256": "a12589310a6dba3e7b0ae6ff275842da51c7f71a8a16740077eb44c4e3247263",
        "epoch": 359,
    },
    {
        "label": "mean_twohop_h1",
        "mode": "mean",
        "path": mean_path,
        "sha256": report["best_sha256"],
        "epoch": 149,
    },
]

data_dir = root / "data/sbm"
cache_paths = [
    data_dir / "processed" / (
        "proc_" + split + "_h1_r1_majority_max500_bs8.pt"
    )
    for split in ("train", "val", "test")
]
expected_paths = set(cache_paths)

for path in cache_paths:
    require(path.is_file() and path.stat().st_size > 0, "Missing cache: " + str(path))

for spec in specs:
    require(spec["path"].is_file(), "Missing checkpoint: " + str(spec["path"]))
    require(
        sha256(spec["path"]) == spec["sha256"],
        "Checkpoint SHA256 mismatch: " + spec["label"],
    )

input_paths = cache_paths + [spec["path"] for spec in specs] + [
    root / "dgd/models/two_hop_edge.py",
    root / "dgd/models/h1_twohop_transformer.py",
    root / "dgd/diffusion_model_beta_h1.py",
    root / "higgs/interface_ablation.py",
]
before = {str(path): sha256(path) for path in input_paths}

cache_manifest = mean_run / "cache_manifest.json"
with cache_manifest.open(encoding="utf-8") as handle:
    training_caches = json.load(handle)

require(training_caches.get("unchanged") is True, "Training cache check failed")
for path in cache_paths:
    require(
        training_caches["after"].get(str(path)) == before[str(path)],
        "Cache differs from the mean training run: " + str(path),
    )

def cache_only_guard(self, *args, **kwargs):
    actual = {Path(path).resolve() for path in self.processed_paths}
    require(float(self.h) == 1, "Expected H1")
    require(Path(self.root).resolve() == data_dir, "Unexpected dataset root")
    require(actual == expected_paths, "Unexpected processed cache paths")
    for path in actual:
        require(
            path.is_file() and path.stat().st_size > 0,
            "Missing existing cache: " + str(path),
        )

def forbid_processing(self, *args, **kwargs):
    raise RuntimeError("Downloading/repartitioning/processing is forbidden")

SBMDataset._download = cache_only_guard
SBMDataset._process = cache_only_guard
SBMDataset.download = forbid_processing
SBMDataset.process = forbid_processing

dataset = SBMDataset(
    stage="val",
    root=str(data_dir),
    h=1,
    resolution=1,
    max_size=500,
    batch_size=8,
    n_workers=0,
    regime="majority",
)
require(len(dataset) == 108, "Expected exactly 108 validation graphs")

records = []
for index in range(len(dataset)):
    data = dataset[index]
    n = int(data.num_nodes)
    require(n > 0, "Empty validation graph")
    require(data.x.shape == (n, 1), "Unexpected SBM node encoding")
    require(data.edge_attr.shape[1] == 2, "Unexpected edge encoding")
    require(
        bool(torch.all(data.edge_attr.argmax(-1) == 1)),
        "Unexpected absent-edge entries in sparse H1 cache",
    )

    real = torch.zeros(n, n, dtype=torch.long)
    real[data.edge_index[0], data.edge_index[1]] = 1
    validate_discrete_graph(
        data.x.argmax(-1).long(), real, label="real H1"
    )

    condition = data.y.detach().clone().float().reshape(1, -1)
    require(condition.shape == (1, 2), "Unexpected condition shape")
    require(bool(torch.isfinite(condition).all()), "Nonfinite condition")
    require(
        bool(torch.all((condition == 0) | (condition == 1)))
        and float(condition.sum()) == 1.0,
        "Condition must be one-hot",
    )

    records.append({
        "index": index,
        "n": n,
        "condition": condition,
        "real_stats": graph_stats(real),
    })

print("Validation graphs:", len(records), flush=True)
print("Protocol: batch_size=1; fixed n/y; seed=graph_index (0..107)", flush=True)

out.mkdir(parents=True, exist_ok=False)
save_json(out / "input_manifest.json", {
    "input_sha256_before": before,
    "physical_gpu": 0,
    "validation_count": 108,
    "seeds": list(range(108)),
})

rows = []
metadata = {}
original_model_cfg = None

with (out / "per_graph.csv").open("x", newline="", encoding="utf-8") as handle:
    writer = None

    for spec in specs:
        label = spec["label"]
        path = spec["path"]
        checkpoint = _load_checkpoint_cpu(path)
        cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])

        for name, expected in (
            ("dataset.name", "sbm"),
            ("dataset.h", 1),
            ("dataset.resolution", 1),
            ("dataset.max_size", 500),
            ("dataset.regime", "majority"),
            ("model.family", "beta_h1"),
        ):
            require(OmegaConf.select(cfg, name) == expected, label + ": " + name)

        require(checkpoint["epoch"] == spec["epoch"], label + ": wrong epoch")

        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        two_hop = model_cfg.pop("two_hop", None) or {}
        enabled = bool(two_hop.get("enabled", False))
        require(enabled == (spec["mode"] != "disabled"), label + ": wrong two-hop flag")

        if enabled:
            require(two_hop.get("channels") == 16, label + ": wrong channel count")
            require(
                two_hop.get("normalization", "sqrt") == spec["mode"],
                label + ": wrong checkpoint configuration",
            )
            checkpoint_mode = checkpoint.get("h1_two_hop", {}).get(
                "normalization", "sqrt"
            )
            require(checkpoint_mode == spec["mode"], label + ": wrong metadata")

        if original_model_cfg is None:
            original_model_cfg = model_cfg
        else:
            require(
                model_cfg == original_model_cfg,
                "Model settings differ beyond two-hop",
            )

        metadata[label] = {
            "path": str(path),
            "sha256": spec["sha256"],
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
            "normalization": spec["mode"],
        }
        del checkpoint

        model = ConditionalGraphBetaDiffusionH1.load_from_checkpoint(
            str(path), map_location="cpu", strict=True
        ).to("cuda:0").eval()

        require(not any(module.training for module in model.modules()), "Eval mode failed")
        require(model.T == 100, "Unexpected diffusion step count")
        require(model.edge_threshold == 0.5, "Unexpected edge threshold")

        options = model.two_hop_options
        require(options["enabled"] == enabled, "Loaded two-hop flag mismatch")
        if enabled:
            require(
                options.get("normalization") == spec["mode"],
                "Loaded normalization mismatch",
            )
            require(
                getattr(model.model.two_hop_edge, "normalization", None)
                == spec["mode"],
                "Actual two-hop block normalization mismatch",
            )

        print("\nModel:", label, metadata[label], flush=True)
        saved_graphs = []

        for record in records:
            index = record["index"]
            n = record["n"]
            condition = record["condition"]
            require(condition.shape == (1, model.ydim_output), "Condition dimension mismatch")

            seed(index)

            with torch.inference_mode():
                generated, _ = model.sampling_batch_h1(
                    batch_id=index,
                    batch_size=1,
                    keep_chain=0,
                    number_chain_steps=40,
                    num_nodes=n,
                    y=condition.to("cuda:0"),
                    figures=False,
                )

            nodes, full = [
                value.detach().cpu() for value in generated[0]
            ]
            require(nodes.numel() == n and full.shape == (n, n), "Fixed-n mismatch")
            validate_discrete_graph(nodes, full, label=label)

            used_nodes, used = apply_upstream_lcc(nodes, full)

            row = {
                "model": label,
                "graph_index": index,
                "seed": index,
                "condition_y": int(condition.argmax(-1).item()),
                "lcc_retention": used.shape[0] / n,
            }
            for prefix, stats in (
                ("real", record["real_stats"]),
                ("full", graph_stats(full)),
                ("used", graph_stats(used)),
            ):
                row.update({prefix + "_" + name: value for name, value in stats.items()})

            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)

            saved_graphs.append({
                "graph_index": index,
                "seed": index,
                "condition_y": row["condition_y"],
                "full_node_types": nodes.numpy(),
                "full_adjacency": full.numpy(),
                "used_node_types": used_nodes.numpy(),
                "used_adjacency": used.numpy(),
            })

            if (index + 1) % 18 == 0:
                print(label + ": " + str(index + 1) + "/108", flush=True)

        with (out / (label + "_generated_h1.pkl")).open("xb") as graph_file:
            pickle.dump(saved_graphs, graph_file, protocol=pickle.HIGHEST_PROTOCOL)

        del model
        gc.collect()
        torch.cuda.empty_cache()

fields = (
    "nodes", "edges", "density", "mean_degree",
    "max_degree", "clustering", "components", "connected",
)
error_fields = ("edges", "density", "mean_degree", "max_degree", "clustering")

summary = {
    "protocol": "108 validation H1 graphs; fixed n/y; seed=graph_index; batch_size=1",
    "diffusion_steps": 100,
    "edge_threshold": 0.5,
    "validation_count": 108,
    "checkpoints": metadata,
    "real_mean": {
        name: mean(record["real_stats"][name] for record in records)
        for name in fields
    },
    "models": {},
}

for spec in specs:
    label = spec["label"]
    selected = [row for row in rows if row["model"] == label]
    require(len(selected) == 108, "Incomplete results: " + label)

    summary["models"][label] = {
        prefix: {
            name: mean(row[prefix + "_" + name] for row in selected)
            for name in fields
        }
        for prefix in ("full", "used")
    }
    summary["models"][label]["lcc_retention"] = mean(
        row["lcc_retention"] for row in selected
    )
    summary["models"][label]["mae_to_real"] = {
        prefix: {
            name: mean(
                abs(row[prefix + "_" + name] - row["real_" + name])
                for row in selected
            )
            for name in error_fields
        }
        for prefix in ("full", "used")
    }

after = {str(path): sha256(path) for path in input_paths}
save_json(out / "input_hash_check.json", {
    "before": before,
    "after": after,
    "unchanged": before == after,
})
require(before == after, "An input file changed during evaluation")

summary["input_hashes_unchanged"] = True
save_json(out / "summary.json", summary)

print("\nSBM H1 three-way validation generation comparison")
print(
    f"{'data':<25} {'nodes':>8} {'edges':>9} {'density':>10} "
    f"{'clustering':>11} {'maxdeg':>8} {'connected':>10}"
)
table = [("real", summary["real_mean"])]
for label, result in summary["models"].items():
    for prefix in ("full", "used"):
        table.append((label + "/" + prefix, result[prefix]))

for label, value in table:
    print(
        f"{label:<25} {value['nodes']:8.3f} {value['edges']:9.3f} "
        f"{value['density']:10.5f} {value['clustering']:11.5f} "
        f"{value['max_degree']:8.3f} {value['connected']:10.3f}"
    )

print("\nFull-graph mean absolute errors: lower is better")
print(
    f"{'model':<21} {'edge_MAE':>10} {'density_MAE':>12} "
    f"{'degree_MAE':>11} {'maxdeg_MAE':>11} {'clust_MAE':>10} {'LCC':>8}"
)
for label, result in summary["models"].items():
    error = result["mae_to_real"]["full"]
    print(
        f"{label:<21} {error['edges']:10.3f} {error['density']:12.6f} "
        f"{error['mean_degree']:11.6f} {error['max_degree']:11.3f} "
        f"{error['clustering']:10.6f} {result['lcc_retention']:8.4f}"
    )
    print("  Used/LCC MAE:", json.dumps(result["mae_to_real"]["used"]))

print("\nCSV:", out / "per_graph.csv")
print("JSON:", out / "summary.json")
print("PASS: SBM_H1_THREEWAY_VALIDATION_COMPLETE")
print("Same 108 validation graphs/n/y/seeds; strict checkpoint loading; inputs unchanged.")
print("No training, H2/Glue sampling or whole-graph MMD evaluation was performed.")
