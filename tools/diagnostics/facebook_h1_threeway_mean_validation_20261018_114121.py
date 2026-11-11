
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()

os.chdir(root)
sys.path[:0] = [
    str(root / "tools/diagnostics"),
    str(root / "dgd"),
    str(root),
]

import torch
from omegaconf import OmegaConf
import facebook_beta_h1_fixed_validation as base

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)

def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)

require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "Expected physical GPU 1")
require(torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback")
require(not out.exists(), "Output already exists: " + str(out))
require(Path(base.ROOT).resolve() == root, "Unexpected diagnostic ROOT")
require(base.BASE_SEED == 20260825, "Original sampling seed was changed")
require(base.PER_CLASS == 5, "Original per-class selection was changed")

cache = (
    root / "data/fb/processed"
    / "proc_val_h1_r10_majority_max400_bs8.pt"
)
require(
    Path(base.VAL_CACHE).resolve() == cache.resolve(),
    "Unexpected validation cache",
)
require(cache.is_file(), "Missing Facebook H1 validation cache")

lock_handle = (
    root / "logs/evaluation/facebook_h1_threeway_mean_gpu1.lock"
).open("a+")
try:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError("Another Facebook three-way evaluation is already running")

torch.cuda.set_device(0)
free_bytes, _ = torch.cuda.mem_get_info()
print("Physical GPU 1 -> cuda:0:", torch.cuda.get_device_name(0), flush=True)
print("Free GPU memory GiB:", round(free_bytes / 1024**3, 2), flush=True)
require(free_bytes >= 4 * 1024**3, "Less than 4 GiB free GPU memory")

sqrt_run = (
    root / "outputs"
    / "fb_beta_h1_twohop_r10_e500_scratch_seed0_20261016_204432"
)
mean_run = (
    root / "outputs"
    / "fb_beta_h1_twohop_mean_r10_e500_scratch_seed0_20261017_202904"
)

mean_best = (
    mean_run / "checkpoints" / mean_run.name / "epoch=419.ckpt"
)
training_report_path = mean_run / "formal_training_check.json"
training_cache_path = mean_run / "cache_manifest.json"

require(training_report_path.is_file(), "Missing mean training completion report")
require(training_cache_path.is_file(), "Missing mean training cache manifest")
require(mean_best.is_file(), "Missing mean best419 checkpoint")

report = read_json(training_report_path)
require(report.get("passed") is True, "Mean training report did not pass")
require(report.get("dataset") == "facebook", "Wrong training report dataset")
require(report.get("epoch") == 499, "Mean training did not reach epoch499")
require(report.get("global_step") == 123500, "Unexpected mean final step")
require(
    Path(report["best_checkpoint"]).resolve() == mean_best.resolve(),
    "Mean best419 does not match the recorded validation-best checkpoint",
)
require(math.isfinite(float(report["best_score"])), "Nonfinite best score")

specs = [
    {
        "label": "old_beta_best419",
        "mode": "disabled",
        "path": root / "saved_models/facebook/facebook_h1.ckpt",
        "sha256": "7475d4e177c6ffa89371bd775490aeb83e136b482ced54732818da45260926c4",
    },
    {
        "label": "sqrt_twohop_best419",
        "mode": "sqrt",
        "path": sqrt_run / "checkpoints" / sqrt_run.name / "epoch=419.ckpt",
        "sha256": "eae327fe4dcc51d0044b491fa59d06eaa7a3336333224ea0fcb7484e09b959b8",
    },
    {
        "label": "mean_twohop_best419",
        "mode": "mean",
        "path": mean_best,
        "sha256": report["best_sha256"],
    },
]

metadata = {}
original_model_cfg = None

for spec in specs:
    label = spec["label"]
    path = spec["path"]

    require(path.is_file(), "Missing checkpoint: " + str(path))
    require(sha256(path) == spec["sha256"], label + ": SHA256 mismatch")

    checkpoint = base._load_checkpoint_cpu(path)
    cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])

    for name, expected in (
        ("dataset.name", "fb_hierarchies"),
        ("dataset.h", 1),
        ("dataset.resolution", 10),
        ("dataset.max_size", 400),
        ("dataset.regime", "majority"),
        ("model.family", "beta_h1"),
        ("model.diffusion_steps", 100),
        ("model.edge_threshold", 0.5),
    ):
        require(
            OmegaConf.select(cfg, name) == expected,
            label + ": unexpected " + name,
        )

    require(checkpoint["epoch"] == 419, label + ": wrong epoch")
    require(checkpoint["global_step"] == 103740, label + ": wrong global_step")

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    two_hop = model_cfg.pop("two_hop", None) or {}
    enabled = bool(two_hop.get("enabled", False))

    require(
        enabled == (spec["mode"] != "disabled"),
        label + ": wrong two-hop enabled flag",
    )

    if enabled:
        require(two_hop.get("channels") == 16, label + ": wrong channels")
        require(
            two_hop.get("normalization", "sqrt") == spec["mode"],
            label + ": wrong normalization configuration",
        )
        require(
            checkpoint.get("h1_two_hop", {}).get("normalization", "sqrt")
            == spec["mode"],
            label + ": wrong normalization metadata",
        )

    if original_model_cfg is None:
        original_model_cfg = model_cfg
    else:
        require(
            model_cfg == original_model_cfg,
            "Model settings differ beyond two-hop",
        )

    metadata[label] = {
        "checkpoint": str(path),
        "sha256": spec["sha256"],
        "epoch": 419,
        "global_step": 103740,
        "normalization": spec["mode"],
    }
    del checkpoint

input_paths = [
    cache,
    Path(base.__file__).resolve(),
    training_report_path,
    training_cache_path,
    root / "dgd/models/two_hop_edge.py",
    root / "dgd/models/h1_twohop_transformer.py",
    root / "dgd/diffusion_model_beta_h1.py",
    root / "higgs/interface_ablation.py",
] + [spec["path"] for spec in specs]

before = {str(path): sha256(path) for path in input_paths}

training_caches = read_json(training_cache_path)
require(training_caches.get("unchanged") is True, "Training cache check failed")
require(
    training_caches["after"].get(str(cache)) == before[str(cache)],
    "Validation cache differs from the mean training run",
)

selected, selection_histogram = base.extract_validation_endpoints()

require(len(selected) == 20, "Expected exactly 20 validation communities")
require(
    Counter(int(row["condition_y"]) for row in selected)
    == Counter({index: 5 for index in range(4)}),
    "Expected five communities per condition",
)

selection = []
for record in selected:
    record_id = int(record["record_id"])
    selection.append({
        "record_id": record_id,
        "validation_pair_index": int(record["pair_index"]),
        "side": record["side"],
        "reverse_seed": base.BASE_SEED + record_id,
        "condition_y": int(record["condition_y"]),
        "signature": record["signature"],
        **base.prefix_stats("real", base.graph_stats(record["graph"])),
    })

require(
    [row["record_id"] for row in selection] == list(range(20)),
    "Unexpected record ordering",
)

def check_identity(rows, label):
    require(len(rows) == 20, label + ": expected 20 rows")
    by_id = {int(float(row["record_id"])): row for row in rows}
    require(set(by_id) == set(range(20)), label + ": wrong record IDs")

    for expected in selection:
        row = by_id[expected["record_id"]]

        for name in (
            "record_id", "validation_pair_index", "reverse_seed",
            "condition_y", "real_nodes", "real_edges", "real_components",
        ):
            require(
                int(float(row[name])) == expected[name],
                label + ": identity mismatch in " + name,
            )

        require(row["side"] == expected["side"], label + ": side mismatch")

        for name in (
            "real_density", "real_mean_degree",
            "real_max_degree", "real_clustering",
        ):
            require(
                math.isclose(
                    float(row[name]), float(expected[name]),
                    rel_tol=1e-10, abs_tol=1e-10,
                ),
                label + ": real statistic mismatch in " + name,
            )

history_dir = (
    root / "higgs/outputs"
    / "facebook_h1_twohop_validation_compare_20261017_114758"
    / "old_best419"
)
history_csv = history_dir / "per_endpoint_results.csv"
history_check = "Historical CSV unavailable; current three models share one frozen selection"

if history_csv.is_file():
    with history_csv.open(newline="", encoding="utf-8") as handle:
        check_identity(list(csv.DictReader(handle)), "Historical selection")

    history_check = "Matched historical 20 communities, n/y/seeds and real statistics"

    history_summary = history_dir / "summary.json"
    if history_summary.is_file():
        old_summary = read_json(history_summary)
        require(
            old_summary["validation_cache_sha256"] == before[str(cache)],
            "Validation cache differs from the historical diagnostic",
        )

print("Selected communities:", len(selected), flush=True)
print("Selected conditions:", dict(sorted(Counter(
    int(row["condition_y"]) for row in selected
).items())), flush=True)
print("Selection check:", history_check, flush=True)
print("Sampling seeds: 20260825 + record_id", flush=True)

out.mkdir(parents=True, exist_ok=False)
save_json(out / "selected_endpoints.json", selection)
save_json(out / "input_manifest.json", {
    "input_sha256_before": before,
    "checkpoints": metadata,
    "base_seed": base.BASE_SEED,
    "per_class": base.PER_CLASS,
    "historical_selection_check": history_check,
})

base.extract_validation_endpoints = (
    lambda selected=selected, histogram=selection_histogram:
    (selected, histogram)
)

OriginalClass = base.ConditionalGraphBetaDiffusionH1
active_spec = None

class CheckedSampler:
    @staticmethod
    def load_from_checkpoint(*args, **kwargs):
        require(
            Path(args[0]).resolve() == active_spec["path"].resolve(),
            "Unexpected checkpoint requested by the diagnostic",
        )

        kwargs["strict"] = True
        model = OriginalClass.load_from_checkpoint(*args, **kwargs)

        require(model.T == 100, "Unexpected diffusion step count")
        require(model.edge_threshold == 0.5, "Unexpected edge threshold")
        require(model.ydim_output == 4, "Unexpected condition dimension")
        require(model.Xdim_output == 4, "Unexpected node-class dimension")

        options = model.two_hop_options
        enabled = active_spec["mode"] != "disabled"
        require(options["enabled"] == enabled, "Loaded two-hop flag mismatch")

        if enabled:
            require(
                options.get("normalization") == active_spec["mode"],
                "Loaded normalization mismatch",
            )
            require(
                getattr(model.model.two_hop_edge, "normalization", None)
                == active_spec["mode"],
                "Actual two-hop block normalization mismatch",
            )

        print(
            "Strictly loaded:", active_spec["label"],
            "backbone:", type(model.model).__name__,
            "normalization:", active_spec["mode"],
            flush=True,
        )
        return model

base.ConditionalGraphBetaDiffusionH1 = CheckedSampler

results = {}
combined_rows = []

for spec in specs:
    active_spec = spec
    label = spec["label"]
    subdir = out / label

    require(
        not subdir.exists(),
        "Refusing to resume an existing output: " + str(subdir),
    )

    base.CHECKPOINT = spec["path"]
    base.OUTPUT_DIR = subdir

    print("\nStarting:", label, flush=True)
    base.main()

    with (subdir / "per_endpoint_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    check_identity(rows, label)

    result = read_json(subdir / "summary.json")
    require(result["selected_count"] == 20, label + ": incomplete summary")
    require(
        result["checkpoint_sha256"] == spec["sha256"],
        label + ": summary checkpoint mismatch",
    )
    require(result["base_seed"] == 20260825, label + ": seed changed")

    results[label] = result
    combined_rows.extend({"model": label, **row} for row in rows)

    gc.collect()
    torch.cuda.empty_cache()
    print("Finished:", label, flush=True)

with (out / "per_endpoint_results_all_models.csv").open(
    "x", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=["model"] + list(base.FIELDS))
    writer.writeheader()
    writer.writerows(combined_rows)

after = {str(path): sha256(path) for path in input_paths}
save_json(out / "input_hash_check.json", {
    "before": before,
    "after": after,
    "unchanged": before == after,
})
require(before == after, "An input file changed during evaluation")

save_json(out / "comparison.json", {
    "protocol": "same 20 validation communities; fixed n/y; original reverse sampler",
    "checkpoint_selection": "validation-loss best419 for all three variants",
    "base_seed": 20260825,
    "per_class": 5,
    "historical_selection_check": history_check,
    "checkpoints": metadata,
    "models": results,
    "input_hashes_unchanged": True,
})

print("\nFacebook H1 three-way best419 validation comparison")
print(
    f"{'model':<24} {'edges':>10} {'density':>10} {'clustering':>11} "
    f"{'components':>11} {'connected':>10} {'LCC':>8} {'cond_match':>11}"
)

first = next(iter(results.values()))
print(
    f"{'real':<24} "
    f"{first['means']['edges']['real']:10.3f} "
    f"{first['means']['density']['real']:10.6f} "
    f"{first['means']['clustering']['real']:11.6f} "
    f"{first['means']['components']['real']:11.3f}"
)

for label, result in results.items():
    print(
        f"{label:<24} "
        f"{result['means']['edges']['full']:10.3f} "
        f"{result['means']['density']['full']:10.6f} "
        f"{result['means']['clustering']['full']:11.6f} "
        f"{result['means']['components']['full']:11.3f} "
        f"{result['full_connected_rate']:10.3f} "
        f"{result['mean_lcc_retention']:8.4f} "
        f"{result['condition_majority_match_rate']:11.3f}"
    )

print("\nFull H1 graph mean absolute errors: lower is better")
print(
    f"{'model':<24} {'edge_MAE':>11} {'density_MAE':>12} "
    f"{'degree_MAE':>12} {'maxdeg_MAE':>12} {'clust_MAE':>11}"
)

for label, result in results.items():
    error = result["mean_absolute_errors"]["full"]
    print(
        f"{label:<24} {error['edges']:11.3f} {error['density']:12.6f} "
        f"{error['mean_degree']:12.6f} {error['max_degree']:12.3f} "
        f"{error['clustering']:11.6f}"
    )
    print("  Used/LCC MAE:", json.dumps(result["mean_absolute_errors"]["used"]))

print("\nCSV:", out / "per_endpoint_results_all_models.csv")
print("JSON:", out / "comparison.json")
print("PASS: FACEBOOK_H1_THREEWAY_MEAN_VALIDATION_COMPLETE")
print("Strict checkpoint loading; same 20 communities/n/y/seeds; input hashes unchanged.")
print("Existing last499 checkpoints and previous results were not overwritten.")
print("No training, H2/Glue sampling or whole-graph MMD evaluation was performed.")
