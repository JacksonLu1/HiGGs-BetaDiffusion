
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
import cora_beta_h1_fixed_validation as base

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

require(os.environ.get("CUDA_VISIBLE_DEVICES") == "2", "Expected physical GPU 2")
require(torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback")
require(not out.exists(), "Output already exists: " + str(out))
require(Path(base.ROOT).resolve() == root, "Unexpected diagnostic ROOT")
require(base.BASE_SEED == 20260807, "Original sampling seed was changed")
require(base.PER_CLASS == 5, "Original per-class selection was changed")

expected_cache = (
    root / "data/cora/processed"
    / "proc_val_X_r2_majority_max500_bs2.pt"
)
require(Path(base.VAL_CACHE).resolve() == expected_cache, "Validation cache changed")
require(expected_cache.is_file(), "Missing validation cache")

lock_handle = (
    root / "logs/evaluation/cora_h1_threeway_mean_gpu2.lock"
).open("a+")
try:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError("Another Cora three-way evaluation is already running")

torch.cuda.set_device(0)
free_bytes, _ = torch.cuda.mem_get_info()
print("Physical GPU 2 -> cuda:0:", torch.cuda.get_device_name(0), flush=True)
print("Free GPU memory GiB:", round(free_bytes / 1024**3, 2), flush=True)
require(free_bytes >= 4 * 1024**3, "Less than 4 GiB free GPU memory")

sqrt_run = (
    root / "outputs"
    / "cora_beta_h1_twohop_r2_e1000_scratch_seed0_20261016_152540"
)
mean_run = (
    root / "outputs"
    / "cora_beta_h1_twohop_mean_r2_e1000_scratch_seed0_20261017_182430"
)

mean_last = mean_run / "checkpoints" / mean_run.name / "last.ckpt"
mean_best = mean_run / "checkpoints" / mean_run.name / "epoch=789.ckpt"

require(mean_last.is_file(), "Missing mean last.ckpt")
require(mean_best.is_file(), "Missing mean epoch789 checkpoint")

last = base._load_checkpoint_cpu(mean_last)
require(last["epoch"] == 999, "Mean training did not reach epoch999")
require(last["global_step"] == 251000, "Unexpected mean training step count")
require(
    last.get("h1_two_hop", {}).get("normalization") == "mean",
    "Mean last checkpoint has wrong normalization",
)

best_states = [
    state for state in last.get("callbacks", {}).values()
    if isinstance(state, dict) and state.get("monitor") == "val/h1_total"
]
require(len(best_states) == 1, "Cannot identify validation-best checkpoint")

best_state = best_states[0]
saved_best = Path(best_state["best_model_path"])
if not saved_best.is_absolute():
    saved_best = mean_run / saved_best
require(
    saved_best.resolve() == mean_best.resolve(),
    "epoch789 does not match the recorded validation-best checkpoint",
)
require(
    math.isfinite(float(best_state["best_model_score"])),
    "Nonfinite validation-best score",
)
print("Mean validation-best score:", float(best_state["best_model_score"]), flush=True)
del last

specs = [
    {
        "label": "old_beta_h1_epoch649",
        "mode": "disabled",
        "path": root / "saved_models/cora/cora_h1.ckpt",
        "sha256": "47b8a5fa3a9e213efd348586f84b82d40d6117c26cd1708df21fa21d82ff758c",
        "epoch": 649,
        "step": 163150,
    },
    {
        "label": "sqrt_twohop_epoch939",
        "mode": "sqrt",
        "path": sqrt_run / "checkpoints" / sqrt_run.name / "epoch=939.ckpt",
        "sha256": "c3e9c34a4aef0b4b5d3b3626463047b22e69b9df2f4f672fa197456e7e7c3fe2",
        "epoch": 939,
        "step": 235940,
    },
    {
        "label": "mean_twohop_epoch789",
        "mode": "mean",
        "path": mean_best,
        "sha256": sha256(mean_best),
        "epoch": 789,
        "step": 198290,
    },
]

metadata = {}
original_model_cfg = None

for spec in specs:
    path = spec["path"]
    label = spec["label"]
    require(path.is_file(), "Missing checkpoint: " + str(path))
    require(sha256(path) == spec["sha256"], label + ": SHA256 mismatch")

    checkpoint = base._load_checkpoint_cpu(path)
    cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])

    for name, expected in (
        ("dataset.name", "cora"),
        ("dataset.h", 1),
        ("dataset.resolution", 2),
        ("dataset.regime", "majority"),
        ("model.family", "beta_h1"),
        ("model.diffusion_steps", 100),
        ("model.edge_threshold", 0.5),
    ):
        require(
            OmegaConf.select(cfg, name) == expected,
            label + ": unexpected " + name,
        )

    require(checkpoint["epoch"] == spec["epoch"], label + ": wrong epoch")
    require(checkpoint["global_step"] == spec["step"], label + ": wrong step")

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
        "epoch": spec["epoch"],
        "global_step": spec["step"],
        "normalization": spec["mode"],
    }
    del checkpoint

input_paths = [
    expected_cache,
    Path(base.__file__).resolve(),
    mean_last,
    root / "dgd/models/two_hop_edge.py",
    root / "dgd/models/h1_twohop_transformer.py",
    root / "dgd/diffusion_model_beta_h1.py",
    root / "higgs/interface_ablation.py",
] + [spec["path"] for spec in specs]

before = {str(path): sha256(path) for path in input_paths}

selected, selection_histogram = base.extract_validation_endpoints()
require(len(selected) == 35, "Expected exactly 35 validation endpoints")
require(
    Counter(int(row["condition_y"]) for row in selected)
    == Counter({index: 5 for index in range(7)}),
    "Expected five endpoints per condition",
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
    [row["record_id"] for row in selection] == list(range(35)),
    "Unexpected record ordering",
)

def check_identity(rows, label):
    require(len(rows) == 35, label + ": expected 35 rows")
    by_id = {int(float(row["record_id"])): row for row in rows}
    require(set(by_id) == set(range(35)), label + ": wrong record IDs")

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
    / "cora_beta_h1_fixed_validation_epoch649_seed20260807"
)
history_csv = history_dir / "per_endpoint_results.csv"
history_check = "Historical CSV unavailable; current three models share one frozen selection"

if history_csv.is_file():
    with history_csv.open(newline="", encoding="utf-8") as handle:
        check_identity(list(csv.DictReader(handle)), "Historical selection")
    history_check = "Matched historical 35 endpoints, n/y/seeds and real statistics"

    history_summary = history_dir / "summary.json"
    if history_summary.is_file():
        with history_summary.open(encoding="utf-8") as handle:
            old_summary = json.load(handle)
        require(
            old_summary["validation_cache_sha256"] == before[str(expected_cache)],
            "Validation cache differs from the historical diagnostic",
        )

print("Selected endpoints:", len(selected), flush=True)
print("Selected conditions:", dict(sorted(Counter(
    int(row["condition_y"]) for row in selected
).items())), flush=True)
print("Selection check:", history_check, flush=True)
print("Sampling seeds: 20260807 + record_id", flush=True)

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
        require(model.ydim_output == 7, "Unexpected Cora condition dimension")

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
    require(not subdir.exists(), "Refusing to resume an existing output: " + str(subdir))

    base.CHECKPOINT = spec["path"]
    base.OUTPUT_DIR = subdir

    print("\nStarting:", label, flush=True)
    base.main()

    with (subdir / "per_endpoint_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    check_identity(rows, label)

    with (subdir / "summary.json").open(encoding="utf-8") as handle:
        result = json.load(handle)

    require(result["selected_count"] == 35, label + ": incomplete summary")
    require(
        result["checkpoint_sha256"] == spec["sha256"],
        label + ": summary checkpoint mismatch",
    )
    require(result["base_seed"] == 20260807, label + ": seed changed")

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
    "protocol": "same 35 validation endpoints; fixed n/y; original reverse sampler",
    "base_seed": 20260807,
    "per_class": 5,
    "historical_selection_check": history_check,
    "checkpoints": metadata,
    "models": results,
    "input_hashes_unchanged": True,
})

print("\nCora H1 three-way validation generation comparison")
print(
    f"{'model':<25} {'edges':>10} {'density':>10} {'clustering':>11} "
    f"{'components':>11} {'connected':>10} {'LCC':>8} {'cond_match':>11}"
)

first = next(iter(results.values()))
print(
    f"{'real':<25} "
    f"{first['means']['edges']['real']:10.3f} "
    f"{first['means']['density']['real']:10.6f} "
    f"{first['means']['clustering']['real']:11.6f} "
    f"{first['means']['components']['real']:11.3f}"
)

for label, result in results.items():
    print(
        f"{label:<25} "
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
    f"{'model':<25} {'edge_MAE':>11} {'density_MAE':>12} "
    f"{'degree_MAE':>12} {'maxdeg_MAE':>12} {'clust_MAE':>11}"
)
for label, result in results.items():
    error = result["mean_absolute_errors"]["full"]
    print(
        f"{label:<25} {error['edges']:11.3f} {error['density']:12.6f} "
        f"{error['mean_degree']:12.6f} {error['max_degree']:12.3f} "
        f"{error['clustering']:11.6f}"
    )
    print("  Used/LCC MAE:", json.dumps(result["mean_absolute_errors"]["used"]))

print("\nCSV:", out / "per_endpoint_results_all_models.csv")
print("JSON:", out / "comparison.json")
print("PASS: CORA_H1_THREEWAY_MEAN_VALIDATION_COMPLETE")
print("Strict checkpoint loading; same 35 endpoints/n/y/seeds; input hashes unchanged.")
print("No training, H2/Glue sampling or whole-graph MMD evaluation was performed.")
