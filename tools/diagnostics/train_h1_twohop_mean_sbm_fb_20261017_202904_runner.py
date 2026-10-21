
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
key = sys.argv[2]
out = Path(sys.argv[3]).resolve()

specs = {
    "sbm": {
        "group": "sbm",
        "data": "sbm",
        "resolution": 1,
        "max_size": 500,
        "gpu": "0",
        "free_gib": 4,
        "module": "datasets.sbm_hierarchies",
        "class": "SBMDataset",
        "run": "sbm_beta_h1_twohop_r1_e500_scratch_seed0_20261016_205246_cacheonly",
        "checkpoint": "epoch=359.ckpt",
        "sha256": "a12589310a6dba3e7b0ae6ff275842da51c7f71a8a16740077eb44c4e3247263",
        "steps": 29500,
    },
    "facebook": {
        "group": "fb_hierarchies",
        "data": "fb",
        "resolution": 10,
        "max_size": 400,
        "gpu": "1",
        "free_gib": 20,
        "module": "datasets.fb_hierarchies",
        "class": "FBHierarchiesDataset",
        "run": "fb_beta_h1_twohop_r10_e500_scratch_seed0_20261016_204432",
        "checkpoint": "epoch=419.ckpt",
        "sha256": "eae327fe4dcc51d0044b491fa59d06eaa7a3336333224ea0fcb7484e09b959b8",
        "steps": 123500,
    },
}

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
        json.dump(value, handle, indent=2, ensure_ascii=False)

require(key in specs, "Unknown dataset: " + key)
s = specs[key]

require(out.parent == root / "outputs", "Unexpected output directory")
require(not out.exists(), "Output already exists: " + str(out))
require(
    os.environ.get("CUDA_VISIBLE_DEVICES") == s["gpu"],
    "Unexpected CUDA_VISIBLE_DEVICES",
)

os.chdir(root)
sys.path[:0] = [str(root / "dgd"), str(root)]

lock_path = root / "logs/training" / (key + "_twohop_mean_seed0.lock")
lock_handle = lock_path.open("a+")
try:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError("This dataset's mean training is already running")

if key == "sbm":
    os.environ["HIGGS_SBM_CACHE_ONLY"] = "1"

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from higgs.higgs_utils import _load_checkpoint_cpu
from models.two_hop_edge import two_hop_options

require(torch.cuda.is_available(), "CUDA unavailable; refusing CPU training")
torch.cuda.set_device(0)
free_bytes, total_bytes = torch.cuda.mem_get_info()
free_gib = free_bytes / 1024**3

print("Dataset:", key, flush=True)
print("Physical GPU", s["gpu"], "-> cuda:0:", torch.cuda.get_device_name(0))
print("Free GPU memory GiB:", round(free_gib, 2), flush=True)
require(free_gib >= s["free_gib"], "Insufficient free GPU memory")

reference = (
    root / "outputs" / s["run"] / "checkpoints"
    / s["run"] / s["checkpoint"]
)
require(reference.is_file(), "Missing reference checkpoint: " + str(reference))
require(sha256(reference) == s["sha256"], "Reference checkpoint SHA256 mismatch")

checkpoint = _load_checkpoint_cpu(reference)
old = OmegaConf.to_container(
    OmegaConf.create(checkpoint["hyper_parameters"]["cfg"]),
    resolve=True,
)

require(old["model"]["family"] == "beta_h1", "Wrong reference model family")
require(old["model"]["two_hop"]["enabled"] is True, "Reference two-hop disabled")
require(
    old["model"]["two_hop"].get("normalization", "sqrt") == "sqrt",
    "Reference is not the previous sqrt experiment",
)
require(
    old["model"]["two_hop"].get("warm_start_checkpoint") is None,
    "Reference was not configured for scratch training",
)

train = dict(old["train"])
require(train["n_epochs"] == 500, "Unexpected reference epoch budget")
require(train["batch_size"] == 8, "Unexpected reference batch size")
require(train["seed"] == 0, "Unexpected reference seed")
require(train["save_model"] is True, "Checkpoint saving must be enabled")

reference_epoch = int(checkpoint["epoch"])
reference_step = int(checkpoint["global_step"])
steps_per_epoch, remainder = divmod(reference_step, reference_epoch + 1)
require(remainder == 0, "Cannot infer reference steps per epoch")
require(steps_per_epoch * 500 == s["steps"], "Unexpected reference step count")
del checkpoint

data_dir = root / "data" / s["data"]
dataset = dict(old["dataset"])
require(dataset["name"] == s["group"], "Wrong reference dataset")
require(float(dataset["h"]) == 1, "Reference is not H1")
require(dataset["resolution"] == s["resolution"], "Wrong resolution")
require(dataset["max_size"] == s["max_size"], "Wrong max_size")
require(dataset["regime"] == "majority", "Wrong conditioning regime")
dataset["datadir"] = str(data_dir)

cache_paths = [
    data_dir / "processed" / (
        "proc_" + split + "_h1_r" + str(s["resolution"])
        + "_majority_max" + str(s["max_size"]) + "_bs8.pt"
    )
    for split in ("train", "val", "test")
]
for path in cache_paths:
    require(path.is_file() and path.stat().st_size > 0, "Missing cache: " + str(path))

before = {str(path): sha256(path) for path in cache_paths}
expected_paths = set(cache_paths)

module = importlib.import_module(s["module"])
dataset_class = getattr(module, s["class"])

def cache_only_guard(self, *args, **kwargs):
    actual = {Path(path).resolve() for path in self.processed_paths}
    require(float(self.h) == 1, "Cache-only guard requires H1")
    require(Path(self.root).resolve() == data_dir, "Unexpected dataset root")
    require(actual == expected_paths, "Unexpected processed cache paths")
    for path in actual:
        require(
            path.is_file() and path.stat().st_size > 0,
            "Cache-only mode cannot create missing data: " + str(path),
        )

def forbid_processing(self, *args, **kwargs):
    raise RuntimeError("Downloading/repartitioning/processing is forbidden in this run")

dataset_class._download = cache_only_guard
dataset_class._process = cache_only_guard
dataset_class.download = forbid_processing
dataset_class.process = forbid_processing

counts = {}
for split in ("train", "val", "test"):
    ds = dataset_class(
        stage=split,
        root=str(data_dir),
        h=1,
        resolution=s["resolution"],
        max_size=s["max_size"],
        n_workers=train["num_workers"],
        batch_size=8,
        regime="majority",
    )
    counts[split] = len(ds)
    require(len(ds) > 0, "Empty dataset: " + split)
    del ds

require(
    math.ceil(counts["train"] / 8) == steps_per_epoch,
    "Training cache size does not match the previous training setup",
)
print("Existing cache graph counts:", counts, flush=True)

def override(name, value):
    return "++" + name + "=" + json.dumps(value, separators=(",", ":"))

arguments = [
    "dataset=" + s["group"],
    "model=beta_h1_twohop_mean",
]

for section, values in (("train", train), ("dataset", dataset)):
    for name, value in values.items():
        arguments.append(override(section + "." + name, value))

general = {
    "name": out.name,
    "gpus": 1,
    "wandb": "disabled",
    "resume": None,
    "test_only": None,
    "run_test_after_fit": False,
    "limit_train_batches": 1.0,
    "limit_val_batches": 1.0,
    "limit_test_batches": 0,
}
for name, default in (
    ("check_val_every_n_epochs", 10),
    ("val_check_interval", None),
    ("log_every_steps", 5),
    ("samples_to_generate", 4),
    ("final_model_samples_to_generate", 40),
):
    general[name] = old["general"].get(name, default)

require(
    general["samples_to_generate"] > 0
    and general["final_model_samples_to_generate"] >= general["samples_to_generate"],
    "Invalid inherited sampling budget",
)

for name, value in general.items():
    arguments.append(override("general." + name, value))

arguments.extend([
    "model.two_hop.warm_start_checkpoint=null",
    "hydra.run.dir=" + str(out),
    "++hydra.job.chdir=true",
])

with initialize_config_dir(version_base="1.1", config_dir=str(root / "configs")):
    cfg = compose(config_name="config", overrides=arguments)

options = two_hop_options(cfg.model.get("two_hop"))
require(options["enabled"] and options["normalization"] == "mean", "Mean mode not active")
require(options["warm_start_checkpoint"] is None, "Warm start must be disabled")

actual = OmegaConf.to_container(cfg, resolve=True)
require(actual["train"] == train, "Training configuration differs from reference")
require(actual["dataset"] == dataset, "Dataset configuration differs from reference")

expected_model = json.loads(json.dumps(old["model"]))
actual_model = actual["model"]
expected_model["two_hop"].pop("normalization", None)
actual_model["two_hop"].pop("normalization", None)
require(actual_model == expected_model, "Model differs beyond normalization")

out.mkdir(parents=True, exist_ok=False)
save_json(out / "launch_manifest.json", {
    "dataset": key,
    "physical_gpu": s["gpu"],
    "runner": str(Path(__file__).resolve()),
    "reference_checkpoint_for_configuration_only": str(reference),
    "reference_sha256": s["sha256"],
    "resume": None,
    "warm_start_checkpoint": None,
    "normalization": "mean",
    "cache_graph_counts": counts,
    "cache_sha256_before": before,
    "arguments": arguments,
})

print("Starting FORMAL H1 mean training FROM SCRATCH", flush=True)
print("Epochs: 500; batch size: 8; seed: 0", flush=True)
print("No checkpoint weights or optimizer state will be restored", flush=True)

sys.argv = [str(root / "dgd/main.py")] + arguments
seed_everything(0, workers=True)

try:
    runpy.run_path(str(root / "dgd/main.py"), run_name="__main__")
finally:
    after = {str(path): sha256(path) for path in cache_paths}
    save_json(out / "cache_manifest.json", {
        "before": before,
        "after": after,
        "unchanged": before == after,
    })
    require(before == after, "Processed cache SHA256 changed")

last_paths = list((out / "checkpoints").rglob("last.ckpt"))
require(len(last_paths) == 1, "Expected exactly one last.ckpt")
last = _load_checkpoint_cpu(last_paths[0])
metadata = last.get("h1_two_hop", {})

require(last["epoch"] == 499, "Training did not reach epoch499")
require(last["global_step"] == s["steps"], "Unexpected final global_step")
require(metadata.get("version") == 2, "Wrong two-hop metadata version")
require(metadata.get("normalization") == "mean", "Checkpoint is not mean mode")
require(metadata.get("warm_start") is None, "Unexpected warm start")

last_cfg = OmegaConf.create(last["hyper_parameters"]["cfg"])
require(last_cfg.general.resume is None, "Unexpected resume configuration")
require(last_cfg.model.two_hop.warm_start_checkpoint is None, "Unexpected warm start")
require(last_cfg.model.two_hop.normalization == "mean", "Wrong saved normalization")

best_states = [
    state for state in last.get("callbacks", {}).values()
    if isinstance(state, dict) and state.get("monitor") == "val/h1_total"
]
require(len(best_states) == 1, "Cannot identify validation-best checkpoint")
best_state = best_states[0]
require(math.isfinite(float(best_state["best_model_score"])), "Nonfinite best score")

best = Path(best_state["best_model_path"])
if not best.is_absolute():
    best = out / best
best = best.resolve()
require(best.is_relative_to(out) and best.is_file(), "Invalid best checkpoint path")

best_checkpoint = _load_checkpoint_cpu(best)
require(
    best_checkpoint.get("h1_two_hop", {}).get("normalization") == "mean",
    "Best checkpoint is not mean mode",
)

loss_counts = {"train": 0, "val": 0}
for csv_path in out.rglob("metrics.csv"):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for name, value in row.items():
                if "h1_total" not in name or value in (None, ""):
                    continue
                split = name.split("/", 1)[0]
                if split in loss_counts:
                    require(math.isfinite(float(value)), "Nonfinite loss: " + name)
                    loss_counts[split] += 1
require(all(loss_counts.values()), "Missing train/validation loss records")

save_json(out / "formal_training_check.json", {
    "passed": True,
    "dataset": key,
    "epoch": last["epoch"],
    "global_step": last["global_step"],
    "two_hop_metadata": metadata,
    "best_checkpoint": str(best),
    "best_sha256": sha256(best),
    "best_score": float(best_state["best_model_score"]),
    "finite_loss_counts": loss_counts,
    "processed_cache_unchanged": True,
})

print("Last checkpoint:", last_paths[0], flush=True)
print("Best checkpoint:", best, flush=True)
print("Two-hop metadata:", metadata, flush=True)
print("PASS:", key.upper() + "_MEAN_FORMAL_TRAINING_COMPLETE", flush=True)
print("No H2/Glue sampling or generation-quality evaluation was performed.", flush=True)
