import csv
import hashlib
import json
import math
import os
import runpy
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("Usage: runner.py PROJECT_ROOT OUTPUT_DIR")

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
data_dir = root / "data" / "sbm"

os.chdir(root)
sys.path[:0] = [str(root / "dgd"), str(root)]

import torch
from pytorch_lightning import seed_everything

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

# Construct all launch arguments before starting the training entry point.
overrides = [
    str(root / "dgd" / "main.py"),
    "dataset=sbm",
    "model=beta_h1_twohop_mean_hog_rw",
    "dataset.datadir=" + str(data_dir),
    "dataset.h=1",
    "dataset.resolution=1",
    "dataset.max_size=500",
    "dataset.regime=majority",
    "dataset.dataset_testing=false",
    "dataset.random_subset=null",
    "dataset.subsample=false",
    "train.n_epochs=500",
    "train.batch_size=8",
    "train.num_workers=0",
    "train.lr=0.0002",
    "train.weight_decay=1e-12",
    "train.amsgrad=true",
    "train.optimizer=adamw",
    "train.clip_grad=null",
    "train.save_model=true",
    "train.ema_decay=0",
    "train.seed=0",
    "general.name=" + out.name,
    "general.gpus=1",
    "general.wandb=disabled",
    "general.resume=null",
    "general.test_only=null",
    "general.check_val_every_n_epochs=10",
    "general.val_check_interval=1.0",
    "general.log_every_steps=10",
    "general.samples_to_generate=4",
    "general.final_model_samples_to_generate=40",
    "++general.limit_train_batches=1.0",
    "++general.limit_val_batches=1.0",
    "++general.limit_test_batches=0",
    "++general.run_test_after_fit=false",
    "model.two_hop.warm_start_checkpoint=null",
    "model.two_hop.enabled=true",
    "model.two_hop.normalization=mean",
    "model.two_hop.channels=16",
    "model.hog_rw.enabled=true",
    "model.hog_rw.rw_depth=4",
    "model.pre_condition=false",
    "model.eta_strategy=fixed",
    "hydra.run.dir=" + str(out),
    "++hydra.job.chdir=true",
]

config_path = root / "configs/model/beta_h1_twohop_mean_hog_rw.yaml"
if not config_path.is_file():
    raise FileNotFoundError(config_path)

caches = [
    data_dir / "processed"
    / f"proc_{stage}_h1_r1_majority_max500_bs8.pt"
    for stage in ("train", "val", "test")
]

for path in caches:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required existing cache: {path}")

before = {str(path): sha256(path) for path in caches}

if out.exists():
    raise FileExistsError(f"Refusing to overwrite output directory: {out}")

if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("Wrong physical GPU selected")

import fcntl

lock_handle = (
    root / "logs/diagnostics/sbm_mean_hog_smoke.lock"
).open("a+")

try:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError("This dataset's mean + HOG job is already running")

if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
    raise RuntimeError("Wrong physical GPU selected")

import fcntl

formal_lock_handle = (
    root / "logs/training/sbm_mean_hog_formal_seed0.lock"
).open("a+")

try:
    fcntl.flock(
        formal_lock_handle.fileno(),
        fcntl.LOCK_EX | fcntl.LOCK_NB,
    )
except BlockingIOError:
    raise RuntimeError(
        "A formal mean + HOG job for this dataset is already running"
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; refusing CPU training")

torch.cuda.set_device(0)
free, total = torch.cuda.mem_get_info(0)

print("Runner:", __file__)
print("Physical GPU 0 -> cuda:0:", torch.cuda.get_device_name(0))
print("Free GPU memory GiB:", round(free / 1024**3, 2))

if free < 4 * 1024**3:
    raise RuntimeError(
        "GPU 0 has less than 4 GiB free; no training started"
    )

seed_everything(0, workers=True)

from datasets import sbm_hierarchies as smoke_dataset

def refuse_rebuild(self, *args, **kwargs):
    raise RuntimeError(
        "This formal run only permits existing caches. "
        "Dataset download or reprocessing has been blocked."
    )

# These guards affect this process only; no dataset source files are edited.
dataset_class = smoke_dataset.SBMDataset
expected_paths = {path.resolve() for path in caches}

def existing_cache_only(self, *args, **kwargs):
    actual = {Path(path).resolve() for path in self.processed_paths}

    if float(self.h) != 1 or Path(self.root).resolve() != data_dir:
        raise RuntimeError("Unexpected dataset stage/root")

    if actual != expected_paths:
        raise RuntimeError("Unexpected processed cache paths")

    for path in actual:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Missing existing cache: " + str(path))

dataset_class._download = existing_cache_only
dataset_class._process = existing_cache_only
dataset_class.download = refuse_rebuild
dataset_class.process = refuse_rebuild

out.mkdir(parents=True, exist_ok=False)

(out / "launch_manifest.json").write_text(
    json.dumps(
        {
            "runner": str(Path(__file__).resolve()),
            "physical_gpu": 0,
            "source_smoke_runner": '/path/to/HiGGs-GBD/HiGGs-main/tools/diagnostics/sbm_h1_mean_hog_real_smoke_seed0_20261022_132411_runner.py',
            "source_smoke_sha256": 'd185f2adef6e4b034374cfac428195a36854ae3521823466bbe4b08e7946b91f',
            "seed": 0,
            "arguments": overrides,
            "cache_sha256_before": before,
        },
        indent=2,
    ),
    encoding="utf-8",
)

sys.argv = overrides

print("Starting sbm H1 mean + HOG FORMAL TRAINING FROM SCRATCH")
print("FORMAL: full train/val loaders; epochs=500; batch_size=8; seed=0")
print("No old checkpoint will be loaded.")
print("Initialization still scans the existing caches.")

try:
    runpy.run_path(
        str(root / "dgd" / "main.py"),
        run_name="__main__",
    )
finally:
    after = {str(path): sha256(path) for path in caches}
    unchanged = before == after

    (out / "cache_manifest.json").write_text(
        json.dumps(
            {
                "before": before,
                "after": after,
                "unchanged": unchanged,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not unchanged:
        raise RuntimeError("Processed cache SHA256 changed")

    print("PASS: processed cache SHA256 unchanged")

from higgs.higgs_utils import _load_checkpoint_cpu
from omegaconf import OmegaConf

last_paths = list(out.rglob("last.ckpt"))
assert len(last_paths) == 1, last_paths

last_path = last_paths[0]
checkpoint = _load_checkpoint_cpu(last_path)
cfg = checkpoint["hyper_parameters"]["cfg"]
metadata = checkpoint["h1_two_hop"]

assert checkpoint["epoch"] == 499
assert checkpoint["global_step"] == 29500
assert checkpoint["model_family"] == "beta_h1"
assert metadata["normalization"] == "mean"
assert metadata["warm_start"] is None

assert OmegaConf.select(cfg, "model.two_hop.normalization") == "mean"
assert OmegaConf.select(cfg, "model.two_hop.warm_start_checkpoint") is None
assert OmegaConf.select(cfg, "general.resume") is None
assert OmegaConf.select(cfg, "general.test_only") is None
assert int(OmegaConf.select(cfg, "train.n_epochs")) == 500
assert int(OmegaConf.select(cfg, "train.batch_size")) == 8


hog_metadata = checkpoint["h1_hog_rw"]

assert metadata["version"] == 2
assert metadata["channels"] == 16
assert hog_metadata["version"] == 2
assert hog_metadata["rw_depth"] == 4
assert hog_metadata["two_hop"] is True

expected_config = {
    "dataset.name": "sbm",
    "dataset.h": 1,
    "dataset.resolution": 1,
    "dataset.max_size": 500,
    "model.two_hop.enabled": True,
    "model.two_hop.channels": 16,
    "model.hog_rw.enabled": True,
    "model.hog_rw.rw_depth": 4,
    "model.pre_condition": False,
    "model.eta_strategy": "fixed",
    "general.run_test_after_fit": False,
}

for name, value in expected_config.items():
    assert OmegaConf.select(cfg, name) == value, name

input_widths = {
    key: int(
        checkpoint["state_dict"][
            f"model.mlp_in_{key}.0.weight"
        ].shape[1]
    )
    for key in ("X", "E", "y")
}

assert input_widths == {'X': 12, 'E': 6, 'y': 14}, input_widths

assert float(
    OmegaConf.select(cfg, "general.limit_train_batches")
) == 1.0
assert float(
    OmegaConf.select(cfg, "general.limit_val_batches")
) == 1.0

matching = [
    state for state in checkpoint.get("callbacks", {}).values()
    if isinstance(state, dict)
    and state.get("monitor") == "val/h1_total"
]
assert len(matching) == 1, matching

callback = matching[0]
best_score = float(callback["best_model_score"])
assert math.isfinite(best_score)

saved = Path(callback["best_model_path"])
candidates = (
    [saved] if saved.is_absolute()
    else [out / saved, last_path.parent / saved.name]
)

best_path = next(
    (
        p.resolve() for p in candidates
        if p.is_file() and p.resolve().is_relative_to(out)
    ),
    None,
)
assert best_path is not None, candidates

best_checkpoint = _load_checkpoint_cpu(best_path)

for field in ("h1_two_hop", "h1_hog_rw"):
    assert best_checkpoint[field] == checkpoint[field], field

print("Best epoch:", best_checkpoint["epoch"])
print("Best val/h1_total:", best_score)

losses = {"train": [], "val": []}

for path in out.rglob("metrics.csv"):
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if not key or not value or "h1_total" not in key:
                    continue
                for stage in losses:
                    if key.startswith(stage + "/"):
                        number = float(value)
                        assert math.isfinite(number), (key, value)
                        losses[stage].append(number)

assert losses["train"], "No training loss found"
assert losses["val"], "No validation loss found"

report = {
    "passed": True,
    "epoch": checkpoint["epoch"],
    "global_step": checkpoint["global_step"],
    "two_hop_metadata": metadata,
    "hog_metadata": hog_metadata,
    "actual_input_widths": input_widths,
    "last_checkpoint": str(last_path),
    "best_checkpoint": str(best_path),
    "best_epoch": best_checkpoint["epoch"],
    "best_validation_score": best_score,
    "losses": {k: {"count": len(v), "last": v[-1]} for k, v in losses.items()},
    "processed_cache_unchanged": True,
}

(out / "formal_training_check.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("Last checkpoint:", last_path)
print("Best checkpoint:", best_path)
print("Epoch:", checkpoint["epoch"])
print("Global step:", checkpoint["global_step"])
print("Two-hop metadata:", metadata)
print("HOG metadata:", hog_metadata)
print("Actual input widths:", input_widths)
print("Finite loss counts:", {k: len(v) for k, v in losses.items()})
print("PASS: SBM_MEAN_HOG_FORMAL_TRAINING_COMPLETE")
print("No H2/Glue sampling or test evaluation was performed.")
