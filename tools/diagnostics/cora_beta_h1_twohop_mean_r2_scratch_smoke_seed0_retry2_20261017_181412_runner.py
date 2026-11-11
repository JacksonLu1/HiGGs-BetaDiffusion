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
data_dir = root / "data" / "cora"

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
    "dataset=cora",
    "model=beta_h1_twohop_mean",
    "dataset.datadir=" + str(data_dir),
    "dataset.h=1",
    "dataset.resolution=2",
    "dataset.max_size=500",
    "dataset.regime=majority",
    "dataset.dataset_testing=false",
    "dataset.random_subset=null",
    "dataset.subsample=false",
    "train.n_epochs=1",
    "train.batch_size=4",
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
    "general.check_val_every_n_epochs=1",
    "general.val_check_interval=1.0",
    "general.log_every_steps=1",
    "general.samples_to_generate=4",
    "general.final_model_samples_to_generate=40",
    "++general.limit_train_batches=2",
    "++general.limit_val_batches=2",
    "++general.limit_test_batches=0",
    "++general.run_test_after_fit=false",
    "model.two_hop.warm_start_checkpoint=null",
    "hydra.run.dir=" + str(out),
    "++hydra.job.chdir=true",
]

config_path = root / "configs/model/beta_h1_twohop_mean.yaml"
if not config_path.is_file():
    raise FileNotFoundError(config_path)

caches = [
    data_dir / "processed"
    / f"proc_{stage}_h1_r2_majority_max500_bs4.pt"
    for stage in ("train", "val", "test")
]

for path in caches:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required existing cache: {path}")

before = {str(path): sha256(path) for path in caches}

if out.exists():
    raise FileExistsError(f"Refusing to overwrite output directory: {out}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; refusing CPU training")

torch.cuda.set_device(0)
free, total = torch.cuda.mem_get_info(0)

print("Runner:", __file__)
print("Physical GPU 2 -> cuda:0:", torch.cuda.get_device_name(0))
print("Free GPU memory GiB:", round(free / 1024**3, 2))

if free < 20 * 1024**3:
    raise RuntimeError(
        "GPU 2 has less than 20 GiB free; no training started"
    )

seed_everything(0, workers=True)

from datasets import cora_dataset

def refuse_rebuild(self, *args, **kwargs):
    raise RuntimeError(
        "This smoke only permits existing caches. "
        "Dataset download or reprocessing has been blocked."
    )

# These guards affect this process only; no dataset source files are edited.
cora_dataset.CORADataset.download = refuse_rebuild
cora_dataset.CORADataset.process = refuse_rebuild

out.mkdir(parents=True, exist_ok=False)

(out / "launch_manifest.json").write_text(
    json.dumps(
        {
            "runner": str(Path(__file__).resolve()),
            "physical_gpu": 2,
            "seed": 0,
            "arguments": overrides,
            "cache_sha256_before": before,
        },
        indent=2,
    ),
    encoding="utf-8",
)

sys.argv = overrides

print("Starting Cora H1 mean smoke FROM SCRATCH")
print("Training batches: 2; validation batches: 2")
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

assert checkpoint["epoch"] == 0
assert checkpoint["global_step"] == 2
assert checkpoint["model_family"] == "beta_h1"
assert metadata["normalization"] == "mean"
assert metadata["warm_start"] is None

assert OmegaConf.select(cfg, "model.two_hop.normalization") == "mean"
assert OmegaConf.select(cfg, "model.two_hop.warm_start_checkpoint") is None
assert OmegaConf.select(cfg, "general.resume") is None
assert OmegaConf.select(cfg, "general.test_only") is None
assert int(OmegaConf.select(cfg, "train.n_epochs")) == 1
assert int(OmegaConf.select(cfg, "train.batch_size")) == 4

best_path = last_path.parent / "epoch=0.ckpt"
assert best_path.is_file(), best_path

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
    "last_checkpoint": str(last_path),
    "best_checkpoint": str(best_path),
    "losses": losses,
    "processed_cache_unchanged": True,
}

(out / "smoke_check.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("Last checkpoint:", last_path)
print("Best checkpoint:", best_path)
print("Epoch:", checkpoint["epoch"])
print("Global step:", checkpoint["global_step"])
print("Two-hop metadata:", metadata)
print("Finite losses:", losses)
print("PASS: CORA_MEAN_REAL_SMOKE")
print("No H2/Glue sampling or test evaluation was performed.")
