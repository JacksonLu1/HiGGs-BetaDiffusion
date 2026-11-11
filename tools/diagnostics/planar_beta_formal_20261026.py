"""Formal Planar training: Beta-H2, Beta-H1 mean+HOG, OriginalGlue; 500 epochs from scratch."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

root, out = [Path(p).resolve() for p in sys.argv[1:3]]
stage = sys.argv[3]
os.chdir(root)
sys.path[:0] = [str(root / "dgd"), str(root)]
manifest = root / "data_sources/splits/planar_hierarchy_r2_s1234_l0.json"
expected = "05b42aa49d076985096e1c0e3408a452a433a4173953c60b27b438d8b4971bcc"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert sha(manifest) == expected, "Frozen manifest differs"

if stage == "all":
    import fcntl
    lock_path = root / "logs/training/planar_beta_formal.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert out.parent == root / "outputs" and not out.exists()
    inputs = [Path(__file__).resolve(), manifest, root / "data_sources/planar/planar_64_200.pt"]
    for h in ("2", "1", "1.5"):
        for split in ("train", "val", "test"):
            inputs.append(root / "data/planar/processed" /
                f"proc_{split}_v1_h{h}_max500_{expected}.pt")
    for rel in (
        "dgd/main.py", "dgd/datasets/planar_beta_adapter.py",
        "dgd/datasets/graph_collection.py", "dgd/datasets/collection_sources.py",
        "dgd/diffusion_model_beta_h2.py", "dgd/diffusion_model_beta_h1.py",
        "dgd/diffusion_model_discrete.py", "dgd/diffusion/distributions.py",
        "higgs/higgs_utils.py", "higgs/sampling.py",
        "configs/model/beta_h1_twohop_mean_hog_rw.yaml",
    ):
        inputs.append(root / rel)
    before = {str(p): sha(p) for p in inputs}
    out.mkdir()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_sequence(keys, gpu):
        for key in keys:
            assert before == {str(p): sha(p) for p in inputs}, "Input/source changed"
            log = out / f"{key}.log"
            print(f"START {key} / GPU {gpu}: {log}", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            with log.open("x") as handle:
                child = subprocess.Popen(
                    [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                     str(root), str(out / key), key],
                    cwd=root, stdout=handle, stderr=subprocess.STDOUT, env=env)
                (out / f"{key}.pid").write_text(str(child.pid) + "\n")
                rc = child.wait()
            if rc:
                print(log.read_text(errors="replace")[-10000:], flush=True)
                raise RuntimeError(f"{key} formal training failed; see {log}")
            report = json.loads((out / key / "formal_result.json").read_text())
            report.pop("losses", None)
            print(json.dumps(report, indent=2), flush=True)
            print(f"PASS: PLANAR_{key.upper()}_FORMAL_TRAINING_COMPLETE", flush=True)

    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(run_sequence, ("h2", "glue"), 2),
                pool.submit(run_sequence, ("h1",), 3)]
        for future in as_completed(jobs):
            try:
                future.result()
            except Exception as exc:
                errors.append(repr(exc))
                print("FAILED GROUP:", repr(exc), flush=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    assert before == {str(p): sha(p) for p in inputs}, "Input/source changed"
    (out / "input_hash_check.json").write_text(json.dumps(
        {"unchanged": True, "sha256": before}, indent=2))
    print("PASS: PLANAR_BETA_MEAN_HOG_THREE_STAGE_FORMAL_TRAINING_COMPLETE", flush=True)
    print("Formal training finished. No full-chain sampling or test evaluation.")
    sys.exit(0)

spec = {
    "h2": ("2", "beta_h2", "val/beta_h2_loss"),
    "h1": ("1", "beta_h1_twohop_mean_hog_rw", "val/h1_total"),
    "glue": ("1.5", "original_glue", "val/epoch_NLL"),
}
h, group, monitor = spec[stage]
assert out.parent.parent == root / "outputs" and not out.exists()

import torch
from pytorch_lightning import Callback, seed_everything
from omegaconf import OmegaConf
from datasets import graph_collection as gc
import main as entry

assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU training"
torch.cuda.set_device(0)
assert torch.cuda.mem_get_info()[0] >= 8 * 1024**3, "Less than 8 GiB free"
print("Physical GPU:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Logical GPU: cuda:0", torch.cuda.get_device_name(0))
seed_everything(0, workers=True)

paths = {
    (root / "data/planar/processed" /
     f"proc_{s}_v1_h{h}_max500_{expected}.pt").resolve()
    for s in ("train", "val", "test")
}

def cache_guard(self, *args, **kwargs):
    assert {Path(p).resolve() for p in self.processed_paths} == paths
    assert all(p.is_file() for p in paths), "Missing cache; rebuilding forbidden"

def forbidden(*args, **kwargs):
    raise RuntimeError("Downloading/reprocessing/repartitioning is forbidden")

def frozen(name, source, path, resolution, split_seed, louvain_seed):
    assert name == "planar" and Path(path).resolve() == manifest
    assert (resolution, split_seed, louvain_seed) == (2, 1234, 0)
    assert sha(manifest) == expected
    return gc.read_manifest(manifest)

gc.GraphCollectionDataset._download = cache_guard
gc.GraphCollectionDataset._process = cache_guard
gc.GraphCollectionDataset.process = forbidden
gc.ensure_manifest = frozen

class FiniteAudit(Callback):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        assert torch.isfinite(loss).all(), "Nonfinite training loss"
        module.log("formal/train_loss", loss.detach(), on_step=False,
                   on_epoch=True, batch_size=batch.num_graphs)
    def on_validation_batch_end(self, trainer, module, outputs,
                                batch, batch_idx, dataloader_idx=0):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        assert torch.isfinite(loss).all(), "Nonfinite validation loss"
        module.log("formal/val_loss", loss.detach(), on_step=False,
                   on_epoch=True, batch_size=batch.num_graphs)
    def on_after_backward(self, trainer, module):
        if trainer.global_step % 1000 == 0:
            assert all(torch.isfinite(p.grad).all()
                       for p in module.parameters() if p.grad is not None)

Trainer = entry.Trainer
def checked_trainer(*args, **kwargs):
    kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + [FiniteAudit()]
    return Trainer(*args, **kwargs)
entry.Trainer = checked_trainer

job = f"planar_{stage}_mean_hog_formal"
sys.argv = [str(root / "dgd/main.py"),
    "dataset=planar", f"dataset.h={h}", f"model={group}",
    f"dataset.datadir={root / 'data/planar'}",
    f"dataset.manifest={manifest}",
    f"dataset.source={root / 'data_sources/planar/planar_64_200.pt'}",
    "dataset.resolution=2", "dataset.split_seed=1234",
    "dataset.louvain_seed=0", "dataset.max_size=500",
    f"general.name={job}", "general.gpus=1", "general.wandb=disabled",
    "general.resume=null", "general.test_only=null",
    "general.check_val_every_n_epochs=10", "general.val_check_interval=1.0",
    "general.sample_every_val=1000000000", "general.log_every_steps=5",
    "general.samples_to_generate=4", "general.final_model_samples_to_generate=40",
    "general.samples_to_save=0", "general.chains_to_save=0",
    "++general.run_test_after_fit=false",
    f"++general.checkpoint_every_n_epochs={1 if stage == 'h1' else 10}",
    "++general.limit_train_batches=1.0", "++general.limit_val_batches=1.0",
    "train.n_epochs=500", "train.batch_size=4", "train.num_workers=0",
    "train.lr=0.0002", "train.seed=0", "train.ema_decay=0",
    "train.save_model=true", f"hydra.run.dir={out}"]
entry.main()

from higgs.higgs_utils import _load_checkpoint_cpu
last = out / "checkpoints" / job / "last.ckpt"
state = _load_checkpoint_cpu(last)
expected_steps = {"h2": 16000, "h1": 146500, "glue": 301500}[stage]
assert state["epoch"] == 499 and state["global_step"] == expected_steps
cfg = state["hyper_parameters"]["cfg"]
cfg = cfg if OmegaConf.is_config(cfg) else OmegaConf.create(cfg)
assert cfg.dataset.name == "planar" and float(cfg.dataset.h) == float(h)
assert cfg.general.resume is None and cfg.general.test_only is None
callbacks = [v for v in state["callbacks"].values()
             if isinstance(v, dict) and v.get("monitor") == monitor]
assert len(callbacks) == 1
cb = callbacks[0]
assert math.isfinite(float(cb["best_model_score"]))
best = last.parent / Path(cb["best_model_path"]).name
assert best.is_file()

# Strict load tests the persistent Planar prior class and HOG metadata.
model = entry.get_model_class(cfg).load_from_checkpoint(
    str(last), map_location="cpu", cfg=cfg, strict=True)
assert model.dataset_info.manifest_sha256 == expected
assert type(model.node_dist).__module__ == "datasets.planar_beta_adapter"
assert all(torch.isfinite(p).all() for p in model.parameters())
if stage == "h1":
    assert state["h1_two_hop"]["normalization"] == "mean"
    assert state["h1_two_hop"]["warm_start"] is None
    assert state["h1_hog_rw"]["rw_depth"] == 4
    assert state["h1_hog_rw"]["two_hop"] is True
    widths = {k: getattr(model.model, "mlp_in_" + k)[0].in_features
              for k in ("X", "E", "y")}
    assert widths == {"X": 12, "E": 6, "y": 14}, widths
else:
    widths = None

best_state = _load_checkpoint_cpu(best)
best_cfg = best_state["hyper_parameters"]["cfg"]
best_cfg = best_cfg if OmegaConf.is_config(best_cfg) else OmegaConf.create(best_cfg)
assert best_cfg.dataset.name == "planar"
assert float(best_cfg.dataset.h) == float(h)
best_model = entry.get_model_class(best_cfg).load_from_checkpoint(
    str(best), map_location="cpu", cfg=best_cfg, strict=True)
assert best_model.dataset_info.manifest_sha256 == expected
assert all(torch.isfinite(p).all() for p in best_model.parameters())
assert cfg.train.n_epochs == 500 and cfg.train.batch_size == 4
assert cfg.general.limit_train_batches == 1.0
assert type(cfg.general.limit_train_batches) is float
assert cfg.general.limit_val_batches == 1.0
assert type(cfg.general.limit_val_batches) is float

csvs = list(out.rglob("metrics.csv"))
assert len(csvs) == 1
with csvs[0].open() as handle:
    rows = list(csv.DictReader(handle))
losses = {k: [float(r[k]) for r in rows if r.get(k)]
          for k in ("formal/train_loss", "formal/val_loss")}
assert all(v and all(math.isfinite(x) for x in v) for v in losses.values())
result = {"stage": stage, "epoch": int(state["epoch"]),
          "global_step": int(state["global_step"]), "monitor": monitor,
          "best_score": float(cb["best_model_score"]), "best": str(best),
          "last": str(last), "strict_reload": True,
          "strict_best_reload": True,
          "best_epoch": int(best_state["epoch"]),
          "best_sha256": sha(best), "last_sha256": sha(last),
          "finite_loss_counts": {k: len(v) for k, v in losses.items()},
          "losses": losses,
          "input_widths": widths, "manifest_sha256": expected,
          "two_hop": state.get("h1_two_hop"), "hog": state.get("h1_hog_rw")}
(out / "formal_result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(
    {k: v for k, v in result.items() if k != "losses"},
    indent=2), flush=True)

