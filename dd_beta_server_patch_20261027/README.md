# DD: modified Beta-H2 / Beta-H1 + mean two-hop + HOG / original Glue

This is a new Linux deployment package, not a Windows production-source edit.
Modified root: `/path/to/HiGGs-GBD/HiGGs-main`.
Read-only original data source: `/path/to/HiGGs-main1/HiGGs-main`.
No commands stop or signal the original DD sampling process.

## Frozen experiment

- Reuse the exact original DD.zip, frozen manifest, and all 9 processed caches.
- 918 eligible complete graphs; train587 / validation147 / test184.
- Whole-graph split seed1234, training Louvain resolution2 / seed0 / weightNone.
- Structure only: H2/H1 condition class0, Glue class1; real edge class1.
- Existing files with different hashes are refused, never overwritten.
- Original model/prior fixes are NOT copied into modified shared model files.
- A DD-only floating/range-safe node-count prior preserves the sampling law.
- New H1 retains mean two-hop and HOG RW depth4, precondition=false, fixed eta.
- New H1 input widths must actually equal X12/E6/y14 during smoke/reload.

## 1. Install and run a short real-cache test

Upload this ZIP to the modified root, then run on Linux:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
cd "$ROOT"
unzip -n dd_beta_server_patch_20261027.zip
env CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg WANDB_MODE=disabled \
  PYTHONDONTWRITEBYTECODE=1 "$PY" -u -B \
  dd_beta_server_patch_20261027/install_dd_beta.py
"$PY" -B dd_beta_server_patch_20261027/launch_dd.py smoke --gpu 0
```

The installer modifies only the DD dataset route/checkpoint cadence in main.py,
with a backup first, and adds DD-specific adapter/drivers. All existing model,
Beta transition, HOG, two-hop, prior and sampler sources remain unchanged.
It expects the proven post-Planar main.py; do not bypass a hash/anchor failure.

Smoke trains each stage for one epoch with only 2 train and 2 validation batches,
batch4, seed0, no warm-start/resume. It checks finite loss/gradients, callback-best
and last strict checkpoint loads, HOG/mean metadata, and input/source hashes.
This is integration testing, NOT formal training, sampling or quality evidence.
Finish marker: `PASS: DD_BETA_MEAN_HOG_THREE_STAGE_SMOKE`.

## 2. Formal training, only after smoke completes

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
"$PY" -B "$ROOT/dd_beta_server_patch_20261027/launch_dd.py" train --gpu-a 0 --gpu-b 1
```

H2 then Glue use GPU0; H1 concurrently uses GPU1. Each stage is fresh training
for500 epochs, batch4, lr0.0002, workers0, EMA0, train seed0. All train/validation
batches are used. Validation every10 epochs. No utilization/exclusive GPU gate,
only an 8GiB free-memory check. Choose different physical indices explicitly if
needed; unrelated tasks are never stopped. No automatic test evaluation/sampling.

The launcher rechecks the latest smoke outputs before formal launch. Each stage
records formal_result.json; callback-selected best is used, not the largest epoch.
Total marker: `PASS: DD_BETA_MEAN_HOG_THREE_STAGE_FORMAL_TRAINING_COMPLETE`.
The original DD Glue was stopped around250 epochs, so final comparisons must
report actual training budgets and not claim both arms trained500 Glue epochs.

## 3. Full-chain sampling, only after all three formal stages complete

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
"$PY" -B "$ROOT/dd_beta_server_patch_20261027/launch_dd.py" sample --gpu 0 --count 184
```

Sampling seed1234 matches the original DD launcher, H1batch4 / Gluebatch1.
Loader uses each checkpoint's own saved config with strict loading, preserving
mean/HOG settings. H2-to-H1 class0 structure-only adaptation and empty Glue
handling occur only inside this sampling process, just as in the Planar bridge.
Upstream LCC component policy, present-class1 routing, original Beta transitions,
thresholds, original discrete Glue and existing dynamic padding are preserved.
No per-graph reseeding, output count forcing, quality filtering, retries or
resampling are introduced. 184 nonempty simple graphs must be saved and inputs
remain hash-identical. One-graph probes are allowed via --count1 and labeled
exploratory; they cannot replace the formal184-graph result.

Marker: `PASS: DD_MEAN_HOG_FULLCHAIN_184_COMPLETE`.
Metric evaluation is deliberately separate and must reuse the frozen DD test IDs
and original DD evaluation mode/component policy. No DD values exist in the
HiGGs paper; do not invent a DD paper baseline.

## Read logs after reconnecting

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
MODE=smoke  # smoke, train, or sample
LOG="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["log"])' "$ROOT/logs/dd_latest_${MODE}.json")"
OUT="$(cat "${LOG}.out")"
printf 'LOG=%s\nOUTPUT=%s\n' "$LOG" "$OUT"
ps -p "$(cat "${LOG}.pid")" -o pid,stat,etime,pcpu,pmem,args || true
tail -n 70 -F "$LOG"
```

Ctrl+C stops only log viewing. It does not stop detached training/sampling.
Do not launch a stage twice. The drivers also take dataset-specific locks.

## Local verification boundary

The package was syntax-/unit-tested with synthetic metadata, CPU priors and
mocked GPU/launcher calls. This does not establish real DD training success,
GPU memory requirements, speed, sampling quality, or any server-side changes.
Run the server smoke and inspect its completion marker before formal training.
