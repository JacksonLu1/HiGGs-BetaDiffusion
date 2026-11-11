# Community-small / ENZYMES: modified HiGGs Beta + mean two-hop + HOG

This is an offline Linux deployment package, not an edit to either Windows
production checkout. It adds isolated dataset/training/sampling modules only.
It does not patch dgd/main.py, shared collection loaders, priors, model sources,
Beta transitions, HOG, two-hop, shared sampling code, or any running DD job.

Modified root: `/path/to/HiGGs-GBD/HiGGs-main`.
Read-only source: `/path/to/HiGGs-main1/HiGGs-main`.

## Frozen comparison protocol

| Dataset | Complete train/validation/test | Formal generated graphs |
| --- | --- | --- |
| Community-small | 64 / 16 / 20 | 20 |
| ENZYMES | 376 / 94 / 117 | 117 |

The installer copies the exact original raw files, r1 manifests and 9 server
PyG caches per dataset. It NEVER reruns Louvain or random_split. Existing
different destination files are refused, never overwritten. ENZYMES means
the pinned 587-graph HOG-Diff collection, not TU's 600-graph release.

Whole-graph split RNG1234; resolution1, Louvainseed0, weightNone. The independent
HiGGs split differs from HOG-Diff's ordered author80/20 split. All descendants
stay inside their source complete graph's train/validation/test split.
Two-node H2s, edgeless hierarchy graphs and disconnected source graphs remain.
No singleton merges were applied in these exact frozen r1 manifests.

Training priors use train only. H2/H1 yclass0, Glue yclass1; presentedgeclass1.
Each stage starts from scratch: batch32,500epochs,lr0.0002,workers0,EMA0,
AdamW/amsgrad,weight_decay1e-12. Actual RNG1234 matches the original baseline
runner; the original saved train-default seed0 was not its actual RNG seed.

H2 uses Beta100steps. H1 uses Beta100steps + mean two-hop16 + HOG RWdepth4,
with pre_condition=false/fixedeta. Glue stays original discrete400steps.
Validation every10epochs; alltrain/validationbatches. Best/last strict reload.
Validation monitors: H2 `val/beta_h2_loss`, H1 `val/h1_total`, Glue
`val/epoch_NLL`. These are different objectives from the original arm's NLL;
do not call the checkpoint objectives identical. Test never selects models.

Formal sampling: seed1234 once, H1batch4/Gluebatch1, full H2/H1 components,
no quality filtering/retries. Actual-size Glue padding matches the original
small-collection baseline; empty H2 edges assemble all H1s without Glue.
Each stage strictly loads its OWN saved config, preserving H1 HOG/mean settings.
Community-small H2 always has two nodes; it alone cannot establish a Beta-H2 gain.

## 1. Upload and install, no training started

Upload `community_enzymes_beta_20261028.zip` to the modified root. Check the
SHA256 shown in chat, then:

```bash
(
set -e
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
cd "$ROOT"
unzip -n community_enzymes_beta_20261028.zip
env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg WANDB_MODE=disabled \
  "$PY" -u -B community_enzymes_beta_20261028/install.py
env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg WANDB_MODE=disabled \
  "$PY" -u -B community_enzymes_beta_20261028/install.py --apply
)
```

Expected `PASS: SMALL_COLLECTIONS_BETA_INSTALLED`. Installer writes a before
audit under interface_patch_backups, checks shared source hashes unchanged,
and neither starts nor signals any process. No GitHub access is required.

## 2. Real-cache smoke (explicitly choose physical GPUs)

These example GPUs are not assumed free; only >=8GiB memory is checked, with
no utilization or exclusive-GPU restriction. Change indices as needed.

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
"$PY" -B "$ROOT/community_enzymes_beta_20261028/launch.py" smoke --dataset community_small --gpu 2
"$PY" -B "$ROOT/community_enzymes_beta_20261028/launch.py" smoke --dataset enzymes --gpu 3
```

Each dataset's smoke stages run sequentially: oneepoch, atmost2train/valbatches,
finite losses/gradients, real H1input X12/E6/y14, strictbest/last reload.
Allthree must pass before formal training. A printed PID is submission only.

## Persistent log viewing (works after reconnect)

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
DATASET=community_small  # or enzymes
ACTION=smoke            # smoke, train, sample
STAGE=all               # all, h2, h1, glue; sample for sample log
INFO="$ROOT/logs/${DATASET}_beta_latest_${ACTION}.json"
if [ -f "$INFO" ]; then
    LOG="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["log"])' "$INFO")"
    OUT="$(cat "${LOG}.out")"
    if [ "$STAGE" = all ]; then
        PID="$(cat "${LOG}.pid")"
    else
        LOG="$OUT/$STAGE.log"
        PID="$(cat "$OUT/$STAGE.pid" 2>/dev/null || true)"
    fi
    printf 'DATASET=%s ACTION=%s STAGE=%s\nLOG=%s\nOUTPUT=%s\n' "$DATASET" "$ACTION" "$STAGE" "$LOG" "$OUT"
    if [ -n "$PID" ]; then
        ps -p "$PID" -o pid,ppid,stat,etime,pcpu,pmem,args || true
    fi
    tail -n 60 -F "$LOG"
else
    echo "No launch record for the selected dataset/action."
fi
```

Ctrl+C exits tail, not background training. Do not launch again to view progress.

## 3. Formal training, only after successful smoke

Both example commands select their own successful smoke run. Within each
dataset H2,H1,Glue train concurrently on that dataset's selected GPU (as in
the original v4 baseline schedule). Three stages500epochs each. No sampling
starts automatically, and no old DD/Planar/SBM model is reused as initialization.

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
DATASET=community_small
GPU=2
SMOKE="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$ROOT/logs/${DATASET}_beta_latest_smoke.json")"
"$PY" -B "$ROOT/community_enzymes_beta_20261028/launch.py" train \
  --dataset "$DATASET" --gpu "$GPU" --smoke-output "$SMOKE"
```

Repeat with `DATASET=enzymes` and `GPU=3` after its smoke has passed.
Success marker `PASS: <DATASET>_BETA_MEAN_HOG_THREE_STAGE_FORMAL_TRAINING_COMPLETE`.

## 4. Full-chain sampling, only after all three stages complete

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
DATASET=community_small
GPU=2
TRAIN="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$ROOT/logs/${DATASET}_beta_latest_train.json")"
"$PY" -B "$ROOT/community_enzymes_beta_20261028/launch.py" sample \
  --dataset "$DATASET" --gpu "$GPU" --training-output "$TRAIN"
```

Repeat with enzymes/GPU3. Defaultcounts20/117 cannot accidentally become40.
`--count 1` is an exploratory probe, not a replacement for the formal count.
Graphs are under reported OUTPUT/sample/sampling/. Completion means sampling
finished, NOT measured quality improvement. Metrics remain separate: use the
same frozen real test IDs and original small-collection TV+LCC evaluator, r1/seed0
for evaluation communities. Do not replace historical SBM mixed metrics.

Keep this uploaded package unchanged until training and sampling audits finish.
