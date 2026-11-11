# TU three-benchmark server package: latest Beta + mean two-hop + HOG

This is an offline Linux upload package, not an edit of the Windows production
checkout. It adapts IMDB-BINARY, COLLAB and NCI1 only. Existing datasets, models,
samplers, caches and previous results are not overwritten.

- Modified target: `/path/to/HiGGs-GBD/HiGGs-main`
- Original read-only data source: `/path/to/HiGGs-main1/HiGGs-main`
- Interpreter: `/path/to/conda/envs/higgs/bin/python`

## Frozen experiment

| Dataset | Complete graphs | Train / validation / test | Epochs H2 / H1 / Glue | Batch H2 / H1 / Glue | Formal samples |
|---|---:|---|---|---|---:|
| IMDB-BINARY (`imdb_binary`) | 1000 | 640 / 160 / 200 | 1000 / 1000 / 1000 | 32 / 32 / 32 | 200 |
| COLLAB (`collab`) | 5000 | 3200 / 800 / 1000 | 500 / 500 / 500 | 32 / 16 / 8 | 1000 |
| NCI1 (`nci1`) | 4110 | 2630 / 658 / 822 | 500 / 500 / 500 | 32 / 32 / 32 | 822 |

The original conversation's exact frozen complete-graph split IDs AND initial
community partitions are reused: resolution1, split seed1234, Louvain seed0,
weight=None. The installer copies original source JSON, manifest, dataset YAML
and all27 hierarchy caches byte-for-byte. It checks cached topology, source
graph IDs, conditions, singleton retention and Boolean Glue membership masks.
Missing original caches are errors: it does not regenerate or repartition.
Wait for original `TU_BENCHMARK_PREPARATION_COMPLETE` before installing here.
Original training can keep reading its files while they are copied.

These are topology-only experiments. NCI1 chemical atom/bond labels and TU graph
classes are not generation targets. No size filtering, deduplication, singleton
deletion, train/test reassignment or target-test-size conditioning is introduced.

Training seed0; fresh models, no warm start/resume; lr0.0002, AdamW/amsgrad,
weight_decay1e-12, EMA0, workers0. Validation every10epochs. Each stage selects
its own validation-best checkpoint, not test MMD or necessarily the last epoch.

- H2: existing Beta diffusion,100steps, monitor `val/beta_h2_loss`.
- H1: existing Beta diffusion,100steps, gated mean two-hop16channels AND HOG
  RW/SPD depth4 retained; X12/E6/y14 inputs; monitor `val/h1_total`.
- Glue: existing original discrete diffusion,400steps, monitor `val/epoch_NLL`.

Beta transitions, objectives, threshold0.5, HOG algorithm and shared model source
are unchanged. Dataset-scoped parameter-free singleton forward padding avoids
undefined unbiased std; padded vertices are masked and cropped, never added to
the generated graph. Empty edge CE is handled only in the scoped Glue subclass.
No silent eigensolver fallback, graph rejection, edge budget or new repair is
enabled; numerical errors stop and preserve the failed run for diagnosis.

Sampling aligns with the original TU arm: seed0 ONCE per whole series,
H2 upstream-LCC, H1 upstream-LCC, H1 batch4, Glue batch1. Full assembled graphs are
saved; evaluation takes independent LCC for each real and generated graph.
Glue uses the actual maximum pair-size Boolean membership padding. The existing
upstream Glue minimum-connection behavior is inherited, not a new intervention.

## Install and launch ONCE

Upload the ZIP to the MODIFIED project root. Check the ZIP SHA256 shown with
the download, then run:

```bash
(
set -e
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
cd "$ROOT"
unzip -n tu_benchmarks_beta_20261103.zip
B="$ROOT/tu_benchmarks_beta_20261103"
"$PY" -B "$B/install.py" --root "$ROOT"
"$PY" -B "$B/install.py" --root "$ROOT" --apply
"$PY" -u -B "$B/launch.py" pipeline --root "$ROOT" \
    --dataset all --max-parallel 3 --min-free-gib 32
)
```

The first installer command is read-only. Apply writes additive files plus an
audit inventory under interface_patch_backups; identical files are reusable,
different existing files are refused. Rollback removes only files that failed
invocation created. Original source and data are never written.

Each dataset: three real-cache stage smoke tests -> one smoke full-chain graph
-> fresh concurrent H2/H1/Glue formal training -> validation-best formal full
test-count sampling -> CPU mixed Table1 / mean Table3. Smoke checkpoints are
NOT used as formal weights; smoke samples are NOT evaluated as test results.

Up to3 datasets run concurrently, each on the eligible GPU with the most
currently free VRAM, excluding GPUs this queue already assigned. The default
32GiB threshold is for three stage processes sharing one GPU, not a measured
peak guarantee. `--gpus 0 3` restricts candidates; `--max-parallel 1` is sequential.
There is NO utilization or other-compute-process exclusion gate. Other users
can allocate memory later: this is not a reservation. No unrelated job is
stopped. Per-dataset locks prevent duplicate submissions. Pending jobs wait
for sufficient memory. Do not launch again to view progress.

## Reconnect-safe progress

All datasets snapshot:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
"$PY" -B "$ROOT/tu_benchmarks_beta_20261103/launch.py" status \
    --root "$ROOT" --dataset all
```

Continuous total log:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
INFO="$ROOT/logs/tu_benchmarks_beta_latest.json"
if [ -f "$INFO" ]; then
    LOG="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["log"])' "$INFO")"
    printf 'LOG=%s\n' "$LOG"
    tail -n 60 -F "$LOG"
else
    echo "尚未找到三个新数据集的启动记录。"
fi
```

One dataset/stage in each terminal; edit only DATASET and STAGE:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
DATASET=imdb_binary
STAGE=h1
INFO="$ROOT/logs/${DATASET}_tu_beta_latest_pipeline.json"
if [ -f "$INFO" ]; then
    OUT="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1])).get("output") or "")' "$INFO")"
    if [ -n "$OUT" ]; then
        case "$STAGE" in
            h2|h1|glue) LOG="$OUT/train/$STAGE.log" ;;
            *) LOG="$OUT/$STAGE.log" ;;
        esac
        printf 'DATASET=%s\nOUTPUT=%s\nLOG=%s\n' "$DATASET" "$OUT" "$LOG"
        if [ -f "$LOG" ]; then
            tail -n 50 -F "$LOG"
        else
            echo "该阶段尚未开始；可能正在短测试，请勿重复提交。"
        fi
    else
        echo "该数据集正在排队。"
    fi
else
    echo "尚未找到该数据集的启动记录。"
fi
```

DATASET: imdb_binary/collab/nci1. STAGE: h2/h1/glue/sample/evaluation/
smoke_sample. Ctrl+C exits only the log viewer, not detached training.

## Frozen mixed evaluation

| Table1 metric | Kernel |
|---|---|
| Nodes | Gaussian-TV |
| Degree | Gaussian-EMD |
| Clustering | Gaussian-EMD |
| Eccentricity | Gaussian on legacy50-bin ordered-pair shortest-path histograms |
| Spectre | Gaussian-EMD,200bins,all eigenvalues and all requested graphs |

The original checkout's metric helper functions are imported read-only in a
separate CPU process. Protocol JSON records their source hashes, normalization,
histogram ranges, bandwidths, independent-LCC handling and full sample hashes.
No all-TV/all-EMD substitution and no historical Spectre list truncation.
Table3 community counts/sizes use evaluation Louvain r1/seed0/weightNone,
separately from the unchanged initial training partition. Edgeless graphs are
retained, with one community per node. Table3 displays means only; CSV can keep
std for audit. Exactly200/1000/822 samples are required respectively.

This is the user's frozen author-postprocessing mixed branch, not a claim to
recover every undocumented historical paper execution setting. These three
datasets have no published HiGGs Table1/Table3 rows: compare original-code
reproduction against modified-code measurements, not invented paper numbers.

Completion per dataset:
`PASS: <DATASET>_BETA_MEAN_HOG_TRAIN_SAMPLE_EVALUATE_COMPLETE`.
Overall: `PASS: TU_BENCHMARKS_BETA_MEAN_HOG_PIPELINES_COMPLETE`.
Submission, stage training completion, sampling completion and metric
completion are different states; none proves a quality improvement.

See VERIFICATION.md for locally run checks and remaining server-only checks.
