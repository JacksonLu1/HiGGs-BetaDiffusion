# Latest modified-model mixed-kernel re-evaluation

This package evaluates existing completed sample graphs only. It never trains,
samples, changes weights, rebuilds data, or overwrites a previous result.

Modified root (sample owner and destination of NEW results):
`/path/to/HiGGs-GBD/HiGGs-main`

Original root (read-only metric implementation and matching frozen manifests):
`/path/to/HiGGs-main1/HiGGs-main`

The only original-root coordination write is the shared evaluation lock file,
`run_logs/.mixed_collection_evaluation.lock`. No project source is installed or
patched in either root. Do not delete this lock file to bypass a running batch.

## Why re-evaluate

The completed Planar, Community-small, ENZYMES, Tree and Grid evaluations used
`--mode tv` for all five Table 1 metrics. Those values are not the frozen C4 mixed
protocol. Their existing generation results can be re-evaluated without another
training or sampling run. DD is evaluated only after all 184 samples and their
final hash audit exist; incomplete/probe DD outputs remain explicitly pending.

The original project's recent `mixed_collection_eval_20261030` tool defaults to
BOTH original and improved variants. If that batch is already running, the
improved graphs may already be queued. This launcher shares its lock and refuses
to create a concurrent duplicate. It lists the existing improved arms and log.
Completed exact matching mixed results are reused after provenance/CSV checks.

## Frozen mixed kernel definition

| Metric | Implementation |
|---|---|
| Nodes | Aggregate node-count bincount histogram; Gaussian-TV, sigma 1 |
| Degree | Per-graph degree histogram; Gaussian-EMD, sigma 1, scaling 1 |
| Clustering | 100 bins over [0,1]; Gaussian-EMD, sigma 0.1, distance_scaling 100 |
| Eccentricity | Author `radius_stats(compute_emd=True,is_parallel=True)`: Gaussian, sigma 1, not Gaussian-EMD |
| Spectre | All graphs, full normalized-Laplacian spectra, 200 bins over (-1e-5,2); Gaussian-EMD, sigma 1, scaling 1 |

Eccentricity retains the historical function's actual descriptor: take LCC, all
ordered shortest-path distances including self-distance zero, then a 50-bin
histogram with the original per-graph automatic range. It is not a new vector of
per-node eccentricities. The existing helper's normalization is preserved.

The calculations directly call the same repository functions used by the
original project's mixed re-evaluation, not renamed TV output rows. The bundled
worker is copied from that tool with only its allowed result-directory guard
extended; mathematical functions and graph preprocessing are unchanged.

## Scope and comparison rules

- Same complete-graph test IDs and same source graph hashes as each prior TV
  evaluation. No fresh training/community partitioning.
- Keep the prior collection evaluation's independent LCC handling. This matches
  the KERNEL mapping of historical SBM C4, not a claim that every dataset has
  identical full-graph/LCC preprocessing.
- Table 3 evaluation-only Louvain: resolution 1, seed 0, weight None. Training
  hierarchy resolution is unchanged, including r2 for Planar/DD/Grid.
- Table 3 must numerically match its previous version. Nodes MMD must match its
  old TV value, because the same input graphs and TV node kernel are retained.
  The worker checks both rather than silently accepting changed input policies.
- Other four MMD metrics are recalculated. Do not transfer an improvement count
  or ranking obtained under all-TV into the new mixed comparison.
- These six extension datasets have no published HiGGs Table 1/Table 3 rows in
  the supplied paper. Their original-code experiment is the baseline; do not
  invent published scores or copy SBM scores into these datasets.
- DD early-stop training budget is preserved if present in its sampler metadata.
  It must not be described as an equal-training-budget comparison.
- Existing mixed SBM/Cora/Facebook results are not re-evaluated by this package.

## Upload and launch on Linux

Upload the ZIP to the MODIFIED root and verify its SHA256 from the chat. Then:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
cd "$ROOT"
unzip -n latest_hog_mixed_eval_20261030.zip
B="$ROOT/latest_hog_mixed_eval_20261030"
"$PY" -u -B "$B/launch_improved.py" plan
"$PY" -u -B "$B/launch_improved.py" launch
```

Default datasets: planar, dd, community_small, enzymes, tree, grid; IMPROVED only.
To handle only a specified subset, give the same `--datasets` arguments to both
commands. Example for DD when its full sampling finishes:

```bash
"$PY" -u -B "$B/launch_improved.py" launch --datasets dd
```

The launcher automatically locates completed latest mean+HOG TV reports using
their metadata and sample hashes. Do not manually change graph lists to pass a
failed input check. Prior evaluations are discovery/provenance evidence; their
MMD values are not used as replacement mixed values.

Messages:

- `READY`: existing samples verified; mixed values still need calculation.
- `REUSE ... RESULT=...`: matching complete mixed results already exist.
- `PENDING dd`: all 184 DD samples not yet confirmed; no partial calculation.
- `SHARED_MIXED_BATCH_ALREADY_RUNNING`: no second job started; inspect the
  existing batch's improved queue. If it covers only original arms, wait for it
  to finish and then run this launcher again.
- `ALL_AVAILABLE_RESULTS_ALREADY_COMPLETE`: no duplicate calculation; use the
  RESULT directories printed above. Pending DD may still be pending.

New calculations are CPU-only, sequential, detached from SSH, with BLAS thread
counts 1. They do not require a free GPU and do not stop any running job.

## Status and continuous viewing

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
B="$ROOT/latest_hog_mixed_eval_20261030"
"$PY" -B "$B/launch_improved.py" status
```

Status also locates the latest original-root shared batch if this package has
not started one. It shows its improved arms, progress and log path.

For a NEW batch launched by this package:

```bash
INFO="$ROOT/logs/latest_hog_mixed_eval.json"
if [ -f "$INFO" ]; then
    LOG="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["log"])' "$INFO")"
    tail -n 60 -F "$LOG"
else
    "$PY" -B "$B/launch_improved.py" status
fi
```

Within-dataset progress is in the dataset log, e.g. `OUT/tree_improved.log`.
Ctrl+C stops log viewing, not the detached calculation.

Completion marker for a newly launched batch:
`PASS: LATEST_HOG_MIXED_EVALUATION_COMPLETE`

## New output

`ROOT/higgs/outputs/mixed_collection_tables_latest_hog_<timestamp>/`

- `latest_hog_table1_mixed.csv`: dataset, metric, exact kernel, latest value.
- `latest_hog_table3_means.csv`: real/generated means only, no displayed std.
- `plan.json`, `progress.json`, `batch_summary.json`: completed, failed, reused
  and pending datasets, with exact input/result locations.
- Each newly calculated dataset directory has `table1_mixed.csv`,
  `table3_summary.csv`, `table3_per_graph.csv`, `mixed_evaluation.json`.
- Old TV results and previously completed mixed results remain untouched.

Reused dataset directories may remain under the original root because the
earlier shared tool already calculated that IMPROVED arm there. Their original
and improved graph ownership must not be inferred from output location alone;
the manifest/sample hashes and `variant=improved` identify the arm.

## Verification

See `VERIFICATION.md`. Local tests use tiny synthetic graphs and temporary
directories. They do not establish that any real server dataset was recalculated.
