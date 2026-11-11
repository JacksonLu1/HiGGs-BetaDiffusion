# Tree / Grid: new Beta-H2 + Beta-H1 + mean two-hop + HOG

This is a Linux-server upload bundle, not an installation into the Windows
production checkout. The original server root is READ ONLY:

- Original: /path/to/HiGGs-main1/HiGGs-main
- Modified target: /path/to/HiGGs-GBD/HiGGs-main
- Interpreter: /path/to/conda/envs/higgs/bin/python

No model/diffusion/loss/shared sampler is overwritten. Existing Community,
ENZYMES, Planar, DD, Cora, Facebook and SBM runs/files remain unchanged. No job
is signaled, suspended or killed. An install backup JSON inventories all
protected source hashes and additive files. Conflicting existing files are
refused rather than overwritten.

## Frozen comparison

| Dataset | Original complete-graph split | Resolution | max_size | Samples |
|---|---|---|---|---|
| tree | author 128/32/40, NOT reshuffled | 1 | 64 | 40 |
| grid | 64/16/20, torch permutation seed1234 | 2 | 361 | 20 |

Manifests are pinned to the originals' exact bytes:

- tree_higgs_r1_s1234_l0.json: 59c3e8b9e7eed327098018da3caf691c780c1a1c932900c224d59ebb448a7e37
- grid_higgs_r2_s1234_l0.json: da59020eeea53566a3f60e480c39e481fc797ba2314161a0e6e691603de38338

The installer copies the raw sources, configs, manifests and all18 processed
caches DIRECTLY from the original SERVER checkout. It never reruns Louvain,
downloads graphs, repartitions data or generates new caches. It verifies cache
counts, source graph IDs, node/edge/y classes and Boolean Glue masks. Missing
caches are errors, not permission to reconstruct them.

## Training and sampling

Each dataset's independent pipeline:

1. Three one-epoch, two-batch real-data training/reload smoke tests.
2. One full-chain sample from the smoke checkpoints (not evaluated). This
   exercises the actual interfaces; if sampled H2 has no edges, it remains
   edgeless and Glue has no pairs, rather than resampling to force a pass.
3. Fresh H2/H1/Glue, each500epochs, batch32, seed1234, workers0, lr0.0002,
   AdamW/amsgrad, weight_decay1e-12, EMA0. All3stages train concurrently.
4. Strictly load each stage's validation-best checkpoint, never test-selected.
5. Generate40Tree or20Grid complete graphs, seed1234 once, H1batch4/Gluebatch1.
6. Compute Table1/3 and compare saved matching original-baseline CSVs if present.

New H2 is Beta100steps. New H1 is Beta100steps + gated mean two-hop16channels
+ HOG-Diff RW/SPD feature transfer depth4. Glue remains original discrete400
steps. H1 input widths X12/E6/y14. No warm start or checkpoint resume is used.
The Beta objectives differ from original discrete NLL: monitors are
val/beta_h2_loss, val/h1_total and val/epoch_NLL respectively. Validation every
10epochs, best+laststrict reload and finite-loss/gradient audits are required.

Full H2/H1 components are retained, with no post-hoc repairs/rejection or
manually added edges.
Glue uses actual maximum pair padding and pos Boolean from the outset. H2/H1
conditionclass0, Glueclass1 from frozen cache. The train-only empirical
node-count prior is preserved; the isolated subclass only makes log_prob safe
for held-out count ranges and integer inputs.

At submission, nvidia-smi ranks ALL GPUs by memory.free. Tree gets the largest,
Grid the second-largest, on separate GPUs; single-dataset launch uses largest.
Minimum8192MiB free per chosen GPU. No utilization/compute-process gate. Existing
jobs may share GPUs and memory can change after submission; this is not a peak
memory reservation. Once selected, each pipeline keeps its assigned GPU.

## Upload / install / start ONCE

Upload tree_grid_beta_20261029.zip to the modified target ROOT above, then:

```bash
(
set -e
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
cd "$ROOT"
unzip -n tree_grid_beta_20261029.zip
B="$ROOT/tree_grid_beta_20261029"
"$PY" -B "$B/install.py" --root "$ROOT"
"$PY" -B "$B/install.py" --root "$ROOT" --apply
"$PY" -B "$B/launch.py" pipeline --root "$ROOT" --dataset both
)
```

No need to activate conda with this absolute interpreter. Submission prints
GPU/PID/LOG/OUTPUT; it does NOT mean training or quality evaluation completed.
Per-dataset inherited flock prevents a second active pipeline of this package.
Do not launch again merely to view progress. Existing different-version runs
are not managed by this lock and must not be launched concurrently manually.

## Status after any SSH login

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
"$PY" -B "$ROOT/tree_grid_beta_20261029/launch.py" status \
  --root "$ROOT" --dataset both --stage pipeline
```

For stage logs: --dataset tree or grid, and --stage h2/h1/glue/sample/evaluation.
The stage status is a one-time snapshot, not a monitor or second training.
For continuous view, read LOG from logs/<dataset>_tg_beta_latest_pipeline.json
and tail -F it; Ctrl+C stops viewing only. Formal stage logs are
OUTPUT/train/{h2,h1,glue}.log. Smoke logs are OUTPUT/smoke/{stage}.log and
OUTPUT/smoke_sample.log. Sampling and evaluation logs are OUTPUT/sample.log
and OUTPUT/evaluation.log.

Final markers:

- PASS: TREE_BETA_MEAN_HOG_TRAIN_SAMPLE_EVALUATE_COMPLETE
- PASS: GRID_BETA_MEAN_HOG_TRAIN_SAMPLE_EVALUATE_COMPLETE

## Metrics and original baseline

The original server's evaluate_collection_tables.py is invoked READ ONLY with
the modified manifest and output folder. Table1 uses mode=tv, component-policy
lcc; Table3 evaluation-only Louvain resolution1/seed0, same as old baselines.
Training Grid r2 is deliberately NOT the evaluation-only Louvain resolution.
No training hierarchy, saved graph or original output is rewritten.

New results: ROOT/higgs/outputs/<pipeline-name>/evaluation/:

- table1_mmd.csv; table3_summary.csv; table3_per_graph.csv; table_protocol.json
- table1_original_vs_beta_mean_hog.csv
- table3_original_vs_beta_mean_hog.csv (means, no displayed standard deviations)
- comparison_status.json

Automatic baseline lookup only accepts original saved Table1/3 results with
the same manifest/hash, samplecount, TV/LCC and evaluation Louvain protocol.
If not found, new metrics are retained and status says baseline_not_found;
the original model is NOT retrained or resampled. Comparisons refer to the
user's original-code extension experiment, NOT published HiGGs Tree/Grid
numbers. The HiGGs paper does not publish these datasets. No improvement is
promised: Table1lowerisbetter and Table3meanproximity are reported separately.

Local tests under this folder use Windows-only fixture paths and are not the
Linux installer. They are not real-data500epoch/GPU quality claims. The
server pipeline performs its own real-data smoke before any formal training.
