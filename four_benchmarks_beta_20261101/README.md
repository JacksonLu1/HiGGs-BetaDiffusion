# Four frozen benchmarks: Beta-H2 / Beta-H1 + mean two-hop + HOG

This is a Linux upload package. It does not install into the Windows production
checkout and never modifies the original HiGGs experiment.

- Target: `/path/to/HiGGs-GBD/HiGGs-main`
- Read-only original: `/path/to/HiGGs-main1/HiGGs-main`
- Python: `/path/to/conda/envs/higgs/bin/python`

## Frozen comparison settings

| Dataset | Whole graphs | Train / val / test | Epochs per H2 / H1 / Glue | Formal samples |
|---|---:|---|---|---:|
| Ego-small (`ego_small`) | 200 | 128 / 32 / 40 | 1000 / 1000 / 1000 | 40 |
| PROTEINS (`proteins`) | 1113 | 712 / 178 / 223 | 500 / 500 / 500 | 223 |
| MSRC_21 (`msrc21`) | 563 | 360 / 90 / 113 | 500 / 500 / 500 | 113 |
| Ego (`ego`) | 757 | 485 / 121 / 151 | 500 / 500 / 500 | 151 |

These are exactly the four new collections in the original-code conversation,
not replacements for DD, ENZYMES, Tree or Grid. They are topology-only: original
TU attributes/labels remain in source metadata, not model generation targets.
PROTEINS uses the full 1113-graph collection; MSRC_21 is not MSRC_21C.
Ego sources may overlap in their original source-network vertices. The original
frozen Ego source explicitly removes self-loops; this package does not alter it.

The installer copies the original SERVER's source JSONs, dataset configs,
manifests and all 36 processed caches, checking exact bytes and every cached
graph's topology, source graph ID, condition and Boolean Glue membership mask.
No new download, Louvain execution, split, graph filtering, merging, singleton
removal or cache reconstruction is allowed. Missing original caches are errors.
The original training jobs may keep reading their files; this installer sends
no signals and does not write anything into that original checkout.

Initial hierarchy: resolution1, whole-graph split seed1234, Louvain seed0,
weight=None. Exact original manifest SHA256s:

```
ego_small 374529d3bb5bbc013a163bad736d0f12a3418297eeb3e6101d9ba3981234f361
proteins  b48aa81179c3d1cb1c9616df1f343ad995f0659a0f20e98ee3bb0edf786f158c
msrc21    d9c3c8cd6f802492cf71bc0d94c4f3f4d43af9414da8624f9ed7a6eb2e09b2a3
ego       bd5a14072b1cf68f9e6446f0e7e79ef85a103fbfe106bf1b36233bd277027e80
```

## What the pipeline runs

For each dataset independently:

1. Real-cache, two-batch, one-epoch H2/H1/Glue smoke tests, finite losses and
   gradients, strict validation-best and last checkpoint reloads.
2. One full-chain graph from those smoke checkpoints, not evaluated as a formal
   result. An edgeless sampled H2 is retained; it is not repeatedly sampled to
   force a connected template.
3. Fresh formal H2/H1/Glue models, no warm start or resume. All three stages
   train concurrently on the dataset's selected GPU. Every10epochs validation.
4. Validation-best checkpoints are selected separately for each stage, not by
   test MMD and not necessarily the last epoch.
5. Full-chain generation of the exact test-set graph count.
6. CPU-only mixed Table1 and mean Table3 evaluation in a new output directory.

Train batch32, seed0, lr0.0002, AdamW/amsgrad, weight_decay1e-12, EMA0,
workers0, matching the original four-collection training entry. The split seed
1234 is different from training seed0. Sampling seed1234 is set once for the
whole series; H1 batch4 and Glue batch1. Sampling does not force real test sizes.

- H2: existing Beta diffusion, 100steps.
- H1: existing Beta diffusion, 100steps, gated mean two-hop16channels retained,
  HOG-Diff RW/SPD feature transfer depth4 retained; widths X12/E6/y14.
- Glue: original discrete diffusion, 400steps.
- No Beta transition, objective, threshold0.5, existing model or shared sampler
  source is changed. New additive entry points route only these four datasets.
- Monitors: val/beta_h2_loss, val/h1_total, val/epoch_NLL respectively. These
  are different training objectives from the original discrete NLL monitor.

Singleton batches need explicit compatibility: pad the neural forward with one
masked position and crop it afterward; no graph node, parameter or saved edge
is added. Empty-edge CE is zero only in this dataset-scoped Glue subclass.
Normal-width forwards and all diffusion/sampling methods are inherited.

Both full H2 and full H1 are retained during assembly; final evaluation uses
independent LCC of each real and generated graph, as in the original arm.
Glue uses Boolean membership masks and actual maximum pair padding. Original
Glue's existing minimum-connection fallback for an H2-connected community pair
is inherited; this package introduces no additional graph repair or rejection.

## GPU scheduling

At most two datasets run concurrently by default; remaining datasets queue.
Each newly started dataset gets the GPU with the most currently free VRAM,
excluding GPUs assigned to another dataset in THIS queue. Existing unrelated
jobs can share the GPU. Default minimum free memory16GiB, minimum allowed8GiB.
There is no utilization or compute-process exclusion check. Memory is not a
reservation; other users can allocate memory later. On failure no other job is
killed, and other already-started stages are allowed to finish.

Optional `--gpus 0 3` limits the candidate GPUs. `--max-parallel 1` runs one
dataset at a time; up to4 is accepted. The launcher and child workers inherit
per-dataset locks, preventing duplicate active pipelines of this package.

## Upload, install, and start ONCE

Upload `four_benchmarks_beta_20261101.zip` into the TARGET root, then:

```bash
(
set -e
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
cd "$ROOT"
unzip -n four_benchmarks_beta_20261101.zip
B="$ROOT/four_benchmarks_beta_20261101"
"$PY" -B "$B/install.py" --root "$ROOT"
"$PY" -B "$B/install.py" --root "$ROOT" --apply
"$PY" -u -B "$B/launch.py" pipeline --root "$ROOT" --dataset all --max-parallel 2
)
```

The first install command is read-only. Apply writes additive files and an
audit inventory under interface_patch_backups; existing non-identical files
are refused, not overwritten. A repeated identical install is safe. An error
rolls back only files that invocation created. An absolute Python path means
the shell's active conda environment is irrelevant.

Submission prints PID/LOG; this means queued/submitted, NOT training complete.
Do not run pipeline again just to view progress. This launcher trains and
samples only the new version; original-model jobs and results are untouched.

## Reconnect-safe progress

Snapshot of all four:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
"$PY" -B "$ROOT/four_benchmarks_beta_20261101/launch.py" status --root "$ROOT" --dataset all
```

Continuous queue log, after reconnecting:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
INFO="$ROOT/logs/four_benchmarks_beta_latest.json"
if [ -f "$INFO" ]; then
  LOG="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["log"])' "$INFO")"
  echo "LOG=$LOG"
  tail -n 60 -F "$LOG"
else
  echo "尚未找到启动记录"
fi
```

One dataset/stage per SSH tab; change DATASET and STAGE only:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
DATASET=ego_small
STAGE=h1
INFO="$ROOT/logs/${DATASET}_qb_beta_latest_pipeline.json"
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
      echo "该阶段尚未开始，可能还在进行短测试；不要重复启动"
    fi
  else
    echo "该数据集正在排队"
  fi
else
  echo "尚未找到启动记录"
fi
```

STAGE can be h2/h1/glue/sample/evaluation/smoke_sample. Training stage logs
appear only after its smoke checks finish. Ctrl+C exits the viewer only.

## Frozen mixed evaluation

| Metric | Kernel |
|---|---|
| Nodes | Gaussian-TV |
| Degree | Gaussian-EMD |
| Clustering | Gaussian-EMD,100bins,sigma0.1,scale100 |
| Eccentricity | Gaussian on legacy50bin all-pairs shortest-path histograms |
| Spectre | Gaussian-EMD,200bins,all eigenvalues/all requested graphs |

The existing original-checkout metric helper functions are imported read-only
in a separate CPU process, so both arms use the same implementation. Their
paths/hashes and descriptor settings are saved in protocol.json. This is the
frozen author-postprocessing mixed branch, not a claim that every unpublished
Table1 execution detail has been recovered. No all-TV/all-EMD substitution.
Evaluation Louvain resolution1/seed0/weightNone is independent of the frozen
training communities. Edgeless graphs have one community per node, explicitly,
and are not discarded. Table3 prints means only; CSV retains std for auditing.

Per-dataset outputs:

- Training: ROOT/outputs/four_benchmarks_mean_hog_<time>_<dataset>_gpu<id>/
  with smoke/,train/{h2,h1,glue}/formal_result.json and separate logs.
- Samples: ROOT/higgs/outputs/<same-name>/sample/sampling/sampled_graph_*.pkl
- Metrics: ROOT/higgs/outputs/<same-name>/evaluation/
  table1_mixed.csv, table3_summary.csv, table3_per_graph.csv, protocol.json.

Completion per dataset:
`PASS: <DATASET>_BETA_MEAN_HOG_TRAIN_SAMPLE_EVALUATE_COMPLETE`.
Queue completion: `PASS: FOUR_BENCHMARKS_BETA_MEAN_HOG_PIPELINES_COMPLETE`.
The actual numeric quality verdict comes from the saved results, not a PASS.
The HiGGs paper does not publish these four extension datasets: comparison
rows are original-code reproduction vs modified-code experiment, not invented
published paper rows.

## Local verification limitations

See VERIFICATION.md. Local CPU/synthetic and copied-cache checks do not claim
remote GPU success, formal 500/1000epoch completion, or improved quality.
The production GPU smoke is part of the submitted pipeline before training.
