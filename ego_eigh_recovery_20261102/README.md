# Ego Glue eigensolver recovery

This is an additive Linux recovery entry for ONE failed Ego sampling run.
It does not install over any model, diffusion, dataset, sampler or evaluator.
It does not retrain, change thresholds, change the mixed evaluation protocol,
upgrade PyTorch/CUDA/drivers, stop jobs or discard generated graphs.

Target root: `/path/to/HiGGs-GBD/HiGGs-main`

Failed pipeline:
`outputs/four_benchmarks_mean_hog_20261101_233541_555705_ego_gpu2`

The supplied log shows completed graph indices0..30, then a Glue structural
feature eigendecomposition failure while sampling graph32/151. It does not
show completed Ego sampling or metrics. Three earlier datasets are unaffected.

## Numerical policy

Only the recovery process's Glue `ExtraFeatures.eigenfeatures` is wrapped.
The existing EigenFeatures Python function body and feature postprocessing
are reused, with a private solver lookup; global torch functions and installed
source files are not patched.

- Normal native solves remain native and do not consume new random numbers.
- Matrices must be finite, real and exactly symmetric. Invalid matrices are
  captured and rejected, not sanitized or perturbed.
- Only a recognized eigensolver nonconvergence can fall back to CPU float64
  `torch.linalg.eigh` on the SAME matrix. No jitter, graph edits, eigenvalue
  clipping, feature disabling or substitute sample is used.
- CPU eigenpairs and their cast-back values are audited for finiteness,
  residual and orthogonality. A failed audit stops the run.
- Eigenvector signs and degenerate eigenspaces are not unique. CPU fallback
  can change downstream neural features and samples after the fallback. This
  is a recorded numerical-compatibility change, not a new quality improvement
  or a promise of globally bit-identical output.

Reference: https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html
The documentation warns about nonunique eigenvectors and CUDA numerical
stability. The old log alone does not prove which runtime/matrix condition
caused this failure. The recovery records actual runtime versions and matrices.

## Why replay the first31 graphs

The old sampler did not checkpoint Python/NumPy/Torch CPU/CUDA RNG states
between graphs. Starting at graph32 with a new seed would silently change the
random sequence. Recovery instead starts the SAME151-graph sequence from
seed1234 in a NEW directory and verifies the first31 against the original saved
files before continuing. A prefix mismatch stops recovery; it is not accepted
as a substitute experiment. Old samples and their progress/audits stay intact.

GPU2 is retained for replay and at least8GiB free memory is required. There is
no GPU-utilization restriction and no job is killed. If the old coordinator or
sampler still holds a lock, launch refuses: do not remove lock files or stop
unrelated processes. Retry after the failed task has exited.

New RNG snapshots are retained after successful graphs for provenance/future
recovery; this entry does NOT claim an implemented direct-resume command from
those snapshots. Full replay is the currently supported recovery procedure.

## Upload and execute once

Upload `ego_eigh_recovery_20261102.zip` to the target project root, then:

```bash
(
set -e
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
cd "$ROOT"
unzip -n ego_eigh_recovery_20261102.zip
B="$ROOT/ego_eigh_recovery_20261102"
"$PY" -B "$B/run_recovery.py" check --root "$ROOT" \
  --failed-run "$ROOT/outputs/four_benchmarks_mean_hog_20261101_233541_555705_ego_gpu2" --gpu 2
"$PY" -B "$B/run_recovery.py" launch --root "$ROOT" \
  --failed-run "$ROOT/outputs/four_benchmarks_mean_hog_20261101_233541_555705_ego_gpu2" --gpu 2
)
```

No `conda activate` is needed with the absolute interpreter. The `check` action
is preflight only; launch prints PID/LOG/OUTPUT and detaches from SSH.
This is not the old pipeline command and does not train any model.

After151 graphs complete, the unchanged existing evaluator computes frozen
mixed Table1 and mean Table3, and the recovery entry compares completed matching
original Ego result CSVs read-only. Missing original results are reported, not
automatically regenerated. The old failed pipeline's status/pointer is NOT
rewritten. Follow this recovery entry's own pointer instead.

## Reconnect-safe viewing

Snapshot:

```bash
/path/to/conda/envs/higgs/bin/python -B \
  /path/to/HiGGs-GBD/HiGGs-main/ego_eigh_recovery_20261102/run_recovery.py \
  status --root /path/to/HiGGs-GBD/HiGGs-main
```

Continuous recovery log:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
INFO="$ROOT/logs/ego_eigh_recovery_latest.json"
if [ -f "$INFO" ]; then
  LOG="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["log"])' "$INFO")"
  printf 'LOG=%s\n' "$LOG"
  tail -n 60 -F "$LOG"
else
  echo "尚未找到Ego数值恢复任务的启动记录"
fi
```

Ctrl+C stops the viewer only. Submission or a prefix match is not full sampling
completion; metrics and comparison require the recovery's final completion
markers. No quality improvement is promised.

## Comparison scope

The existing original arm uses sampling seed0 and H1-LCC, while this new arm
keeps its previously configured seed1234 and full H1 communities. The final
mixed/LCC evaluation remains the same. Report whole-pipeline single-run
comparisons, not a strict architecture-only ablation. Record this additional
numerical policy with the recovered Ego row; do not erase the failed attempt.

See VERIFICATION.md for locally executed checks and remaining server checks.
