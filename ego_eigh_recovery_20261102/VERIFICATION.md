# Verification, 2026-11-02 Asia/Shanghai

All31 local tests passed using C:/path/to/conda/envs/higgs_env/python.exe.
Files were authored only in this new deliverable directory. No existing
production model/sampler/evaluator, weight, split, cache or sample was edited.

## Numerical adapter:16 tests

- Actual existing EigenFeatures implementation used unchanged, through private
  function-global rebinding rather than global torch monkeypatching.
- Native result type/bits and RNG unchanged on successful normal solves.
- Forced recognized LinAlg nonconvergence uses the same matrix on CPUfloat64.
- Repeated spectra, disconnected/allzero Laplacians and singleton shapes.
- NaN/Inf/asymmetry rejection with diagnostic artifacts; no input sanitization.
- Unrelated RuntimeError propagates without fallback.
- Bad CPU eigenpairs, nonfinite native outputs and nonfinite audit residuals
  are refused. CPU and cast-back residual/orthogonality checks are enforced.
- One feature instance's guard does not affect another instance or global torch.

## Actual Glue reverse-sampler integration:1 test

The existing FourBenchmarkGlue model ran a tiny FOUR-STEP CPU synthetic reverse
sampler while every native eigensolver call was forced to raise the same class
of nonconvergence. CPUfloat64 fallback completed, with finite audit outputs,
saved diagnostic matrices, unchanged model weights and preserved internal
community edges. The real Glue horizon remains400steps; this test is not a
real Ego sample, the actual failed matrix, or a server/GPU success claim.

## Recovery runner:14 tests

- Exact31 contiguous old sample hashes/progress required and preserved.
- Prefix semantic equality AND raw SHA256 equality required; mismatches stop.
- Wrong GPU, active pipeline, changed selected checkpoints and mixed inputs
  refused; sampler probe lock closes before the ordinary sampler locks itself.
- Fake/mocked full151 runpy flow tested hook order, all31prefix checks,
 151snapshot calls, no skipped index and restoration of hooks/argv after
  success or failure. This is NOT151real graphs generated locally.
- RNG snapshot writing does not advance Python/NumPy/Torch RNG state.
- Saved original evaluator's legacy schema (no explicit count) is accepted
  only through matching test IDs/samplecount/manifest/policies/kernels.
- Missing original results are not re-evaluated; kernel/reference mismatches
  are rejected. Only new comparison CSVs are written in the recovery output.

## Executed command

```
python -B -m unittest discover -s deliverables/ego_eigh_recovery_20261102 -p test_*.py -v
```

TEMP/TMP pointed into the workspace temporary directory; WANDBdisabled and
MPLBACKENDAgg were used. Torch/Lightning deprecation warnings did not cause
test failures.

## Remaining runtime checks

The actual server failure matrix is not in the pasted log and was unavailable
locally. The numerical cause is not yet proven. Server `check` verifies frozen
sources/weights, physicalGPU2identity, available memory and old31graph hashes.
The real recovery must still pass prefix replay, capture/audit any failing L,
finish151graphs and then complete the unchanged mixed evaluator.

Linux inherited flock/Popen behavior is guarded in code and fixture-tested,
but no live Linux server was operated from this Windows session. No task was
signaled or stopped. No driver/runtime package was upgraded.

Per-graph RNG snapshots are audit material; direct resume from those snapshots
is NOT implemented. Normal/recovered outputs after an eigensolver fallback need
not be bit-identical because eigenvectors in repeated eigenspaces are not
unique. The recovered result must disclose this numerical policy alongside
the earlier seed/component-policy differences from the original HiGGs arm.
