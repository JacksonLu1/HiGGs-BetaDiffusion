# Server-only HOG-Diff RW/SPD feature transfer

## Frozen baseline

- Repository: JacksonLu1/HiGGs-BetaDiffusion
- Backup branch: `backup/server-twohop-20261021`
- Baseline commit: `59d7dacb02e021d9b8da8ac4805fa2bf405e5a20`
- Expected unmodified `dgd/diffusion_model_beta_h1.py` SHA256:
  `4d64a3cb591ffd530f13c92ac3d7a8d19aa3caa5a53ff00d19f19470686ec367`

This branch distributes a patch, not already-modified production model sources.
The Windows production H1/H2/Glue implementations have not been edited for this
experiment. Apply the patch explicitly in the Linux project only after preserving
the server H1 source. Do not use `git pull` or checkout this branch over server
datasets/checkpoints: this is a source backup, not a dataset or weights backup.

## Scope and scientific status

GBD preconditioning and degree modulation are already-published GBD components.
HOG-Diff RW/SPD encoding is also a published component. None is, by itself, a new
algorithmic contribution by this project. The new experiment is a transfer of a
specific existing input feature component into conditional HiGGs Beta-H1.
Effectiveness and any contribution claim require independent experimental evidence.

The patch installs only:

1. `dgd/diffusion_model_beta_h1.py`: opt-in input dimensions, concatenation and
   checkpoint metadata validation;
2. `dgd/diffusion/hog_rw_features.py`: attributed upstream computation and masks;
3. `configs/model/beta_h1_hog_rw.yaml`: new single-factor configuration;
4. `tests/test_h1_hog_rw.py`: feature, compatibility and synthetic smoke tests;
5. `THIRD_PARTY_NOTICES_HOG_RW.md`: attribution and noncommercial license notice.

Only `get_rw_feat` RW and SPD channels are transferred. This is not the complete
HOG-Diff ScoreNet, degree input, CCF, spectral diffusion or diffusion bridges.
The original RW powers and zero-count SPD encoding are retained; it is not an
exact shortest-path implementation. See the notice for the CC BY-NC 4.0 terms.

The default old model configurations remain disabled, and legacy checkpoints
retain their original input dimensions/state dict. New RW/SPD configuration uses
depth=4, disables two-hop, leaves preconditioning off and eta fixed, and needs new
training because the input layers change. No warm-start, permissive state loading,
data/cache regeneration, H2/Glue changes or metric edits are included.

## Installation mechanism

Use `git apply --check --whitespace=nowarn <patch>` first, then the same
command without `--check`. Standard hunk counts are provided and validated.
The patch neither deletes files nor changes datasets or checkpoints.
New-file collisions and context mismatches fail the check. Verify the base H1 SHA
before applying. Make a separate backup of the old H1 file for rollback/testing.

## New-component validation on Linux

Use the server's existing higgs interpreter, not the Windows Python environment:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
H1_HOG_BASELINE_FILE="/absolute/path/to/pre-patch/diffusion_model_beta_h1.py" \
/data2/wenb/wbai/miniconda3/envs/higgs/bin/python -B \
  tests/test_h1_hog_rw.py --require-integration
```

This runs synthetic training only. It does not rerun old real-data experiments,
start formal H1 training, sample H2/Glue, or evaluate Table1/Table3.

Local validation at preparation time: standard patch check/apply in an isolated
copy passed; three changed/new Python files compiled in memory; all five pure
feature tests passed. Full Lightning/PyG model tests were not run locally because
the available local Torch interpreter has no pytorch_lightning. The Linux command
requires integration tests and fails instead of silently accepting skipped tests.

After these new-component checks pass, the intended next step is a separate
from-scratch H1 smoke/formal launch using `model=beta_h1_hog_rw` and the existing
dataset caches. No formal training command is executed by this patch.
