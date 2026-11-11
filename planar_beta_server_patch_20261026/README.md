# Planar Beta / mean two-hop / HOG server integration

This offline installer targets the modified checkout only:
 /path/to/HiGGs-GBD/HiGGs-main

The original checkout and its manifest/caches are never changed.

## Changes

- Registers Planar in the existing hierarchy training entry.
- Adds a Planar-only DistributionNodes subclass with floating, range-safe log_prob.
  The shared distribution class, its sampler and existing datasets stay unchanged.
- Allows Planar-only checkpoint cadence override for one-epoch best/last saving.
- Adds a three-stage real-cache smoke driver. H2, H1, Glue run sequentially
  on one GPU selected from physical GPU2/3. GPU0/1 are not selected.
- Does not edit Beta transitions, HOG/two-hop model sources, OriginalGlue model
  source, or full-chain samplers.

## Install

From the modified checkout:
 PY=/path/to/conda/envs/higgs/bin/python
 env CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 \
   "$PY" -B planar_beta_server_patch_20261026/install_planar_beta.py "$PWD"

The installer requires the exact audited main/model hashes and backs up main.py.
It refuses reinstallation or unrecognized changes rather than overwriting.

## Smoke

 bash planar_beta_server_patch_20261026/run_smoke.sh "$PWD"

Each stage uses one epoch, 2 train batches, 2 validation batches, batch size4,
seed0. No test evaluation, image generation, full-chain sampling, formal
500-epoch training or data rebuilding is run. All 9 copied caches are mandatory.

The driver checks finite train/val losses and gradients; validation-best and last
checkpoints; strict checkpoint reload; frozen manifest identity; H1 mean/HOG
metadata and input widths; recorded input/source SHA256 after all stages.
The finite-audit callback is used only by the smoke driver.

Completion marker:
 PASS: PLANAR_BETA_MEAN_HOG_THREE_STAGE_SMOKE

Server execution and GPU results have not been produced by the local package build.

