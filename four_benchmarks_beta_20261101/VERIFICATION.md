# Verification, 2026-11-01 Asia/Shanghai

## Scope

All authored files are under the new deliverables/four_benchmarks_beta_20261101
directory. No existing model/diffusion/sampler source in the production Windows
checkout was changed. No remote installation, GPU job, formal training or test
generation quality is claimed. Original-server input files remain read-only.

## Executed checks

1. Exact frozen source/config/manifest hashes for all four collections were
   compared with the original-code bundle. Complete-graph split counts and all
   nine hierarchy cache counts per dataset match that original evidence.
2. All 36 ACTUAL cached datasets from the original isolated preparation copy
   were read, and every sample's node/edge topology, source ID, stage condition,
   community ID and Boolean padded Glue mask was checked against the frozen
   manifest. Before/after cache SHA256s were unchanged. Singleton graphs remain.
3. All41 unit/integration tests passed: configuration composition across all24 combinations,
   parameter/state-dict equality for ordinary-width forwards, singleton H2/H1
   finite backward/sampling, strict checkpoint reload, inherited diffusion
   methods, actual Boolean-mask Glue reverse sampling and internal-edge
   preservation, invalid cache/file/hash refusal, mixed dispatch/all graphs,
   empty-graph handling, pipeline phase order and memory-only GPU ranking.
4. Actual-data CPU smoke: each of four datasets x H2/H1/Glue trained two batches
   and validated one or two batches for one epoch, using the copied original
   caches and FULL configured100/100/400 diffusion horizons. All12 tests saved
   a finite validation-best checkpoint and strictly reloaded it. Training
   gradients and reloaded parameters were finite. This is not500/1000epochs.

Actual-cache CPU validation scores (software readiness, NOT generation quality):

| Dataset | H2 | H1 | Glue |
|---|---:|---:|---:|
| ego_small | 6.3623351737 | 10.9765148481 | 13.3638868332 |
| proteins | 9.0726890306 | 5.8557481453 | 127.0408096313 |
| msrc21 | 12.7309837927 | 5.2314206674 | 150.2949676514 |
| ego | 12.3571224099 | 2.3042630959 | 438.8428955078 |

## Local execution

Interpreter: C:/path/to/conda/envs/higgs_env/python.exe; CPU-only tests use temporary
copies under this new bundle and workspace tmp. WANDB is disabled. The data
source was C:/path/to/HiGGs-main1/HiGGs-main/tmp/small_benchmark_stage_20261101.

```
python -B -m unittest discover -s deliverables/four_benchmarks_beta_20261101 -p test_*.py -v
python -B verify_local_real_cache_smoke.py
```

The second script is LOCAL QA only and contains fixture paths, not a Linux
server launch command. See README for the actual server commands.

## Remaining production verification

The Linux installer rechecks exact SERVER source/cache bytes. The production
pipeline repeats real-data GPU smoke plus one full-chain graph before fresh
formal training. No synthetic-only check is treated as remote GPU success.
Linux flock/Popen queue handoff cannot be live-tested from the Windows-only
local environment here; it uses inherited per-dataset locks and emits separate
process/log/output records. Existing jobs are never signaled or stopped.

Metrics call the same original mixed helpers read-only. The ordinary mixed
dispatch is regression-tested; no large real held-out MMD calculation was run
while building this package. Final numerical comparison requires the server's
completed formal samples, not the smoke validation losses above.
