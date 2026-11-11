# Verification scope, 2026-10-29

Prepared from the original HiGGs project conversation and its checked Tree/Grid
partition/training deliverables. The two pinned manifests, raw source hashes,
Tree author split and Gridseed1234split were read directly; no new split was
inferred from dataset names. Cache counts were recomputed from frozen JSON
topology and partitions and agree with all18original stage/split counts.

Local verification completed:

- 16unit/regression tests: exact manifest identities, split mutations refused,
  resolution/max_size, cache naming, cache-only/no-rebuild, train-only node
  priors, GPU free-memory ordering and distinct assignment, no utilization
  query, comparable baseline protocol, all12dataset/stage/mode configurations,
  H1/Glue conditions, pair uniqueness, empty/edgeless inputs, complete training
  report checks and refusal of partial/changed checkpoints.
- Scheduler mock: all3formal stages start before polling; one failed stage
  does not kill siblings or produce successful training audit/sampling.
- Real preserved Glue reverse method on synthetic CPU data: float masks
  reproduce the historical TypeError; new Boolean prepared masks complete the
  reverse call, preserving intra-community edges, symmetry and parameters.
  That test alone uses4steps, not the formal400stepGlue configuration.
- All3real model classes: synthetic CPU finite loss/backward/gradients,
  one-epoch train/validation, best monitor and strict checkpoint reload. H1
  X12/E6/y14, mean16andHOGdepth4verified. These are software tests, not real
  Tree/Grid model-quality experiments.
- Final CPU tests use a workspace-local temp directory; initial runs passed
  but printed Windows WandB atexit temp-permission warnings, resolved in the
  rerun by workspace TMP/TEMP. No runtime monkeypatch is required on server.

Not executed by this assistant:

- Linux installer/cache-copy against the live server.
- Real Tree/Grid GPU smoke, formal500epochtraining, fulltestsampling, metrics.
- Live GPU memory inspection. Selection occurs on server at command submission.

The bundle changes only additive dataset/entry files when installed. Protected
shared sources are hash-checked and never rewritten. The pre-existing Windows
production checkout, original server project, trained checkpoints and saved
results were not edited. Existing packages were read as templates only.

The original server evaluator will be reused read-only with TV/LCC and
evaluation-only Louvainr1/seed0. Automatic comparisons require same pinned
manifest, sample/test counts and metric protocol; missing old saved CSVs do
not trigger new original-model training/sampling. Results will be one split
and one seed, not evidence of universal improvement or separate causal
contributions of Beta, two-hop or HOG.
