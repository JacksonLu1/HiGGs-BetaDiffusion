# Verification and limitations — 2026-11-03 (Asia/Shanghai)

## Local checks

Python: `C:/path/to/conda/envs/higgs_env/python.exe`, Windows CPU. All writes were
confined to this new deliverable and temporary test fixtures. Existing original
and modified production checkout files were read, not installed over or edited.

Final combined run: **42 tests passed in129.235seconds; exitcode0**.
The combined suite also verifies test-fixture module path isolation; local
fixtures do not change installed production package paths.

42 unittest cases:

- 14 installer/data-contract tests: every original source JSON, manifest and
  dataset YAML byte hash; all27 expected hierarchy-cache counts derived from
  frozen records/partitions; disjoint whole-graph IDs; singleton retention;
  topology/source-ID/Boolean-mask verification; cache-only missing-cache
  refusal; finite node prior; upload tamper, path escape and conflicting
  existing target refusal.
- 7 orchestration tests: memory-only GPU ranking (high utilization remains
  eligible), three-dataset default, duplicate active lock refusal before
  spawning, matched COLLAB32/16/8 reports, smoke/train/sample/CPU-eval order,
  and no sampling after failed training. Process launches are mocked here.
- 10 evaluation tests: all five metric-dispatch calls AST-match the original
  `evaluate_small_benchmarks.py`; no all-TV or all-EMD substitution; complete
  canonical sample indices; all graphs/eigenvalues for Spectre; finite MMD;
  no silent empty-graph filtering; fixed evaluation Louvain and descriptor
  metadata. Dispatch tests use mocks, not claimed experimental MMD values.
- 11 payload tests: all18 dataset/stage/mode config combinations; frozen
  identities and counts; seed0 once and BOTH upstream-LCC with exact NumPy
  component tie semantics; finite actual model backward and strict reload;
  singleton H2/H1 generation without adding graph vertices; unchanged
  non-singleton neural forward/parameter state; empty edge CE; actual tiny
  Glue reverse generation with Boolean mask preserving intra-community edges
  and weights; inherited diffusion methods.

The payload suite also runs9 tiny CPU fits (3datasets x3stages), using actual
original frozen source records and their original partition/node IDs. It
constructs separate QA-only subset caches, performs one train and one validation
batch with formal100/100/400 horizons, and strictly reloads the validation-best
checkpoint with an identical state_dict. These subset caches are not shipped,
not the actual27 server caches, and not formal full-dataset experiments.
Synthetic reverse-sampling tests use a shorter4step horizon for runtime only.

Command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP='C:/path/to/HiGGs-GBD/tmp'; $env:TMP=$env:TEMP
$env:MPLBACKEND='Agg'; $env:WANDB_MODE='disabled'
& 'C:/path/to/conda/envs/higgs_env/python.exe' -B -m unittest discover `
  -s deliverables/tu_benchmarks_beta_20261103 -p 'test_*.py' -v
```

## Source-based protocol review

The original chat was read live. Its TU plan, frozen manifests, model subclass,
training/sampling runner, sampling method bodies and mixed evaluator were
inspected read-only. A separate review confirmed exact settings and namespace
isolation. In particular, the original H2 method itself takes LCC even though
the runner only explicitly labels H1's inherited LCC policy. This new package
therefore sets BOTH policies to upstream_lcc, rather than repeating the earlier
four-collection package's seed1234/full-H1 setting.

## Server checks still required

No connection to the server was used; no GPU job or formal training/sample was
started locally. The full27 original prepared caches are unavailable locally.
The Linux installer must validate each original cache byte identity/topology
before copying, then GPU smoke tests must succeed. Actual peak GPU memory,
complete1000/500epoch training, complete200/1000/822graph sampling, numerical
stability and final model quality remain unverified until the server runs.

There is no guarantee of improvement, convergence, runtime or globally bitwise
identical original/new samples. Model families, losses and diffusion horizons
differ by the requested intervention. Data, budgets, seeds, component policies
and final mixed evaluator settings are explicitly matched and audited.

No comparison to unpublished paper rows is fabricated. The original TU arm may
still be training/evaluating; keep its eventual original-code measurements
separate from the HiGGs paper's published dataset results.
