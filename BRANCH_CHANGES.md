# Branch Changes

Branch: `codex/paper-data-config-fixes`

This branch aligns the hierarchy dataset configuration and preprocessing behavior
with the confirmed HiGGs paper settings while retaining parameters that could not
be confirmed from the paper or repository evidence.

## Configuration changes

- Set the SBM Louvain resolution to `2`.
- Set the Facebook Louvain resolution to `10`.
- Set the SBM sampling count to `40`.
- Made the SBM training script pass `dataset.resolution=2` explicitly for h2, h1,
  and glue training.

## Dataset and cache changes

- Added resolution, regime, maximum-size, and batch-size tags to processed cache
  names where applicable, preventing incompatible caches from being silently
  reused.
- Passed the configured resolution consistently to train, validation, and test
  dataset constructors.
- Applied the generated train, validation, and test community IDs when building
  hierarchy datasets, preventing the same partitions from appearing in every
  split.
- Changed generated partition cache names from misleading `.csv` names to `.pkl`.
- Preserved downloaded Facebook edge JSON data and stored the processed graph in
  a separate pickle file.
- Updated the SBM split to 128 training, 32 validation, and 40 test graphs for a
  200-graph dataset.

## Not included

- No checkpoint, model weight, raw dataset, processed dataset, output, or IDE file
  is included in these commits.
- No network architecture, diffusion equation, loss, sampling algorithm, or
  unconfirmed experiment parameter was changed.
- This branch has not been merged into `main`.

## Data cleanup and metric correction

- Removed the legacy `data/cora/` cache tree; Cora now uses `data/cora_r2/`.
- Versioned the external sources and the canonical regenerated Cora, Facebook,
  and SBM raw, split, and processed data. Large binaries use Git LFS.
- Removed author-generated processed tensors, old partition serializations, and
  untagged split files so current code regenerates compatible caches.
- Added `data/.gitignore`, `.gitattributes`, and `data/DATA_LAYOUT.md` to
  document the versioned hierarchy data and Git LFS storage.
- Corrected test metric routing so test batches update test NLL, KL, and
  reconstruction log-probability metrics instead of validation metrics.
- Made preprocessing visualization imports lazy and disabled preprocessing
  plots when W&B is disabled, avoiding unused GUI/layout work during cache-only
  generation.
- Normalized Facebook and SBM class indices to `torch.long` before one-hot
  encoding so preprocessing works consistently on Windows.
- Accepted the configured SBM worker count and skipped invalid empty h1 graphs
  produced by edge-list conversion of communities with no internal edges.
- Declared only the three glue cache files that preprocessing actually writes,
  preventing h1.5 datasets from being regenerated on every startup.
- Retained `higgs/notepad_code/` and `figures/` because they are documentation
  artifacts and are not used by training or sampling.
