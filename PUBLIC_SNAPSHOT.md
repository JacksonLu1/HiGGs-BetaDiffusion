# Public server source snapshot

This branch archives a sanitized source snapshot of the HiGGs-GBD server
version captured on 2026-11-11, before the H1 k-core experiment. It includes
Beta H2, Beta H1 with mean two-hop and HOG RW/SPD features, original Glue,
and the associated training, sampling and mixed evaluation code.

Server and local workstation paths were replaced with example paths. The
original archive remains local. Raw datasets, checkpoints, logs, generated
graphs, and server audit outputs are not included. Do not deploy or run the
historical launch or cleanup scripts without reviewing their paths and inputs.
Dataset, checkpoint and deployed-source hash guards were preserved. Internal
bundle payload digests were refreshed for these sanitized files; this does not
revalidate deployment. The empty ORCA stubs are retained as captured.

See THIRD_PARTY_NOTICES_HOG_RW.md for attribution and the separate HOG-Diff
license. See PUBLIC_SNAPSHOT.json for a per-file integrity manifest.
