# Verification

Date: 2026-10-30

Executed locally:

```text
python -B -m unittest discover -s deliverables/dd_earlystop_20261030 -p test_*.py -q
Ran 25 tests
OK
```

The tests cover:

- exact worker-identity checks and PID reuse refusal;
- snapshot before termination, verified exit, and timeout refusal;
- resume on invalid checkpoints, preflight failure, or operator interruption;
- no signaling of already-paused or wrong-stage workers;
- no stop when no GPU has enough free VRAM;
- highest-free-memory GPU choice without a utilization condition;
- saved-last/best metadata distinction (synthetic H1 last 329 / best 109);
- strict reload requests for all six copied checkpoints;
- unchanged original synthetic checkpoint hashes;
- no forged H1 formal-500 completion report;
- manifest, source, snapshot, and completed-stage provenance gates;
- sampler preparation before H1 termination;
- model-independent Boolean Glue mask validation;
- Python source syntax.

Process signaling, model loading, and checkpoint contents in these unit tests
are mocked or synthetic. No real Linux server process was signaled during local
verification. No real DD training, full-chain sampling, or quality evaluation
was run locally. The actual server launch performs its own input checks and
strict CPU checkpoint reloads before it sends the termination signal.
