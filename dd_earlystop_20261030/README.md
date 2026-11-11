# DD early-stop snapshot and full-chain sampling

Only for the modified Linux checkout:

`/path/to/HiGGs-GBD/HiGGs-main`

This package is a separate launcher. It does not patch model sources, the old
training launcher, dataset caches, split manifests, or existing reports.

## Purpose and experiment meaning

The existing DD experiment is
`dd_beta_mean_hog_r2_e500_scratch_seed0_20261027_205633_003751`.
H2 and Glue have completed 500 epochs. H1 is still running. The user's latest
inspection showed H1 saved-last epoch 329 and validation-best epoch 109.

`launch --stop-h1` authorizes stopping ONLY that experiment's verified H1 worker.
The launcher freezes the current saved-last and validation-best checkpoints for
all three stages, reloads all six strictly on CPU, and validates the sampler
before terminating H1. If a newer checkpoint was saved before launch, its actual
metadata is recorded instead of forcing the older epoch-109 selection.

The selected best epoch is not the training budget. A saved epoch-329 checkpoint
proves 330 saved training epochs, not the exact live progress at termination.
The process's live command/identity and the saved checkpoint progress are
recorded separately. No H1 epoch-499 completion report is invented.

The new result may be compared exploratorily with the user's original DD
500/500/about-250 experiment. It is NOT an equal-training-budget ablation.

## Server launch

Upload `dd_earlystop_20261030.zip` into the modified checkout. Check its SHA256
against the value in the chat before extracting. Then:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
cd "$ROOT"
unzip -n dd_earlystop_20261030.zip
"$PY" -B "$ROOT/dd_earlystop_20261030/run_early.py" launch \
    --root "$ROOT" --stop-h1 --count 184
```

Launch only once. A successful submission prints a PID, log, snapshot, and sample
output, but it does not yet mean that H1 was stopped or sampling completed.
The worker is detached from the SSH session and holds a DD sampling lock.

The full absolute Python path is intentional; it works even if the terminal
prompt displays a different conda environment.

## Check progress

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
"$PY" -B "$ROOT/dd_earlystop_20261030/run_early.py" status --root "$ROOT"
```

Continuous log, read-only:

```bash
INFO="$ROOT/logs/dd_earlystop_latest.json"
LOG="$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["log"])' "$INFO")"
tail -n 60 -F "$LOG"
```

`Ctrl+C` exits the log viewer, not the detached worker.

Expected milestones:

1. `PASS: H1_ONLY_STOPPED_AFTER_VERIFIED_SNAPSHOT`
2. `START DD GRAPH ...`
3. `PASS: DD_EARLYSTOP_MEAN_HOG_FULLCHAIN_184_COMPLETE`
4. `PASS: DD_EARLYSTOP_AND_SAMPLING_COMPLETE`

Only the final two indicate full sampling completion. No quality evaluation is
performed by this package.

## Safety and failure behavior

- Matches H1 by UID, exact worker arguments, process start time, and a pinned
  Linux PIDFD. Native Python PIDFD or the Linux x86_64 syscall fallback is used.
- Briefly pauses that H1 only while freezing checkpoint files. A failed freeze or
  sampler metadata preflight resumes it without terminating it.
- A verified stop uses SIGTERM and SIGCONT, not SIGKILL, process-group kills, or
  broad `pkill`. If exit cannot be confirmed within 60 seconds, sampling refuses
  to start and the uncertain state is recorded.
- Existing H2, Glue, original-code jobs, and unrelated GPU jobs are not signaled.
- Chooses the GPU with the most free VRAM, with an 8-GiB minimum and NO GPU
  utilization gate. GPU memory availability can still change after selection.
- Uses immutable copied checkpoints and a new output directory. Source and input
  SHA256 gates refuse an unknown server version rather than silently adapting it.
- If the old training coordinator reports H1 incomplete after the requested early
  termination, that is expected. Follow the NEW early-stop log, not the old
  500-epoch job's completion marker.
- If H1 has already formally completed, it must have a verified epoch-499 report;
  this package can then snapshot and sample without signaling it.
- If an error appears, keep its log and snapshot and ask for diagnosis. Do not
  delete locks, bypass hash checks, fabricate completion reports, or repeatedly
  launch. Failed snapshots remain for audit.

## Sampling protocol

- Same frozen DD manifest SHA256:
  `3162886166b33c6fd7832996664b0c6a422824c4f7c82a6d44efc4cc9e7745a2`
- Complete graphs: train 587 / validation 147 / test 184; no new partitioning.
- Beta H2; Beta H1 + two-hop mean (16 channels) + HOG RW/SPD depth 4; original
  discrete Glue. No new model, threshold, or transition changes.
- Current validation-best checkpoint per stage, strict loading.
- Sampling seed 1234, H1 batch size 4, Glue batch size 1, one uninterrupted RNG
  stream, 184 full-chain graphs. No re-sampling of the old experiment.
- Upstream largest-component policy for H2 and H1, as in the original DD sampler.
  Both complete output and its structure audit are saved. Evaluation full/LCC
  handling must later match the original DD evaluation metadata.
- A binary Glue membership mask is restored to Boolean dtype in memory if
  necessary; values, partitions, weights, and source files are not changed.
- No retries to reject poor graphs, no test-edge targets, no graph repair.

## Files written on the server

New snapshot: `ROOT/outputs/dd_earlystop_snapshot_<timestamp>/`

- `request.json`, `selection.json`, `stop_audit.json`, `status.json`
- `h2/`, `h1/`, `glue/`: frozen best and saved-last copies
- `sample.pid` after the sampling child starts

New sample run: `ROOT/higgs/outputs/dd_mean_hog_earlystop_seed1234_n184_<timestamp>/`

- `sampling/sampled_graph_0.pkl` through `sampled_graph_183.pkl`
- sampling protocol, hashes, progress, and structure audits

Latest pointer: `ROOT/logs/dd_earlystop_latest.json`

Original checkpoints, training results, and all previous sampling outputs remain
in their original locations and are not overwritten.

## Verification scope

See `VERIFICATION.md`. Passing software tests is not evidence of a completed
server stop, sampling run, or quality improvement.
