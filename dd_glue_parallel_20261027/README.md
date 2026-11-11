# Live DD Glue parallel scheduling for the verified current run

This package adds a new scheduling helper only. It does not edit the running
formal driver, main.py, models, configs, caches, checkpoints or original HiGGs.

Target: modified root `/path/to/HiGGs-GBD/HiGGs-main`, run
`dd_beta_mean_hog_r2_e500_scratch_seed0_20261027_205633_003751`.

The user verified scheduler2361431, H1child2361463 and H2child2361464.
Glue had not started; GPU2 had approximately39GiB free. Runtime checks are still
mandatory because process/GPU state can change after that snapshot.

## What changes

1. Read-only preflight verifies exact process commands, UID, starttime, DD lock
   ownership/inode, no existing Glue output/process, available memory, the frozen
   manifest and successful smoke input/source audit, plus known formal-driver SHA.
2. A detached replacement scheduler obtains an independent migration lock.
   It persists the handoff intent before signaling the old scheduler. Metadata
   write failures after Glue spawn are warnings, not grounds to drop the DD lock.
3. Only the exact old scheduler is briefly SIGSTOPped, never H1/H2. Glue paths
   are checked again while scheduling is frozen, avoiding the H2->Glue race.
   A failure before termination SIGCONTs the old scheduler.
4. Only that scheduler receives SIGTERM and SIGCONT through its pinned pidfd.
   No killpg, pkill, SIGKILL or signal to a training child is used.
5. After old scheduler exit and acquisition of the same DD file lock, existing
   H1/H2 identities are checked and the original, unchanged formal driver starts
   stage=glue on GPU2 inside the same output directory.
6. All three stages remain500 epochs. Existing H1/H2 are not resumed/restarted;
   their PIDs remain the same while PPID may become1 after parent exit.
7. The replacement holds the DD lock until stages finish. Only valid epoch499
   stage reports and matching checkpoint/source hashes produce the final audit.
   Adopted H1/H2 OS exit codes are unavailable and are not fabricated as zero.
   A fatal supervision error keeps the lock until known existing training
   processes exit; it never kills or silently restarts them.

The helper requires Linux/Python pidfd support and refuses mismatched identities,
changed source/data, an already-started Glue or a different run. Do not force it.
It checks only free memory, not utilization or other users' GPU processes.

## Linux execution

Upload the ZIP to the modified root, then run:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
OUT="$ROOT/outputs/dd_beta_mean_hog_r2_e500_scratch_seed0_20261027_205633_003751"
cd "$ROOT"
unzip -n dd_glue_parallel_20261027.zip
"$PY" -B dd_glue_parallel_20261027/dd_glue_parallel.py \
  --root "$ROOT" --out "$OUT" \
  --parent-pid 2361431 --h1-pid 2361463 --h2-pid 2361464 --gpu 2
```

Successful foreground launch prints PID/LOG/OUTPUT. Actual handoff success is
`PASS: DD_GLUE_PARALLEL_STARTED` in the new log, not just the printed PID.
Do not launch twice, delete lock files, or edit running training sources.

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
RUN=dd_beta_mean_hog_r2_e500_scratch_seed0_20261027_205633_003751
LOG="$(ls -t "$ROOT/logs/training/${RUN}_glue_parallel_"*.log 2>/dev/null | head -1)"
if [ -n "$LOG" ]; then
  printf 'LOG=%s\n' "$LOG"
  tail -n 70 -F "$LOG"
else echo "No parallel scheduler log found"; fi
```

The existing stage logs remain `<OUT>/h2.log`, `<OUT>/h1.log`, `<OUT>/glue.log`.
After successful takeover, dd_latest_train.json points to the replacement and
the same effective OUT; its previous value is retained in the handoff audit.
Final marker remains
`PASS: DD_BETA_MEAN_HOG_THREE_STAGE_FORMAL_TRAINING_COMPLETE`.
No sampling or metric evaluation is automatically launched.

Keep this helper in place until training/sampling audit is complete; its hash is
included in the final provenance. Do not overwrite the verified deployment.

## Verification boundary

Local tests mock every process signal, pidfd, GPU call and process launch. No
actual Linux server, existing process or GPU is operated by local verification.
The server must confirm the start marker and that H1/H2 PIDs remain alive or
finish with valid reports. Training quality/speed is not established by tests.
