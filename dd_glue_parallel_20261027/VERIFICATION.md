# Local verification, 2026-10-27

- 27 standard-library CPU/mock tests passed, including ResourceWarning-as-error.
- Exact run/stage/UID/PPID/starttime identity and PID-reuse refusal were tested.
- No actual process signal, pidfd, GPU query or training launch occurred in tests.
- Successful handoff signals only the pinned scheduler: STOP, TERM, CONT.
  No process-group or training-child signal is issued.
- Discovering queued Glue after scheduler freeze restores the old scheduler
  and refuses duplication. A child finishing with a valid report is accepted.
- Intent-write failure occurs before transfer or spawn. Metadata-write failures
  after Glue spawn continue supervision; fatal errors drain known training while
  retaining the DD lock. Migration lock handles are closed on early failures.
- The successful smoke audit, frozen manifest and exact formal-driver SHA are
  mandatory. Missing/invalid stage reports or changed checkpoints/sources cannot
  produce the final completed input audit.
- Independent review found two post-transfer metadata failure gaps; both were
  repaired and regression-tested before packaging.

No real server process or GPU was operated. These checks do not establish real
handoff completion, training performance or generation quality. The server log
must confirm DD_GLUE_PARALLEL_STARTED and the preserved H1/H2 identities.
