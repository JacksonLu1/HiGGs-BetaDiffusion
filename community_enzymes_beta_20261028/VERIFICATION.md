# Verification boundary

Reviewed original project conversation 019f4461-2422-74e1-b9e1-7a1bddff2f42,
the actual original r1 manifest bytes, raw hashes, baseline v4 worker, configs,
and cache semantics. No messages were sent to that chat, and no original files
were modified. Old author80/20 and r2 experiments are not reused.

Local checks:

- 16 unittest methods passed. Includes both exact manifest bytes, all18 actual
  Windows-original caches, complete-graph disjointness, actual sample counts,
  cache-only refusal,12 Hydra configuration combinations,7 sampling/report tests,
  and memory-only launch checks.
- The three existing model classes passed synthetic CPU forward/backward,
  oneepoch train+validation, finite monitored score, and strict checkpoint reload
  in an isolated temporary copy with the HOG overlay. H1 actual widths12/6/14.
- The CPU test is not part of the unittest method count. It does not establish
  real-cache GPU integration, speed, memory demand or generation quality.
- A Windows WandB temporary-directory atexit cleanup permission warning occurred
  during the first local run; stage checks completed and exitcode was0. Repeated
  unit tests with a workspace temporary directory finished normally.

No Linux/server installation, real GPU smoke,500epoch training, fullchain sampling
or metric evaluation has been performed by the assistant. User must upload and
run server commands; do not treat a generated ZIP or printed PID as success.

Installed sources are additions only. Existing shared DD-audited sources remain
byte-identical. No dataset redrawing, threshold calibration, output filtering,
model weight initialization transfer, test-based selection or process killing.
