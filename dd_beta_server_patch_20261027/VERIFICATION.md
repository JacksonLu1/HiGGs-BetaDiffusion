# Local verification, 2026-10-27

- 22 CPU/unit/static checks passed with the existing higgs_env Python runtime.
- DD manifest checks use synthetic graph IDs/partitions, not a fabricated real
  DD manifest. The real shared manifest SHA is read from the original server
  checkout during installation and copied byte-for-byte.
- The exact exported modified main.py plus the already-proven Planar patch was
  reconstructed in memory. Its SHA matches the install precondition. Adding DD
  then parses, preserves Planar, and adds one DD route. No production file was
  changed by this verification.
- The isolated node prior preserves the original categorical sampling draws
  under the same RNG seed. Floating log probabilities, missing-size support,
  empty input and dataset scoping were numerically tested on CPU.
- Driver AST checks cover the selected model groups, fixed shared split settings,
  500-epoch full-batch formal mode, seed1234 sampling, 184 formal graphs,
  structure-only class0 H1 and class1 Glue.
- Launcher completion guards were tested against missing reports, wrong epochs,
  failed strict loads, and changed sources/checkpoints. GPU calls were mocked;
  only free memory is queried, not utilization or active-process occupancy.
- Independent code review checked runtime strict loader bindings, DD conditions,
  no-meta-edge handling, path separation and installer rollback. HOG/two-hop
  component files and inherited model configurations were added to input hashes.
- No real DD cache was rebuilt, no real dataset was trained or sampled, no
  server process was signaled, and no existing Windows model/source was edited.

These checks do not establish a successful Linux installation, real-cache GPU
training, speed, memory fit or generation quality. The server smoke is mandatory
before formal training; it tests all three stages and their real checkpoints.
