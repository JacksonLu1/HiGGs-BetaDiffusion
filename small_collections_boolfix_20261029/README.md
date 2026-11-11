# Runtime Boolean Glue mask correction

The original small-collection sampling adapter mistakenly cast an intra-community
Boolean mask to float (`pos=pos.float()`). The unchanged discrete Glue sampler
uses `~pos`; float masks fail. This is an adapter error, not a failed model or GPU.

This new entry calls the SAME original sampler and converts strictly binary,
finite `Data.pos` values to bool immediately after pair construction, BEFORE
PyG DataLoader collation. It preserves mask values/shape/device, x/y, all nodes
and edges, checkpoint choices, Beta/discrete transitions, RNGseed1234, batch sizes,
full components, graph count and evaluation protocol.

No original adapter, setup file, training input audit, checkpoint, dataset or
shared source is edited. Original audits remain truthful. The new wrapper's own
SHA and correction details are recorded separately in `boolfix_audit.json`.
Do NOT manually update old audit hashes or reinstall a changed original package.

Upload this new ZIP to the modified server root, check its chat-provided SHA,
and unzip. It adds only a new repair directory:

```bash
ROOT=/path/to/HiGGs-GBD/HiGGs-main
PY=/path/to/conda/envs/higgs/bin/python
cd "$ROOT"
unzip -n small_collections_boolfix_20261029.zip
"$PY" -B "$ROOT/small_collections_boolfix_20261029/run_boolfix.py" \
  --dataset enzymes --gpu 3 \
  --training-output "$ROOT/outputs/enzymes_beta_mean_hog_r1_train_20261028_225740_474594_gpu3"
```

This launches ONLY a new117graph sampling run, no retraining. It rechecks all
three formal results, checkpoint hashes and original source/input audits; the
same inherited datasetflock prevents duplicate active sampling/training. It sends
no signals and checks only8GiB free memory, never GPUutilization/exclusivity.
If an old task still holds the lock, do not force/remove it; inspect that task.

The failed output is left intact. Restart seed1234 from graph0 because the shown
failure occurred before the first complete graph was saved. Never reuse the
old failed output directory. No samples are selected or discarded for quality.

Community-small uses the SAME fix, after its formal training passes:

```bash
"$PY" -B "$ROOT/small_collections_boolfix_20261029/run_boolfix.py" \
  --dataset community_small --gpu 2 \
  --training-output "$ROOT/outputs/community_small_beta_mean_hog_r1_train_20261028_225659_310568_gpu2"
```

Existing latest-sampling viewing commands still work: the standard
`logs/<dataset>_beta_latest_sample.json` pointer is updated to the new job.
Its main log contains the full sampling progress; graphs are under
reportedOUTPUT/sample/sampling/.

After the first nonempty Glue pair preparation expect
`PASS: GLUE_POS_BOOL_RUNTIME`. Final repair-run markers are:
`PASS: ENZYMES_BOOLMASK_FIX_FULLCHAIN_COMPLETE` or
`PASS: COMMUNITY_SMALL_BOOLMASK_FIX_FULLCHAIN_COMPLETE`.
Completion is sampling success only, not a quality-improvement verdict.

Local verification: binary-mask/value/attribute/RNG preservation, real PyG
collation and the ACTUAL existing discrete Glue reverse-sampling method on
synthetic CPU inputs. No server/GPU run or real-dataset quality claim is made.
Eight regression/unit tests passed. The real-method test first reproduces the
float-mask TypeError, then completes with the Boolean runtime correction and
checks all intra-community edges/nonedges and model parameters stay unchanged.
