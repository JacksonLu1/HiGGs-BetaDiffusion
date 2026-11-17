# H1 noisy k-core experiment

Opt in with `model=beta_h1_twohop_mean_hog_rw_kcore` in the standard H1
training entry of this isolated copy. With the original model preset, k-core
is off and original checkpoints retain the same tensor shapes and semantics.
The experimental checkpoint requires matching k-core metadata; it is not
interchangeable with a baseline checkpoint. Use a new H1 run directory.

The model still generates each original Louvain community as one H1 graph.
Only its current noisy, thresholded adjacency is read by the new feature
extractor, in the same function for training, validation and reverse sampling.
Inputs gain a normalized core-number channel, a k-core membership channel,
and a pairwise channel indicating the same connected component inside the
induced k-core. Masks are retained, and the added channels are zero at
high noise (`t > max_t`). `k=3` and `max_t=0.5` are starting settings,
not tuned values. Beta transitions, mean two-hop, HOG RW/SPD, graph splits,
Louvain preprocessing, H2, Glue and the mixed evaluation are unchanged.

Run the synthetic checks with `python -B tests/test_h1_kcore_features.py`
and `python -B tests/test_h1_kcore_integration.py` from this copy. Do not
overwrite the active server H1 module: some old launcher scripts pin its
hash or H1 input widths. An isolated run needs its own H1 audit/launcher
and any dataset-specific sampling wrapper must accept the added X/E widths.
Before long training, check feature coverage across noise and community size,
CPU/GPU overhead, and compare the fixed validation protocol with the same
seeds and H1 configuration apart from the new features.
