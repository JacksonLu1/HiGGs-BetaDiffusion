# HOG-Diff RW/SPD input subcomponent

Source: https://github.com/Yiminghh/HOG-Diff/blob/main/models/utils.py
Original file Git blob: `5d72f438b1122d6f05b6ca9e95b135691f5b1eff`.
Original authors: Yiming Huang and Tolga Birdal, 2026.
Paper: HOG-Diff: Higher-Order Guided Diffusion for Graph Generation, ICLR 2026.

The adapted `get_rw_feat` function in `dgd/diffusion/hog_rw_features.py` is
licensed under Creative Commons Attribution-NonCommercial 4.0 International:
https://creativecommons.org/licenses/by-nc/4.0/
Full license: https://creativecommons.org/licenses/by-nc/4.0/legalcode
Upstream notice: https://github.com/Yiminghh/HOG-Diff/blob/main/LICENSE

Copyright (c) 2026 Yiming Huang and Tolga Birdal.
For commercial licensing inquiries, contact the original authors.
This third-party component is not relicensed under the surrounding project's
license. No endorsement by the original authors is claimed. No warranties.

Changes: comments and formatting; opt-in options; an external masking adapter;
concatenation into an existing HiGGs Beta-H1 denoiser. The random-walk powers
and zero-count SPD algorithm are retained. The adapter clears padding and
edge-diagonal channels. Existing cycle/spectral input features are retained.

This is a transfer experiment, NOT a complete HOG-Diff or ScoreNet replication.
It does not include the original degree/binary-adjacency input channels, CCF,
spectral diffusion or diffusion bridges. Beta formulas, losses, H2 and Glue
are unchanged. Copying a published component is not a claim of new-method
novelty, and no generation-quality gain is asserted before experiments.
