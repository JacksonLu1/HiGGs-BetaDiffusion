# SBM resolution-1 C4 experiment manifest

## Experiment identity

- Experiment name: C4
- Model chain: Beta-H2 → Beta-H1 → retrained Original edge-DiGress Glue
- Dataset: SBM
- Hierarchy resolution: 1
- Sampling seed: 0
- Generated samples: 40
- Formal output directory: `higgs/outputs/formal_C4_retrained_glue_seed0_n40_retry1`

## Checkpoints

| Stage | Checkpoint filename | SHA256 |
|---|---|---|
| H2 | `sbm_beta_h2_r1_e500_best.ckpt` | `1BBB4893FF243FA566B249B8FD857BF7E8C4D9A36D1FF73164EA2BBE5216A17E` |
| H1 | `sbm_beta_h1_r1_best.ckpt` | `C69446B3B1B928351AD18A4E8847A429924356C2F6BE6C5145AF277CB540DF99` |
| Glue | `sbm_glue_retrain_r1_e500_best.ckpt` | `212AF1BBD0A0EF110F8456026588B7BA46F4286DBAB1479A52C58A4D2B3E65FE` |

- Glue checkpoint epoch: 499
- Glue checkpoint global step: 5000
- Checkpoint and dataset files are intentionally not committed to Git.

## Sampling interface

- H2 component policy: `upstream_lcc`
- H1 component policy: `upstream_lcc`
- H2-to-H1 condition mode: `sbm_semantic`
- H2-to-Glue condition mode: `sbm_semantic`
- Glue present class index: `0`
- Force evaluation mode: `true`

The formal run used:

```text
general=sample_sbm_beta_h2_beta_h1_original_glue
dataset=sbm_r1
model=beta_h2
general.samples_to_generate=40
general.sampling_seed=0
general.wandb=disabled
general.visualize=false
general.intermediate_figures=false
general.interface.h2_component_policy=upstream_lcc
general.interface.h1_component_policy=upstream_lcc
general.interface.h2_to_h1_condition_mode=sbm_semantic
general.interface.h2_to_glue_condition_mode=sbm_semantic
general.interface.glue_present_class_index=0
general.interface.force_eval=true
general.interface.save_intermediates=false
general.interface.strict_assertions=true
```

## Mixed paper post-processing metrics

Metric modes:

- Nodes: Gaussian-TV
- Degree: Gaussian-EMD
- Clustering: Gaussian-EMD
- Eccentricity: Gaussian
- Spectre: Gaussian-EMD

C4 formal results:

| Metric | Value |
|---|---:|
| Nodes | 0.462228 |
| Degree | 0.232440 |
| Clustering | 0.325272 |
| Eccentricity | 0.006804 |
| Spectre | 0.248930 |

The generated graphs, checkpoints, raw/processed datasets, and local run
outputs remain outside Git. Reproduction requires obtaining the three
checkpoint files separately and verifying the SHA256 values above.
