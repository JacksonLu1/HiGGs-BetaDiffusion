#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "higgs",
    PROJECT_ROOT / "dgd",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import higgs.sampling as sampling
from higgs.interface_ablation import seed_sampling


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
    )

    return parser.parse_args()


def quantiles(values):
    values = np.asarray(values, dtype=float)

    return {
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def main():
    args = parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = OmegaConf.load(args.config)

    cfg_h2 = cfg.copy()
    cfg_h2["dataset"]["h"] = 2
    cfg_h2["model"]["extra_features"] = "all"

    cfg_h2, model = sampling.get_samplers(
        cfg_h2
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and int(cfg.general.gpus) > 0
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    condition = sampling.fixed_h2_condition(
        model.ydim_output
    ).to(device)

    model_module = sys.modules[
        model.__class__.__module__
    ]

    original_conversion = (
        model_module.shifted_clean_to_unit_probability
    )

    capture = {}

    def capture_conversion(
        clean_values,
        scale,
        shift,
    ):
        converted = original_conversion(
            clean_values,
            scale,
            shift,
        )

        # Edge tensor shape: [B, N, N, C]
        if clean_values.ndim == 4:
            capture["pre_conversion"] = (
                clean_values.detach()
                .cpu()
                .clone()
            )
            capture["post_conversion"] = (
                converted.detach()
                .cpu()
                .clone()
            )
            capture["scale"] = float(scale)
            capture["shift"] = float(shift)

        return converted

    model_module.shifted_clean_to_unit_probability = (
        capture_conversion
    )

    seed_rows = []
    all_pre = []
    all_post = []
    all_positive_post = []
    all_negative_post = []
    isolated_rows = []

    try:
        for offset in range(args.count):
            seed = args.start_seed + offset

            seed_sampling(seed)
            capture.clear()

            with torch.inference_mode():
                predicted, _ = model.sampling_batch_h2(
                    batch_id=0,
                    batch_size=1,
                    keep_chain=0,
                    number_chain_steps=40,
                    num_nodes=None,
                    figures=False,
                    y=condition.clone(),
                    graph_number=seed,
                )

            if "pre_conversion" not in capture:
                raise RuntimeError(
                    f"Seed {seed}: edge conversion not captured"
                )

            n = int(predicted[0][0].shape[0])

            pre_tensor = capture[
                "pre_conversion"
            ][0, :n, :n]

            post_tensor = capture[
                "post_conversion"
            ][0, :n, :n]

            pre_score = (
                pre_tensor.max(dim=-1).values.numpy()
            )
            post_score = (
                post_tensor.max(dim=-1).values.numpy()
            )

            upper = np.triu_indices(n, k=1)

            pre = pre_score[upper].astype(float)
            post = post_score[upper].astype(float)

            positive = post > 0.5
            negative = ~positive

            all_pre.extend(pre.tolist())
            all_post.extend(post.tolist())
            all_positive_post.extend(
                post[positive].tolist()
            )
            all_negative_post.extend(
                post[negative].tolist()
            )

            adjacency = np.zeros(
                (n, n),
                dtype=np.uint8,
            )

            adjacency[upper] = positive.astype(
                np.uint8
            )
            adjacency = adjacency + adjacency.T

            degrees = adjacency.sum(axis=1)
            isolated = np.where(degrees == 0)[0]

            for node in isolated:
                incident_post = np.concatenate(
                    (
                        post_score[node, :node],
                        post_score[node, node + 1 :],
                    )
                )

                incident_pre = np.concatenate(
                    (
                        pre_score[node, :node],
                        pre_score[node, node + 1 :],
                    )
                )

                best_index = int(
                    np.argmax(incident_post)
                )

                isolated_rows.append(
                    {
                        "seed": seed,
                        "node": int(node),
                        "max_post_score": float(
                            incident_post[best_index]
                        ),
                        "corresponding_pre_value": float(
                            incident_pre[best_index]
                        ),
                    }
                )

            seed_rows.append(
                {
                    "seed": seed,
                    "nodes": n,
                    "edges": int(positive.sum()),
                    "isolates": int(len(isolated)),

                    "pre_min": float(pre.min()),
                    "pre_median": float(
                        np.median(pre)
                    ),
                    "pre_mean": float(pre.mean()),
                    "pre_max": float(pre.max()),

                    "post_min": float(post.min()),
                    "post_median": float(
                        np.median(post)
                    ),
                    "post_mean": float(post.mean()),
                    "post_max": float(post.max()),

                    "post_le_1e_6_fraction": float(
                        (post <= 1e-6).mean()
                    ),
                    "post_le_1e_3_fraction": float(
                        (post <= 1e-3).mean()
                    ),
                    "post_0_01_to_0_99_fraction": float(
                        (
                            (post > 0.01)
                            & (post < 0.99)
                        ).mean()
                    ),
                    "post_ge_0_99_fraction": float(
                        (post >= 0.99).mean()
                    ),
                    "post_ge_1_minus_1e_6_fraction": float(
                        (
                            post >= 1.0 - 1e-6
                        ).mean()
                    ),
                    "rounded_unique_post_values": int(
                        len(np.unique(
                            np.round(post, 6)
                        ))
                    ),
                }
            )

            print(
                f"[{offset + 1:03d}/{args.count:03d}] "
                f"seed={seed:3d} "
                f"nodes={n:2d} "
                f"edges={int(positive.sum()):3d} "
                f"isolates={len(isolated)} "
                f"mid-prob-frac="
                f"{((post > 0.01) & (post < 0.99)).mean():.6f} "
                f"near-one-frac="
                f"{(post >= 1 - 1e-6).mean():.6f}",
                flush=True,
            )

    finally:
        model_module.shifted_clean_to_unit_probability = (
            original_conversion
        )

    seed_df = pd.DataFrame(seed_rows)
    isolated_df = pd.DataFrame(isolated_rows)

    seed_csv = (
        args.output_dir
        / "cora_h2_saturation_per_seed.csv"
    )
    isolated_csv = (
        args.output_dir
        / "cora_h2_isolated_node_scores.csv"
    )

    seed_df.to_csv(seed_csv, index=False)
    isolated_df.to_csv(isolated_csv, index=False)

    summary = {
        "count": args.count,
        "edge_scale": float(model.edge_scale),
        "edge_shift": float(model.edge_shift),
        "input_space": str(model.input_space),
        "threshold": float(model.edge_threshold),

        "pre_conversion": quantiles(all_pre),
        "post_conversion": quantiles(all_post),

        "positive_post_scores": (
            quantiles(all_positive_post)
        ),
        "negative_post_scores": (
            quantiles(all_negative_post)
        ),

        "post_fraction_le_1e_6": float(
            (np.asarray(all_post) <= 1e-6).mean()
        ),
        "post_fraction_le_1e_3": float(
            (np.asarray(all_post) <= 1e-3).mean()
        ),
        "post_fraction_between_0_01_and_0_99": float(
            (
                (np.asarray(all_post) > 0.01)
                & (np.asarray(all_post) < 0.99)
            ).mean()
        ),
        "post_fraction_ge_0_99": float(
            (np.asarray(all_post) >= 0.99).mean()
        ),
        "post_fraction_ge_1_minus_1e_6": float(
            (
                np.asarray(all_post)
                >= 1.0 - 1e-6
            ).mean()
        ),
    }

    if not isolated_df.empty:
        summary["isolated_nodes"] = {
            "count": int(len(isolated_df)),
            "post_max_scores": quantiles(
                isolated_df["max_post_score"]
            ),
            "corresponding_pre_values": quantiles(
                isolated_df[
                    "corresponding_pre_value"
                ]
            ),
        }

    summary_path = (
        args.output_dir
        / "cora_h2_saturation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("Beta-H2 edge-score saturation audit")
    print("=" * 110)
    print("Input space:", model.input_space)
    print("Edge scale :", model.edge_scale)
    print("Edge shift :", model.edge_shift)
    print("Threshold  :", model.edge_threshold)

    print("\nPre-conversion distribution:")
    print(summary["pre_conversion"])

    print("\nPost-conversion distribution:")
    print(summary["post_conversion"])

    print("\nPositive post scores:")
    print(summary["positive_post_scores"])

    print("\nNegative post scores:")
    print(summary["negative_post_scores"])

    print(
        "\nPost fraction <= 1e-6:",
        summary["post_fraction_le_1e_6"],
    )
    print(
        "Post fraction <= 1e-3:",
        summary["post_fraction_le_1e_3"],
    )
    print(
        "Post fraction in (0.01, 0.99):",
        summary[
            "post_fraction_between_0_01_and_0_99"
        ],
    )
    print(
        "Post fraction >= 0.99:",
        summary["post_fraction_ge_0_99"],
    )
    print(
        "Post fraction >= 1-1e-6:",
        summary[
            "post_fraction_ge_1_minus_1e_6"
        ],
    )

    if "isolated_nodes" in summary:
        print("\nIsolated node diagnostics:")
        print(summary["isolated_nodes"])

    print("\nCSV seeds   :", seed_csv)
    print("CSV isolates:", isolated_csv)
    print("JSON        :", summary_path)
    print("=" * 110)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
