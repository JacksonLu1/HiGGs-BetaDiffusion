#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "dgd",
    PROJECT_ROOT / "higgs",
):
    value = str(path)

    if value not in sys.path:
        sys.path.insert(0, value)

import numpy as np
import pandas as pd
import torch

from dgd.diffusion.beta_diffusion_utils import (
    beta_kl,
    sigmoid_alpha,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=0.09,
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--sampling-weight",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--correction-weight",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--schedule-start",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--schedule-end",
        type=float,
        default=-13.0,
    )
    parser.add_argument(
        "--schedule-power",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=5001,
    )

    return parser.parse_args()


def shifted_probability(
    unit_probability: torch.Tensor,
    scale: float,
    shift: float,
):
    return unit_probability * scale + shift


def beta_kl_values(
    a_pred: torch.Tensor,
    b_pred: torch.Tensor,
    a_true: torch.Tensor,
    b_true: torch.Tensor,
    eps: float,
):
    mask = torch.ones_like(
        a_pred,
        dtype=torch.bool,
    )

    values, diagnostics = beta_kl(
        a_pred,
        b_pred,
        a_true,
        b_true,
        eps,
        mask=mask,
        tensor_name="bayes_risk",
        context=lambda: "scalar Bayes-risk audit",
        return_diagnostics=True,
    )

    return values


def expected_risk(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_s: torch.Tensor,
    eta: float,
    eps: float,
):
    delta_raw = alpha_s - alpha_t
    delta = delta_raw.clamp_min(eps)

    eta_tensor = torch.full_like(
        prediction,
        eta,
    )

    a_pred_sampling = (
        eta_tensor * delta * prediction
    )
    b_pred_sampling = (
        eta_tensor
        * (1.0 - alpha_s * prediction)
    )

    a_true_sampling = (
        eta_tensor * delta * target
    )
    b_true_sampling = (
        eta_tensor
        * (1.0 - alpha_s * target)
    )

    a_pred_correction = (
        eta_tensor * alpha_t * prediction
    )
    b_pred_correction = (
        eta_tensor
        * (1.0 - alpha_t * prediction)
    )

    a_true_correction = (
        eta_tensor * alpha_t * target
    )
    b_true_correction = (
        eta_tensor
        * (1.0 - alpha_t * target)
    )

    sampling = beta_kl_values(
        a_pred_sampling,
        b_pred_sampling,
        a_true_sampling,
        b_true_sampling,
        eps,
    )

    correction = beta_kl_values(
        a_pred_correction,
        b_pred_correction,
        a_true_correction,
        b_true_correction,
        eps,
    )

    return sampling, correction, delta_raw, delta


def main():
    args = parse_args()

    if not args.processed_csv.is_file():
        raise FileNotFoundError(
            args.processed_csv
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = pd.read_csv(
        args.processed_csv
    )

    candidate_pairs = (
        processed["nodes"]
        * (processed["nodes"] - 1)
        / 2
    )

    edge_prevalence = float(
        processed["edges"].sum()
        / candidate_pairs.sum()
    )

    print("=" * 118)
    print("Cora H2 Beta-loss scalar Bayes-risk audit")
    print("=" * 118)
    print("Graphs             :", len(processed))
    print("Total real edges   :", processed["edges"].sum())
    print("Total possible     :", candidate_pairs.sum())
    print("Edge prevalence    :", edge_prevalence)
    print("Non-edge prevalence:", 1.0 - edge_prevalence)
    print("=" * 118)

    dtype = torch.float64

    q_grid = torch.linspace(
        0.0,
        1.0,
        args.grid_size,
        dtype=dtype,
    )

    predicted_shifted = shifted_probability(
        q_grid,
        args.scale,
        args.shift,
    )

    absent_target = torch.full_like(
        q_grid,
        args.shift,
    )

    present_target = torch.full_like(
        q_grid,
        args.shift + args.scale,
    )

    t_values = [
        0.25,
        0.40,
        0.50,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    rows = []
    curves = []

    for t_value in t_values:
        s_value = max(
            0.0,
            t_value
            - 1.0 / args.diffusion_steps,
        )

        t = torch.tensor(
            [t_value],
            dtype=dtype,
        )
        s = torch.tensor(
            [s_value],
            dtype=dtype,
        )

        alpha_t = sigmoid_alpha(
            t,
            args.schedule_start,
            args.schedule_end,
            args.schedule_power,
        )

        alpha_s = sigmoid_alpha(
            s,
            args.schedule_start,
            args.schedule_end,
            args.schedule_power,
        )

        (
            absent_sampling,
            absent_correction,
            delta_raw,
            delta_used,
        ) = expected_risk(
            predicted_shifted,
            absent_target,
            alpha_t,
            alpha_s,
            args.eta,
            args.eps,
        )

        (
            present_sampling,
            present_correction,
            _,
            _,
        ) = expected_risk(
            predicted_shifted,
            present_target,
            alpha_t,
            alpha_s,
            args.eta,
            args.eps,
        )

        expected_sampling = (
            (1.0 - edge_prevalence)
            * absent_sampling
            + edge_prevalence
            * present_sampling
        )

        expected_correction = (
            (1.0 - edge_prevalence)
            * absent_correction
            + edge_prevalence
            * present_correction
        )

        total_risk = (
            args.sampling_weight
            * expected_sampling
            + args.correction_weight
            * expected_correction
        )

        sampling_index = int(
            torch.argmin(
                expected_sampling
            ).item()
        )

        correction_index = int(
            torch.argmin(
                expected_correction
            ).item()
        )

        total_index = int(
            torch.argmin(
                total_risk
            ).item()
        )

        q_sampling = float(
            q_grid[sampling_index].item()
        )

        q_correction = float(
            q_grid[correction_index].item()
        )

        q_total = float(
            q_grid[total_index].item()
        )

        row = {
            "t": t_value,
            "s": s_value,
            "alpha_t": float(
                alpha_t.item()
            ),
            "alpha_s": float(
                alpha_s.item()
            ),
            "delta_alpha_raw": float(
                delta_raw.item()
            ),
            "delta_alpha_used": float(
                delta_used.item()
            ),
            "delta_was_clamped": int(
                float(delta_raw.item())
                < args.eps
            ),

            "edge_prevalence": (
                edge_prevalence
            ),

            "optimal_q_sampling": (
                q_sampling
            ),
            "optimal_q_correction": (
                q_correction
            ),
            "optimal_q_total": q_total,

            "total_predicts_edge": int(
                q_total > 0.5
            ),

            "risk_at_optimum": float(
                total_risk[total_index]
                .item()
            ),
        }

        rows.append(row)

        for index in range(
            args.grid_size
        ):
            curves.append(
                {
                    "t": t_value,
                    "q": float(
                        q_grid[index].item()
                    ),
                    "expected_sampling": float(
                        expected_sampling[index]
                        .item()
                    ),
                    "expected_correction": float(
                        expected_correction[index]
                        .item()
                    ),
                    "expected_total": float(
                        total_risk[index]
                        .item()
                    ),
                }
            )

        print(
            f"t={t_value:4.2f} "
            f"alpha={alpha_t.item():.9e} "
            f"delta_raw={delta_raw.item():.9e} "
            f"delta_used={delta_used.item():.9e} "
            f"q_sampling={q_sampling:.6f} "
            f"q_correction={q_correction:.6f} "
            f"q_total={q_total:.6f} "
            f"class={'EDGE' if q_total > 0.5 else 'NONEDGE'}"
        )

    summary = pd.DataFrame(rows)
    curve_frame = pd.DataFrame(curves)

    summary_csv = (
        args.output_dir
        / "cora_h2_beta_loss_bayes_summary.csv"
    )

    curves_csv = (
        args.output_dir
        / "cora_h2_beta_loss_bayes_curves.csv"
    )

    json_path = (
        args.output_dir
        / "cora_h2_beta_loss_bayes_summary.json"
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    curve_frame.to_csv(
        curves_csv,
        index=False,
    )

    json_path.write_text(
        json.dumps(
            {
                "edge_prevalence": (
                    edge_prevalence
                ),
                "rows": summary.to_dict(
                    orient="records"
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 150)
    print("Bayes-risk summary")
    print("=" * 150)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.9f}"
            ),
        )
    )

    print()
    print("Summary:", summary_csv)
    print("Curves :", curves_csv)
    print("JSON   :", json_path)
    print("=" * 150)


if __name__ == "__main__":
    main()
