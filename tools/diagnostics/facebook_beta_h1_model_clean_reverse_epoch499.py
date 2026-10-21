from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data2/wenb/ylu/HiGGs-GBD/HiGGs-main")
DGD = ROOT / "dgd"
DIAGNOSTICS = ROOT / "tools" / "diagnostics"

sys.path.insert(0, str(DGD))
sys.path.insert(0, str(DIAGNOSTICS))

import facebook_beta_h1_edge_oracle_transition_epoch499 as oracle
import facebook_beta_h1_teacher_forced_epoch499 as teacher

from diffusion.beta_diffusion_utils import (
    apply_graph_masks,
    forward_beta_sample,
    reverse_beta_step,
    shifted_clean_to_unit_probability,
)

base = oracle.base
terminal = oracle.terminal

CHECKPOINT = ROOT / (
    "outputs/2026-07-30/10-58-34/checkpoints/"
    "fb_beta_h1_r10_e500/last.ckpt"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "higgs/outputs"
    / "facebook_beta_h1_model_clean_reverse_epoch499_seed20260830"
)

SEED = 20260830
TIMES = (1.00, 0.75, 0.50)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weighted_mean(rows, key):
    weight = sum(row["candidate_count"] for row in rows)
    if weight == 0:
        return None
    return sum(
        row[key] * row["candidate_count"]
        for row in rows
    ) / weight


def tensor_summary(values):
    values = values.detach().double()
    return {
        "min": float(values.min().cpu()),
        "mean": float(values.mean().cpu()),
        "max": float(values.max().cpu()),
        "std": float(values.std(unbiased=False).cpu()),
        "nonfinite_count": int(
            (~torch.isfinite(values)).sum().cpu()
        ),
    }


@torch.inference_mode()
def audit_endpoint(model, record, endpoint_id, device, repeats):
    values = terminal.build_real_tensors(record, model, device)

    n = values["n"]
    node_mask = values["node_mask"].expand(repeats, -1).contiguous()
    condition_y = values["condition_y"].expand(
        repeats, -1
    ).contiguous()

    X0 = values["X0"].expand(repeats, -1, -1).contiguous()
    E0 = values["E0"].expand(
        repeats, -1, -1, -1
    ).contiguous()

    eta_X, eta_E = model._eta_tensors(repeats, n, X0)

    upper = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=device),
        diagonal=1,
    )
    labels_one = values["E_discrete"][0, :, :, 0][upper].long()
    labels = labels_one.unsqueeze(0).expand(
        repeats, -1
    ).reshape(-1).cpu()

    rows = []

    for time_index, t_value in enumerate(TIMES):
        s_value = max(t_value - 1.0 / model.T, 0.0)

        t = torch.full(
            (repeats, 1), t_value, dtype=X0.dtype, device=device
        )
        s = torch.full(
            (repeats, 1), s_value, dtype=X0.dtype, device=device
        )
        alpha_t = model._alpha(t)
        alpha_s = model._alpha(s)

        forward_seed = (
            SEED + endpoint_id * 1000 + time_index * 100
        )
        seed_everything(forward_seed)

        if model.diffuse_X:
            X_t, X_diag = forward_beta_sample(
                X0,
                alpha_t.view(-1, 1, 1),
                eta_X,
                model.beta_eps,
                model.input_space,
            )
        else:
            X_t = model._shifted_to_state(X0)
            X_diag = None

        E_t, E_diag = forward_beta_sample(
            E0,
            alpha_t.view(-1, 1, 1, 1),
            eta_E,
            model.beta_eps,
            model.input_space,
        )
        X_t, E_t = apply_graph_masks(X_t, E_t, node_mask)

        # Exact teacher-forced model call used by the existing diagnostic.
        pred_X0, pred_E0 = model._predict_clean(
            X_t,
            E_t,
            condition_y,
            t,
            alpha_t,
            eta_X,
            eta_E,
            node_mask,
        )

        if pred_E0.shape != E0.shape:
            raise RuntimeError(
                f"Unexpected pred_E0 shape {tuple(pred_E0.shape)}; "
                f"expected {tuple(E0.shape)}"
            )

        clean_unit = shifted_clean_to_unit_probability(
            pred_E0, model.edge_scale, model.edge_shift
        )[:, :, :, 0]
        clean_upper = clean_unit[:, upper].reshape(-1).cpu()

        # Same E_t and same reverse random draw: only clean condition differs.
        reverse_seed = forward_seed + 50

        seed_everything(reverse_seed)
        E_oracle_s, oracle_diag = reverse_beta_step(
            E_t,
            E0,
            alpha_t.view(-1, 1, 1, 1),
            alpha_s.view(-1, 1, 1, 1),
            eta_E,
            model.beta_eps,
            model.input_space,
        )

        seed_everything(reverse_seed)
        E_model_s, model_diag = reverse_beta_step(
            E_t,
            pred_E0,
            alpha_t.view(-1, 1, 1, 1),
            alpha_s.view(-1, 1, 1, 1),
            eta_E,
            model.beta_eps,
            model.input_space,
        )

        # Formal valid-node / diagonal masking only; no thresholding.
        _, E_oracle_s = apply_graph_masks(
            X_t, E_oracle_s, node_mask
        )
        _, E_model_s = apply_graph_masks(
            X_t, E_model_s, node_mask
        )

        oracle_unit = oracle.raw_beta_probability(
            E_oracle_s, model.input_space
        )[:, :, :, 0]
        model_unit = oracle.raw_beta_probability(
            E_model_s, model.input_space
        )[:, :, :, 0]

        oracle_upper = oracle_unit[:, upper].reshape(-1).cpu()
        model_upper = model_unit[:, upper].reshape(-1).cpu()

        clean_metrics = teacher.binary_metrics(
            [labels], [clean_upper], model.edge_threshold
        )
        state_metrics = teacher.binary_metrics(
            [labels], [model_upper], model.edge_threshold
        )

        clean_error = (
            clean_unit[:, upper]
            - values["E_discrete"][0, :, :, 0][upper]
        ).abs()

        model_oracle_delta = (
            model_unit[:, upper] - oracle_unit[:, upper]
        ).abs()

        diagonal = torch.eye(
            n, dtype=torch.bool, device=device
        )

        candidate_count = int(repeats * upper.sum().cpu())
        gamma_count = int(E_diag.gamma_clamp_count.cpu())
        gamma_count += int(oracle_diag.gamma_clamp_count.cpu())
        gamma_count += int(model_diag.gamma_clamp_count.cpu())
        if X_diag is not None:
            gamma_count += int(X_diag.gamma_clamp_count.cpu())

        row = {
            "endpoint_id": endpoint_id,
            "condition_y": int(record["condition_y"]),
            "nodes": n,
            "true_edges": int(labels_one.sum().cpu()),
            "t": t_value,
            "s": s_value,
            "candidate_count": candidate_count,
            "true_positive_rate": float(labels.float().mean()),

            "clean_auroc": clean_metrics["auroc"],
            "clean_auprc": clean_metrics["auprc"],
            "clean_positive_probability_mean": (
                clean_metrics["positive_probability_mean"]
            ),
            "clean_negative_probability_mean": (
                clean_metrics["negative_probability_mean"]
            ),
            "clean_predicted_positive_rate": (
                clean_metrics["predicted_positive_rate"]
            ),
            "clean_unit_mae_to_true": float(clean_error.mean().cpu()),

            "model_state_auroc": state_metrics["auroc"],
            "model_state_auprc": state_metrics["auprc"],
            "model_state_positive_rate_at_0_5": (
                state_metrics["predicted_positive_rate"]
            ),
            "model_state_positive_mean": (
                state_metrics["positive_probability_mean"]
            ),
            "model_state_negative_mean": (
                state_metrics["negative_probability_mean"]
            ),

            "oracle_state_mean": float(oracle_upper.mean()),
            "model_state_mean": float(model_upper.mean()),
            "model_oracle_state_mae": float(
                model_oracle_delta.mean().cpu()
            ),
            "model_oracle_state_max_abs_delta": float(
                model_oracle_delta.max().cpu()
            ),

            "clean_summary": tensor_summary(clean_unit[:, upper]),
            "oracle_state_summary": tensor_summary(
                oracle_unit[:, upper]
            ),
            "model_state_summary": tensor_summary(
                model_unit[:, upper]
            ),

            "pred_clean_asymmetry_max": float(
                (clean_unit - clean_unit.transpose(1, 2))
                .abs().max().cpu()
            ),
            "model_state_asymmetry_max": float(
                (model_unit - model_unit.transpose(1, 2))
                .abs().max().cpu()
            ),
            "pred_clean_diagonal_nonzero": int(
                (clean_unit[:, diagonal].abs() > 1e-7).sum().cpu()
            ),
            "model_state_diagonal_nonzero": int(
                (model_unit[:, diagonal].abs() > 1e-7).sum().cpu()
            ),
            "gamma_clamp_count": gamma_count,
            "nonfinite_count": int(
                (~torch.isfinite(pred_X0)).sum().cpu()
                + (~torch.isfinite(pred_E0)).sum().cpu()
                + (~torch.isfinite(E_oracle_s)).sum().cpu()
                + (~torch.isfinite(E_model_s)).sum().cpu()
            ),

            "_labels": labels,
            "_clean_scores": clean_upper,
            "_model_state_scores": model_upper,
        }
        rows.append(row)

    return rows


def summarize(rows, threshold):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["t"])].append(row)

    summary = {}
    for time_key, current in sorted(
        grouped.items(), key=lambda item: float(item[0]), reverse=True
    ):
        labels = [row["_labels"] for row in current]
        clean_scores = [row["_clean_scores"] for row in current]
        model_scores = [row["_model_state_scores"] for row in current]

        clean = teacher.binary_metrics(
            labels, clean_scores, threshold
        )
        model_state = teacher.binary_metrics(
            labels, model_scores, threshold
        )

        summary[time_key] = {
            "endpoint_count": len(current),
            "candidate_count": sum(
                row["candidate_count"] for row in current
            ),
            "true_positive_rate": clean["true_positive_rate"],

            "clean_auroc": clean["auroc"],
            "clean_auprc": clean["auprc"],
            "clean_positive_probability_mean": (
                clean["positive_probability_mean"]
            ),
            "clean_negative_probability_mean": (
                clean["negative_probability_mean"]
            ),
            "clean_predicted_positive_rate": (
                clean["predicted_positive_rate"]
            ),
            "clean_unit_mae_to_true": weighted_mean(
                current, "clean_unit_mae_to_true"
            ),

            "model_state_auroc": model_state["auroc"],
            "model_state_auprc": model_state["auprc"],
            "model_state_positive_rate_at_0_5": (
                model_state["predicted_positive_rate"]
            ),
            "model_state_positive_mean": (
                model_state["positive_probability_mean"]
            ),
            "model_state_negative_mean": (
                model_state["negative_probability_mean"]
            ),

            "oracle_state_mean": weighted_mean(
                current, "oracle_state_mean"
            ),
            "model_state_mean": weighted_mean(
                current, "model_state_mean"
            ),
            "model_oracle_state_mae": weighted_mean(
                current, "model_oracle_state_mae"
            ),
            "model_oracle_state_max_abs_delta": max(
                row["model_oracle_state_max_abs_delta"]
                for row in current
            ),
            "gamma_clamp_count": sum(
                row["gamma_clamp_count"] for row in current
            ),
            "nonfinite_count": sum(
                row["nonfinite_count"] for row in current
            ),
            "pred_clean_asymmetry_max": max(
                row["pred_clean_asymmetry_max"]
                for row in current
            ),
            "model_state_asymmetry_max": max(
                row["model_state_asymmetry_max"]
                for row in current
            ),
            "pred_clean_diagonal_nonzero": sum(
                row["pred_clean_diagonal_nonzero"]
                for row in current
            ),
            "model_state_diagonal_nonzero": sum(
                row["model_state_diagonal_nonzero"]
                for row in current
            ),
        }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all 20 fixed-validation endpoints.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="Independent forward/reverse draws per endpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    selected, selection_histogram = (
        base.extract_validation_endpoints()
    )
    if len(selected) != 20:
        raise RuntimeError(
            f"Expected 20 fixed-validation endpoints, "
            f"found {len(selected)}"
        )
    if args.limit:
        selected = selected[:args.limit]

    checkpoint = base._load_checkpoint_cpu(CHECKPOINT)
    model = base.ConditionalGraphBetaDiffusionH1.load_from_checkpoint(
        str(CHECKPOINT), map_location="cpu"
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    print("checkpoint:", CHECKPOINT)
    print("epoch:", checkpoint.get("epoch"))
    print("global_step:", checkpoint.get("global_step"))
    print("device:", device)
    print("endpoint count:", len(selected))
    print("repeats:", args.repeats)
    print("times:", TIMES)
    print("edge threshold:", model.edge_threshold)
    print(
        "Mode: same true E_t; compare true E0 oracle "
        "against model-predicted shifted E0."
    )

    rows = []
    for endpoint_id, record in enumerate(selected):
        print(
            f"Auditing endpoint {endpoint_id + 1}/{len(selected)}, "
            f"y={record['condition_y']}, "
            f"n={record['graph'].number_of_nodes()}"
        )
        rows.extend(
            audit_endpoint(
                model, record, endpoint_id, device, args.repeats
            )
        )

    public_rows = [
        {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        for row in rows
    ]

    csv_path = args.output_dir / "model_clean_transition_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(public_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(public_rows)

    summary = summarize(rows, model.edge_threshold)
    report = {
        "checkpoint": str(CHECKPOINT),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "seed": SEED,
        "endpoint_count": len(selected),
        "selection_histogram": selection_histogram,
        "repeats": args.repeats,
        "times": list(TIMES),
        "edge_threshold": model.edge_threshold,
        "summary": summary,
    }

    json_path = args.output_dir / "summary.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 150)
    print("Facebook Beta-H1 model-clean one-step reverse audit")
    print("=" * 150)
    print(
        f"{'t':>6} {'clean AUROC':>12} {'clean AUPRC':>12} "
        f"{'clean pos':>11} {'clean neg':>11} "
        f"{'state AUROC':>12} {'state AUPRC':>12} "
        f"{'state >=.5':>12} {'state MAE':>11} "
        f"{'clean MAE':>11} {'nonfinite':>10}"
    )
    print("-" * 150)

    for time_key in sorted(summary, key=float, reverse=True):
        item = summary[time_key]
        print(
            f"{float(time_key):6.2f} "
            f"{item['clean_auroc']:12.4f} "
            f"{item['clean_auprc']:12.4f} "
            f"{item['clean_positive_probability_mean']:11.4f} "
            f"{item['clean_negative_probability_mean']:11.4f} "
            f"{item['model_state_auroc']:12.4f} "
            f"{item['model_state_auprc']:12.4f} "
            f"{item['model_state_positive_rate_at_0_5']:12.4f} "
            f"{item['model_oracle_state_mae']:11.5f} "
            f"{item['clean_unit_mae_to_true']:11.5f} "
            f"{item['nonfinite_count']:10d}"
        )

    print("\nCSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
