from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data2/wenb/ylu/HiGGs-GBD/HiGGs-main")
DGD = ROOT / "dgd"
DIAGNOSTICS = ROOT / "tools" / "diagnostics"

sys.path.insert(0, str(DGD))
sys.path.insert(0, str(DIAGNOSTICS))

import cora_beta_h1_terminal_reverse_ablation as terminal

from diffusion.beta_diffusion_utils import (
    apply_graph_masks,
    forward_beta_sample,
    shifted_clean_to_unit_probability,
)

base = terminal.base

CHECKPOINT = ROOT / "saved_models/cora/cora_h1.ckpt"
OUTPUT_DIR = (
    ROOT / "higgs/outputs"
    / "cora_beta_h1_loss_gradient_by_time_epoch649_seed20260810"
)

SEED = 20260810
TIMES = (1.00, 0.75, 0.50, 0.25, 0.10, 0.01)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_endpoints():
    selected, histogram = base.extract_validation_endpoints()
    chosen = []
    seen = set()

    for record in selected:
        condition = int(record["condition_y"])
        if condition in seen:
            continue
        chosen.append(record)
        seen.add(condition)

    if seen != set(range(7)):
        raise RuntimeError(
            f"Expected conditions 0..6, found {sorted(seen)}"
        )

    return chosen, histogram


def gradient_norm(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    squared = torch.zeros(
        (), dtype=torch.float64, device=loss.device
    )

    finite = True
    element_count = 0

    for gradient in gradients:
        if gradient is None:
            continue
        finite = finite and bool(
            torch.isfinite(gradient).all()
        )
        squared = squared + gradient.double().pow(2).sum()
        element_count += gradient.numel()

    return (
        float(torch.sqrt(squared).detach().cpu()),
        bool(finite),
        element_count,
    )


def component_result(
    model,
    predicted_E,
    true_E,
    eta_E,
    alpha_t,
    alpha_s,
    mask,
    parameters,
    class_name,
    probability,
):
    count = int(mask.sum().detach().cpu())

    if count == 0:
        return None

    sampling, correction, minimum, maximum, diagnostics = (
        model.loss_module._component(
            predicted_E,
            true_E,
            eta_E,
            alpha_t.view(-1, 1, 1, 1),
            alpha_s.view(-1, 1, 1, 1),
            mask,
        )
    )

    weighted = (
        model.loss_module.sampling_weight * sampling
        + model.loss_module.correction_weight * correction
    )
    scaled = model.loss_module.lambda_edge * weighted

    grad_norm, grad_finite, gradient_elements = gradient_norm(
        scaled, parameters, retain_graph=True
    )

    expanded = mask.unsqueeze(-1).expand_as(probability)
    selected_probability = probability[expanded]

    return {
        "edge_class": class_name,
        "mask_count": count,
        "sampling_loss": float(
            sampling.detach().cpu()
        ),
        "correction_loss": float(
            correction.detach().cpu()
        ),
        "weighted_edge_loss": float(
            weighted.detach().cpu()
        ),
        "lambda_scaled_loss": float(
            scaled.detach().cpu()
        ),
        "gradient_l2_norm": grad_norm,
        "gradient_finite": grad_finite,
        "gradient_parameter_element_count": (
            gradient_elements
        ),
        "prediction_probability_min": float(
            selected_probability.min().detach().cpu()
        ),
        "prediction_probability_mean": float(
            selected_probability.mean().detach().cpu()
        ),
        "prediction_probability_max": float(
            selected_probability.max().detach().cpu()
        ),
        "prediction_probability_std": float(
            selected_probability.std(
                unbiased=False
            ).detach().cpu()
        ),
        "beta_parameter_min": float(
            minimum.detach().cpu()
        ),
        "beta_parameter_max": float(
            maximum.detach().cpu()
        ),
        "invalid_parameter_count": int(
            diagnostics[
                "invalid_parameter_count"
            ].detach().cpu()
        ),
        "nan_kl_count": int(
            diagnostics["nan_kl_count"].detach().cpu()
        ),
        "inf_kl_count": int(
            diagnostics["inf_kl_count"].detach().cpu()
        ),
        "materially_negative_count": int(
            diagnostics[
                "materially_negative_count"
            ].detach().cpu()
        ),
    }


def audit_endpoint(model, record, endpoint_id, device):
    values = terminal.build_real_tensors(
        record, model, device
    )

    n = values["n"]
    node_mask = values["node_mask"]
    X0 = values["X0"]
    E0 = values["E0"]
    condition_y = values["condition_y"]

    eta_X, eta_E = model._eta_tensors(
        1, n, X0
    )

    edge_label = (
        values["E_discrete"][0, :, :, 0].bool()
    )
    upper = torch.triu(
        torch.ones(
            (n, n), dtype=torch.bool, device=device
        ),
        diagonal=1,
    ).unsqueeze(0)

    positive_mask = upper & edge_label.unsqueeze(0)
    negative_mask = upper & ~edge_label.unsqueeze(0)
    all_mask = upper

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    rows = []

    for time_index, t_value in enumerate(TIMES):
        s_value = max(
            t_value - 1.0 / model.T, 0.0
        )

        t = torch.tensor(
            [[t_value]], dtype=X0.dtype, device=device
        )
        s = torch.tensor(
            [[s_value]], dtype=X0.dtype, device=device
        )

        alpha_t = model._alpha(t)
        alpha_s = model._alpha(s)

        seed_everything(
            SEED + endpoint_id * 1000 + time_index
        )

        X_t, X_diag = forward_beta_sample(
            X0,
            alpha_t.view(-1, 1, 1),
            eta_X,
            model.beta_eps,
            model.input_space,
        )
        E_t, E_diag = forward_beta_sample(
            E0,
            alpha_t.view(-1, 1, 1, 1),
            eta_E,
            model.beta_eps,
            model.input_space,
        )
        X_t, E_t = apply_graph_masks(
            X_t, E_t, node_mask
        )

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

        edge_probability = (
            shifted_clean_to_unit_probability(
                pred_E0,
                model.edge_scale,
                model.edge_shift,
            )
        )

        current_results = []

        for class_name, mask in (
            ("all", all_mask),
            ("present", positive_mask),
            ("absent", negative_mask),
        ):
            result = component_result(
                model,
                pred_E0,
                E0,
                eta_E,
                alpha_t,
                alpha_s,
                mask,
                parameters,
                class_name,
                edge_probability,
            )

            if result is not None:
                current_results.append(result)

        all_count = int(all_mask.sum().detach().cpu())

        for result in current_results:
            class_fraction = (
                result["mask_count"] / all_count
            )

            result.update({
                "endpoint_id": endpoint_id,
                "condition_y": int(
                    record["condition_y"]
                ),
                "nodes": n,
                "true_edges": int(
                    positive_mask.sum().detach().cpu()
                ),
                "candidate_edges": all_count,
                "positive_rate": (
                    int(
                        positive_mask.sum()
                        .detach().cpu()
                    ) / all_count
                ),
                "t": t_value,
                "s": s_value,
                "alpha_t": float(
                    alpha_t[0, 0].detach().cpu()
                ),
                "alpha_s": float(
                    alpha_s[0, 0].detach().cpu()
                ),
                "class_fraction": class_fraction,
                "effective_loss_contribution": (
                    class_fraction
                    * result["lambda_scaled_loss"]
                ),
                "effective_gradient_norm_proxy": (
                    class_fraction
                    * result["gradient_l2_norm"]
                ),
                "forward_gamma_clamp_count": int(
                    X_diag.gamma_clamp_count.cpu()
                    + E_diag.gamma_clamp_count.cpu()
                ),
            })
            rows.append(result)

        del pred_X0, pred_E0, X_t, E_t

    return rows


def weighted_mean(rows, key, weight_key="mask_count"):
    denominator = sum(
        row[weight_key] for row in rows
    )
    if denominator == 0:
        return None
    return sum(
        row[key] * row[weight_key]
        for row in rows
    ) / denominator


def summarize(rows):
    summary = {}

    for t_value in TIMES:
        by_time = [
            row for row in rows
            if row["t"] == t_value
        ]

        result = {}

        for class_name in ("all", "present", "absent"):
            current = [
                row for row in by_time
                if row["edge_class"] == class_name
            ]

            result[class_name] = {
                "row_count": len(current),
                "mask_count": sum(
                    row["mask_count"]
                    for row in current
                ),
                "prediction_probability_mean": (
                    weighted_mean(
                        current,
                        "prediction_probability_mean",
                    )
                ),
                "sampling_loss_mean": weighted_mean(
                    current, "sampling_loss"
                ),
                "correction_loss_mean": weighted_mean(
                    current, "correction_loss"
                ),
                "weighted_edge_loss_mean": (
                    weighted_mean(
                        current,
                        "weighted_edge_loss",
                    )
                ),
                "lambda_scaled_loss_mean": (
                    weighted_mean(
                        current,
                        "lambda_scaled_loss",
                    )
                ),
                "gradient_l2_norm_mean": float(
                    np.mean([
                        row["gradient_l2_norm"]
                        for row in current
                    ])
                ),
                "effective_gradient_norm_proxy_mean": float(
                    np.mean([
                        row[
                            "effective_gradient_norm_proxy"
                        ]
                        for row in current
                    ])
                ),
                "invalid_parameter_count": sum(
                    row["invalid_parameter_count"]
                    for row in current
                ),
                "nonfinite_kl_count": sum(
                    row["nan_kl_count"]
                    + row["inf_kl_count"]
                    for row in current
                ),
            }

        positive_rows = [
            row for row in by_time
            if row["edge_class"] == "present"
        ]
        negative_rows = [
            row for row in by_time
            if row["edge_class"] == "absent"
        ]

        positive_contribution = sum(
            row["lambda_scaled_loss"]
            * row["mask_count"]
            for row in positive_rows
        )
        negative_contribution = sum(
            row["lambda_scaled_loss"]
            * row["mask_count"]
            for row in negative_rows
        )
        total_contribution = (
            positive_contribution
            + negative_contribution
        )

        result["contribution"] = {
            "positive_rate": (
                sum(
                    row["mask_count"]
                    for row in positive_rows
                )
                / (
                    sum(
                        row["mask_count"]
                        for row in positive_rows
                    )
                    + sum(
                        row["mask_count"]
                        for row in negative_rows
                    )
                )
            ),
            "positive_loss_contribution_share": (
                positive_contribution
                / total_contribution
                if total_contribution else None
            ),
            "negative_loss_contribution_share": (
                negative_contribution
                / total_contribution
                if total_contribution else None
            ),
            "positive_to_negative_loss_mean_ratio": (
                result["present"][
                    "lambda_scaled_loss_mean"
                ]
                / result["absent"][
                    "lambda_scaled_loss_mean"
                ]
                if result["absent"][
                    "lambda_scaled_loss_mean"
                ] else None
            ),
            "positive_to_negative_gradient_norm_ratio": (
                result["present"][
                    "gradient_l2_norm_mean"
                ]
                / result["absent"][
                    "gradient_l2_norm_mean"
                ]
                if result["absent"][
                    "gradient_l2_norm_mean"
                ] else None
            ),
        }

        summary[str(t_value)] = result

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    selected, selection_histogram = select_endpoints()

    checkpoint = base._load_checkpoint_cpu(
        CHECKPOINT
    )
    model = (
        base.ConditionalGraphBetaDiffusionH1
        .load_from_checkpoint(
            str(CHECKPOINT),
            map_location="cpu",
        )
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
    print("times:", TIMES)
    print(
        "sampling weight:",
        model.loss_module.sampling_weight,
    )
    print(
        "correction weight:",
        model.loss_module.correction_weight,
    )
    print(
        "lambda edge:",
        model.loss_module.lambda_edge,
    )

    rows = []

    for endpoint_id, record in enumerate(selected):
        print(
            f"Auditing endpoint {endpoint_id + 1}/"
            f"{len(selected)}, y={record['condition_y']}, "
            f"n={record['graph'].number_of_nodes()}"
        )
        rows.extend(
            audit_endpoint(
                model, record, endpoint_id, device
            )
        )

    summary = summarize(rows)

    csv_path = OUTPUT_DIR / "loss_gradient_rows.csv"
    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "checkpoint": str(CHECKPOINT),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "seed": SEED,
        "endpoint_count": len(selected),
        "selection_histogram": selection_histogram,
        "times": list(TIMES),
        "sampling_weight": (
            model.loss_module.sampling_weight
        ),
        "correction_weight": (
            model.loss_module.correction_weight
        ),
        "lambda_edge": (
            model.loss_module.lambda_edge
        ),
        "summary": summary,
    }

    json_path = OUTPUT_DIR / "summary.json"
    json_path.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 210)
    print("Beta-H1 edge loss and gradient by time")
    print("=" * 210)
    print(
        f"{'t':>6} {'pos rate':>9} "
        f"{'pos prob':>10} {'neg prob':>10} "
        f"{'pos loss':>12} {'neg loss':>12} "
        f"{'pos/neg loss':>13} "
        f"{'pos share':>10} {'neg share':>10} "
        f"{'pos grad':>12} {'neg grad':>12} "
        f"{'grad ratio':>11}"
    )
    print("-" * 210)

    for t_value in TIMES:
        item = summary[str(t_value)]
        pos = item["present"]
        neg = item["absent"]
        contribution = item["contribution"]

        print(
            f"{t_value:6.2f} "
            f"{contribution['positive_rate']:9.5f} "
            f"{pos['prediction_probability_mean']:10.5f} "
            f"{neg['prediction_probability_mean']:10.5f} "
            f"{pos['lambda_scaled_loss_mean']:12.6g} "
            f"{neg['lambda_scaled_loss_mean']:12.6g} "
            f"{contribution['positive_to_negative_loss_mean_ratio']:13.5f} "
            f"{contribution['positive_loss_contribution_share']:10.5f} "
            f"{contribution['negative_loss_contribution_share']:10.5f} "
            f"{pos['gradient_l2_norm_mean']:12.6g} "
            f"{neg['gradient_l2_norm_mean']:12.6g} "
            f"{contribution['positive_to_negative_gradient_norm_ratio']:11.5f}"
        )

    print("\nCSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
