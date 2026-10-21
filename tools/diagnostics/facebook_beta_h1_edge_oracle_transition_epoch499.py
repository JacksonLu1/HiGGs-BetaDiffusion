from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path("/data2/wenb/ylu/HiGGs-GBD/HiGGs-main")
DGD = ROOT / "dgd"
DIAGNOSTICS = ROOT / "tools" / "diagnostics"

sys.path.insert(0, str(DGD))
sys.path.insert(0, str(DIAGNOSTICS))

import facebook_beta_h1_terminal_reverse_ablation_epoch499 as terminal

from diffusion.beta_diffusion_utils import (
    forward_beta_sample,
    reverse_beta_step,
)

base = terminal.base

CHECKPOINT = ROOT / "outputs/2026-07-30/10-58-34/checkpoints/fb_beta_h1_r10_e500/last.ckpt"
OUTPUT_DIR = (
    ROOT / "higgs/outputs"
    / "facebook_beta_h1_edge_oracle_transition_epoch499_seed20260829"
)

SEED = 20260829
REPEATS = 32
TIMES = (1.00, 0.75, 0.50, 0.25, 0.10, 0.01)
MEAN_TOLERANCE = 0.005
MAX_KS_SAMPLES = 100000


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def raw_beta_probability(state, input_space):
    if input_space == "logit":
        return torch.sigmoid(state)
    if input_space == "original":
        return state
    raise ValueError(input_space)


def bounded_numpy(tensor, seed):
    values = tensor.detach().reshape(-1).float().cpu().numpy()
    if len(values) <= MAX_KS_SAMPLES:
        return values

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(values), size=MAX_KS_SAMPLES, replace=False
    )
    return values[indices]


def tensor_stats(tensor):
    values = tensor.detach().reshape(-1).double()
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "std": float(values.std(unbiased=False)),
        "p10": float(torch.quantile(values, 0.10)),
        "p50": float(torch.quantile(values, 0.50)),
        "p90": float(torch.quantile(values, 0.90)),
        "nonfinite_count": int(
            (~torch.isfinite(values)).sum()
        ),
    }


def select_endpoints():
    selected, histogram = base.extract_validation_endpoints()

    # One endpoint for every Facebook condition class: four total.
    chosen = []
    seen = set()

    for record in selected:
        condition = int(record["condition_y"])
        if condition in seen:
            continue
        chosen.append(record)
        seen.add(condition)

    if seen != set(range(4)):
        raise RuntimeError(
            f"Expected conditions 0..6, found {sorted(seen)}"
        )

    return chosen, histogram


@torch.inference_mode()
def audit_endpoint(model, record, endpoint_id, device):
    values = terminal.build_real_tensors(
        record, model, device
    )

    n = values["n"]
    E0 = values["E0"].expand(
        REPEATS, -1, -1, -1
    ).contiguous()

    eta_E = torch.full(
        (REPEATS, n, n, 1),
        model.eta_edge,
        dtype=E0.dtype,
        device=device,
    ).clamp_min(model.beta_eps)

    upper = torch.triu(
        torch.ones(
            (n, n), dtype=torch.bool, device=device
        ),
        diagonal=1,
    )

    labels = (
        values["E_discrete"][0, :, :, 0][upper]
        .bool()
    )
    clean_upper = values["E0"][0, :, :, 0][upper]

    distribution_rows = []
    step_rows = []

    for time_index, t_value in enumerate(TIMES):
        s_value = max(
            t_value - 1.0 / model.T, 0.0
        )

        t = torch.full(
            (REPEATS, 1),
            t_value,
            dtype=E0.dtype,
            device=device,
        )
        s = torch.full(
            (REPEATS, 1),
            s_value,
            dtype=E0.dtype,
            device=device,
        )

        alpha_t = model._alpha(t)
        alpha_s = model._alpha(s)

        # q(E_t | E_0)
        seed_everything(
            SEED + endpoint_id * 1000
            + time_index * 10
        )
        E_t, forward_t_diag = forward_beta_sample(
            E0,
            alpha_t.view(-1, 1, 1, 1),
            eta_E,
            model.beta_eps,
            model.input_space,
        )

        # Oracle reverse: predicted clean is exact true E0.
        E_oracle_s, reverse_diag = reverse_beta_step(
            E_t,
            E0,
            alpha_t.view(-1, 1, 1, 1),
            alpha_s.view(-1, 1, 1, 1),
            eta_E,
            model.beta_eps,
            model.input_space,
        )

        # Independent direct reference: q(E_s | E_0)
        seed_everything(
            SEED + endpoint_id * 1000
            + time_index * 10 + 1
        )
        E_direct_s, direct_s_diag = forward_beta_sample(
            E0,
            alpha_s.view(-1, 1, 1, 1),
            eta_E,
            model.beta_eps,
            model.input_space,
        )

        raw_t = raw_beta_probability(
            E_t, model.input_space
        )[:, :, :, 0][:, upper]

        raw_oracle_s = raw_beta_probability(
            E_oracle_s, model.input_space
        )[:, :, :, 0][:, upper]

        raw_direct_s = raw_beta_probability(
            E_direct_s, model.input_space
        )[:, :, :, 0][:, upper]

        monotonic_violations = int(
            (raw_oracle_s + 1e-7 < raw_t).sum().cpu()
        )

        element_count = REPEATS * int(upper.sum())
        gamma_denominator = 2 * element_count

        step_rows.append({
            "endpoint_id": endpoint_id,
            "condition_y": int(record["condition_y"]),
            "nodes": n,
            "true_edges": int(labels.sum()),
            "t": t_value,
            "s": s_value,
            "alpha_t": float(alpha_t[0, 0]),
            "alpha_s": float(alpha_s[0, 0]),
            "element_count": element_count,
            "forward_t_gamma_clamp_count": int(
                forward_t_diag.gamma_clamp_count.cpu()
            ),
            "forward_t_gamma_clamp_rate": (
                int(forward_t_diag.gamma_clamp_count.cpu())
                / gamma_denominator
            ),
            "reverse_gamma_clamp_count": int(
                reverse_diag.gamma_clamp_count.cpu()
            ),
            "reverse_gamma_clamp_rate": (
                int(reverse_diag.gamma_clamp_count.cpu())
                / gamma_denominator
            ),
            "direct_s_gamma_clamp_count": int(
                direct_s_diag.gamma_clamp_count.cpu()
            ),
            "direct_s_gamma_clamp_rate": (
                int(direct_s_diag.gamma_clamp_count.cpu())
                / gamma_denominator
            ),
            "monotonic_violation_count": monotonic_violations,
            "nonfinite_count": int(
                (~torch.isfinite(raw_t)).sum().cpu()
                + (~torch.isfinite(raw_oracle_s)).sum().cpu()
                + (~torch.isfinite(raw_direct_s)).sum().cpu()
            ),
        })

        for class_name, class_mask in (
            ("present", labels),
            ("absent", ~labels),
        ):
            oracle_values = raw_oracle_s[:, class_mask]
            direct_values = raw_direct_s[:, class_mask]
            current_values = raw_t[:, class_mask]

            oracle_stats = tensor_stats(oracle_values)
            direct_stats = tensor_stats(direct_values)
            current_stats = tensor_stats(current_values)

            clean_mean = float(
                clean_upper[class_mask].mean().cpu()
            )
            theoretical_mean = (
                float(alpha_s[0, 0].cpu()) * clean_mean
            )
            theoretical_std = math.sqrt(
                max(
                    theoretical_mean
                    * (1.0 - theoretical_mean)
                    / (model.eta_edge + 1.0),
                    0.0,
                )
            )

            oracle_np = bounded_numpy(
                oracle_values,
                SEED + endpoint_id * 100
                + time_index * 2,
            )
            direct_np = bounded_numpy(
                direct_values,
                SEED + endpoint_id * 100
                + time_index * 2 + 1,
            )

            ks = ks_2samp(
                oracle_np,
                direct_np,
                alternative="two-sided",
                mode="auto",
            )
            wasserstein = wasserstein_distance(
                oracle_np, direct_np
            )

            oracle_direct_delta = (
                oracle_stats["mean"]
                - direct_stats["mean"]
            )
            oracle_theory_delta = (
                oracle_stats["mean"]
                - theoretical_mean
            )
            direct_theory_delta = (
                direct_stats["mean"]
                - theoretical_mean
            )

            row_pass = (
                oracle_stats["nonfinite_count"] == 0
                and direct_stats["nonfinite_count"] == 0
                and abs(oracle_direct_delta)
                <= MEAN_TOLERANCE
                and abs(oracle_theory_delta)
                <= MEAN_TOLERANCE
                and abs(direct_theory_delta)
                <= MEAN_TOLERANCE
            )

            distribution_rows.append({
                "endpoint_id": endpoint_id,
                "condition_y": int(record["condition_y"]),
                "nodes": n,
                "true_edges": int(labels.sum()),
                "edge_class": class_name,
                "t": t_value,
                "s": s_value,
                "sample_count": oracle_stats["count"],
                "clean_shifted_mean": clean_mean,
                "theoretical_mean": theoretical_mean,
                "theoretical_std": theoretical_std,
                "current_mean": current_stats["mean"],
                "current_std": current_stats["std"],
                "oracle_mean": oracle_stats["mean"],
                "oracle_std": oracle_stats["std"],
                "direct_mean": direct_stats["mean"],
                "direct_std": direct_stats["std"],
                "oracle_direct_mean_delta": (
                    oracle_direct_delta
                ),
                "oracle_theory_mean_delta": (
                    oracle_theory_delta
                ),
                "direct_theory_mean_delta": (
                    direct_theory_delta
                ),
                "oracle_direct_std_delta": (
                    oracle_stats["std"]
                    - direct_stats["std"]
                ),
                "wasserstein_distance": float(wasserstein),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "oracle_min": oracle_stats["min"],
                "oracle_p10": oracle_stats["p10"],
                "oracle_p50": oracle_stats["p50"],
                "oracle_p90": oracle_stats["p90"],
                "oracle_max": oracle_stats["max"],
                "direct_min": direct_stats["min"],
                "direct_p10": direct_stats["p10"],
                "direct_p50": direct_stats["p50"],
                "direct_p90": direct_stats["p90"],
                "direct_max": direct_stats["max"],
                "row_pass": bool(row_pass),
            })

    return distribution_rows, step_rows


def weighted_mean(rows, key):
    total_weight = sum(
        row["sample_count"] for row in rows
    )
    if total_weight == 0:
        return None
    return sum(
        row[key] * row["sample_count"]
        for row in rows
    ) / total_weight


def summarize(distribution_rows, step_rows):
    summary = {}

    for t_value in TIMES:
        summary[str(t_value)] = {}

        for edge_class in ("present", "absent"):
            rows = [
                row for row in distribution_rows
                if row["t"] == t_value
                and row["edge_class"] == edge_class
            ]

            summary[str(t_value)][edge_class] = {
                "endpoint_count": len(rows),
                "sample_count": sum(
                    row["sample_count"] for row in rows
                ),
                "theoretical_mean": weighted_mean(
                    rows, "theoretical_mean"
                ),
                "oracle_mean": weighted_mean(
                    rows, "oracle_mean"
                ),
                "direct_mean": weighted_mean(
                    rows, "direct_mean"
                ),
                "oracle_std": weighted_mean(
                    rows, "oracle_std"
                ),
                "direct_std": weighted_mean(
                    rows, "direct_std"
                ),
                "oracle_direct_abs_mean_delta": abs(
                    weighted_mean(
                        rows,
                        "oracle_direct_mean_delta",
                    )
                ),
                "max_endpoint_oracle_direct_abs_delta": max(
                    abs(row["oracle_direct_mean_delta"])
                    for row in rows
                ),
                "max_endpoint_oracle_theory_abs_delta": max(
                    abs(row["oracle_theory_mean_delta"])
                    for row in rows
                ),
                "max_endpoint_direct_theory_abs_delta": max(
                    abs(row["direct_theory_mean_delta"])
                    for row in rows
                ),
                "wasserstein_mean": weighted_mean(
                    rows, "wasserstein_distance"
                ),
                "ks_statistic_mean": weighted_mean(
                    rows, "ks_statistic"
                ),
                "rows_passed": sum(
                    row["row_pass"] for row in rows
                ),
                "rows_total": len(rows),
            }

        steps = [
            row for row in step_rows
            if row["t"] == t_value
        ]
        summary[str(t_value)]["transition"] = {
            "monotonic_violation_count": sum(
                row["monotonic_violation_count"]
                for row in steps
            ),
            "nonfinite_count": sum(
                row["nonfinite_count"]
                for row in steps
            ),
            "forward_t_gamma_clamp_count": sum(
                row["forward_t_gamma_clamp_count"]
                for row in steps
            ),
            "reverse_gamma_clamp_count": sum(
                row["reverse_gamma_clamp_count"]
                for row in steps
            ),
            "direct_s_gamma_clamp_count": sum(
                row["direct_s_gamma_clamp_count"]
                for row in steps
            ),
            "forward_t_gamma_clamp_rate_mean": float(
                np.mean([
                    row["forward_t_gamma_clamp_rate"]
                    for row in steps
                ])
            ),
            "reverse_gamma_clamp_rate_mean": float(
                np.mean([
                    row["reverse_gamma_clamp_rate"]
                    for row in steps
                ])
            ),
            "direct_s_gamma_clamp_rate_mean": float(
                np.mean([
                    row["direct_s_gamma_clamp_rate"]
                    for row in steps
                ])
            ),
        }

    all_rows_pass = all(
        row["row_pass"]
        for row in distribution_rows
    )
    all_steps_valid = all(
        row["nonfinite_count"] == 0
        and row["monotonic_violation_count"] == 0
        for row in step_rows
    )

    return summary, bool(
        all_rows_pass and all_steps_valid
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    selected, selection_histogram = (
        select_endpoints()
    )

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
    print("repeats:", REPEATS)
    print("times:", TIMES)
    print("mean tolerance:", MEAN_TOLERANCE)

    distribution_rows = []
    step_rows = []

    for endpoint_id, record in enumerate(selected):
        print(
            f"Auditing endpoint {endpoint_id + 1}/"
            f"{len(selected)}, y={record['condition_y']}, "
            f"n={record['graph'].number_of_nodes()}"
        )
        rows, steps = audit_endpoint(
            model, record, endpoint_id, device
        )
        distribution_rows.extend(rows)
        step_rows.extend(steps)

    summary, overall_pass = summarize(
        distribution_rows, step_rows
    )

    distribution_csv = (
        OUTPUT_DIR / "distribution_results.csv"
    )
    with distribution_csv.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                distribution_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(distribution_rows)

    step_csv = OUTPUT_DIR / "transition_results.csv"
    with step_csv.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(step_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(step_rows)

    report = {
        "checkpoint": str(CHECKPOINT),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "seed": SEED,
        "endpoint_count": len(selected),
        "selection_histogram": selection_histogram,
        "repeats": REPEATS,
        "times": list(TIMES),
        "mean_tolerance": MEAN_TOLERANCE,
        "input_space": model.input_space,
        "eta_edge": model.eta_edge,
        "summary": summary,
        "overall_pass": overall_pass,
        "method": (
            "E_t is sampled by the real forward process. "
            "The reverse step receives exact true shifted E0, "
            "not a model prediction. Its E_s distribution is "
            "compared with an independent direct q(E_s|E0). "
            "Statistics use the raw underlying Beta probability, "
            "before inverse scale-shift."
        ),
    }

    json_path = OUTPUT_DIR / "summary.json"
    json_path.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 190)
    print("Beta-H1 edge oracle reverse-transition audit")
    print("=" * 190)
    print(
        f"{'t':>6} {'class':>9} "
        f"{'theory':>12} {'oracle':>12} "
        f"{'direct':>12} {'|O-D|':>12} "
        f"{'max |O-D|':>12} {'W-dist':>12} "
        f"{'KS':>9} {'pass':>9}"
    )
    print("-" * 190)

    for t_value in TIMES:
        for edge_class in ("present", "absent"):
            item = summary[str(t_value)][
                edge_class
            ]
            print(
                f"{t_value:6.2f} "
                f"{edge_class:>9} "
                f"{item['theoretical_mean']:12.8f} "
                f"{item['oracle_mean']:12.8f} "
                f"{item['direct_mean']:12.8f} "
                f"{item['oracle_direct_abs_mean_delta']:12.8f} "
                f"{item['max_endpoint_oracle_direct_abs_delta']:12.8f} "
                f"{item['wasserstein_mean']:12.8f} "
                f"{item['ks_statistic_mean']:9.5f} "
                f"{item['rows_passed']:2d}/"
                f"{item['rows_total']:<6d}"
            )

        transition = summary[str(t_value)][
            "transition"
        ]
        print(
            " " * 16
            + f"clamp rates: "
            + f"forward={transition['forward_t_gamma_clamp_rate_mean']:.6g}, "
            + f"reverse={transition['reverse_gamma_clamp_rate_mean']:.6g}, "
            + f"direct={transition['direct_s_gamma_clamp_rate_mean']:.6g}; "
            + f"monotonic violations="
            + f"{transition['monotonic_violation_count']}; "
            + f"nonfinite={transition['nonfinite_count']}"
        )

    print("\noverall_pass:", overall_pass)
    print("Distribution CSV:", distribution_csv)
    print("Transition CSV:", step_csv)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
