from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data2/wenb/ylu/HiGGs-GBD/HiGGs-main")
DGD = ROOT / "dgd"
DIAGNOSTICS = ROOT / "tools" / "diagnostics"

sys.path.insert(0, str(DGD))
sys.path.insert(0, str(DIAGNOSTICS))

import cora_beta_h1_teacher_forced as teacher

from diffusion.beta_diffusion_utils import (
    apply_graph_masks,
    forward_beta_sample,
    reverse_beta_step,
    shifted_clean_to_unit_probability,
    unit_probability_to_shifted_state,
)

base = teacher.base

CHECKPOINT = ROOT / "saved_models/cora/cora_h1.ckpt"
OUTPUT_DIR = (
    ROOT / "higgs/outputs"
    / "cora_beta_h1_terminal_reverse_ablation_epoch649_seed20260808"
)

SEED = 20260808
MODES = (
    "current_zero_base",
    "empirical_prior_base",
    "true_g0_terminal",
)

# step=0 is t=1; step=99 is t=0.01 for T=100.
CAPTURE_FRACTIONS = (1.00, 0.75, 0.50, 0.25, 0.10, 0.01)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_div(a, b):
    return float(a / b) if b else None


def mean(values):
    return float(np.mean(values)) if values else None


def build_real_tensors(record, model, device):
    graph = record["graph"]
    classes = [int(value) for value in record["classes"]]
    condition_class = int(record["condition_y"])
    n = graph.number_of_nodes()

    node_mask = torch.ones(
        (1, n), dtype=torch.bool, device=device
    )

    X_discrete = F.one_hot(
        torch.tensor(classes, dtype=torch.long, device=device),
        num_classes=model.Xdim_output,
    ).float().unsqueeze(0)

    E_discrete = torch.zeros(
        (1, n, n, 1), dtype=torch.float, device=device
    )
    for left, right in graph.edges():
        E_discrete[0, left, right, 0] = 1.0
        E_discrete[0, right, left, 0] = 1.0

    condition_y = F.one_hot(
        torch.tensor([condition_class], device=device),
        num_classes=model.ydim_output,
    ).float()

    X0 = unit_probability_to_shifted_state(
        X_discrete, model.node_scale, model.node_shift
    )
    E0 = unit_probability_to_shifted_state(
        E_discrete, model.edge_scale, model.edge_shift
    )
    X0, E0 = apply_graph_masks(X0, E0, node_mask)

    return {
        "graph": graph,
        "classes": classes,
        "condition_class": condition_class,
        "n": n,
        "node_mask": node_mask,
        "condition_y": condition_y,
        "X_discrete": X_discrete,
        "E_discrete": E_discrete,
        "X0": X0,
        "E0": E0,
    }


def empirical_prior_clean(model, n, node_mask, device, seed):
    node_prior = model.node_channel_prior.detach().float().cpu()
    node_prior = node_prior.clamp_min(0)
    node_prior = node_prior / node_prior.sum()

    edge_prior = float(
        model.edge_channel_prior.detach().reshape(-1)[0].cpu()
    )
    edge_prior = min(max(edge_prior, 0.0), 1.0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    node_types = torch.multinomial(
        node_prior,
        num_samples=n,
        replacement=True,
        generator=generator,
    )
    X_unit = F.one_hot(
        node_types,
        num_classes=model.Xdim_output,
    ).float().unsqueeze(0).to(device)

    random_matrix = torch.rand(
        (n, n), generator=generator
    )
    upper = torch.triu(
        random_matrix < edge_prior, diagonal=1
    )
    adjacency = upper | upper.transpose(0, 1)

    E_unit = adjacency.float().view(1, n, n, 1).to(device)

    X_clean = unit_probability_to_shifted_state(
        X_unit, model.node_scale, model.node_shift
    )
    E_clean = unit_probability_to_shifted_state(
        E_unit, model.edge_scale, model.edge_shift
    )
    return apply_graph_masks(
        X_clean, E_clean, node_mask
    )


def initialise_terminal(
    mode,
    values,
    model,
    eta_X,
    eta_E,
    alpha_T,
    seed,
):
    n = values["n"]
    node_mask = values["node_mask"]
    device = values["X0"].device

    if mode == "current_zero_base":
        clean_X = torch.full(
            (1, n, model.Xdim_output),
            model.node_shift,
            device=device,
        )
        clean_E = torch.full(
            (1, n, n, 1),
            model.edge_shift,
            device=device,
        )
        clean_X, clean_E = apply_graph_masks(
            clean_X, clean_E, node_mask
        )

    elif mode == "empirical_prior_base":
        clean_X, clean_E = empirical_prior_clean(
            model, n, node_mask, device, seed + 73000
        )

    elif mode == "true_g0_terminal":
        clean_X = values["X0"]
        clean_E = values["E0"]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Keep the terminal Gamma draw seed identical between modes.
    seed_everything(seed)

    if model.diffuse_X:
        X_state, X_diag = forward_beta_sample(
            clean_X,
            alpha_T.view(-1, 1, 1),
            eta_X,
            model.beta_eps,
            model.input_space,
        )
    else:
        X_state = model._shifted_to_state(clean_X)
        X_diag = None

    E_state, E_diag = forward_beta_sample(
        clean_E,
        alpha_T.view(-1, 1, 1, 1),
        eta_E,
        model.beta_eps,
        model.input_space,
    )
    X_state, E_state = apply_graph_masks(
        X_state, E_state, node_mask
    )

    gamma_count = int(E_diag.gamma_clamp_count.cpu())
    if X_diag is not None:
        gamma_count += int(X_diag.gamma_clamp_count.cpu())

    return X_state, E_state, gamma_count


def graph_from_prediction(edge_probability, threshold):
    n = edge_probability.size(0)
    upper_mask = torch.triu(
        torch.ones(
            (n, n),
            dtype=torch.bool,
            device=edge_probability.device,
        ),
        diagonal=1,
    )
    upper_indices = upper_mask.nonzero(as_tuple=False)
    predicted = edge_probability[upper_mask] > threshold

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for index in predicted.nonzero(as_tuple=False).flatten():
        left, right = upper_indices[index].tolist()
        graph.add_edge(left, right)

    return graph, upper_mask


def state_statistics(model, state, scale, shift, upper_mask):
    probability = model._state_to_unit(state, scale, shift)
    if probability.ndim == 4:
        probability = probability[0, :, :, 0][upper_mask]
    else:
        probability = probability.reshape(-1)

    return {
        "min": float(probability.min().cpu()),
        "mean": float(probability.mean().cpu()),
        "max": float(probability.max().cpu()),
        "std": float(probability.std(unbiased=False).cpu()),
        "ge_threshold_rate": float(
            (probability > model.edge_threshold)
            .float().mean().cpu()
        ),
        "nonfinite_count": int(
            (~torch.isfinite(probability)).sum().cpu()
        ),
    }


def prediction_snapshot(
    mode,
    record_id,
    time_value,
    values,
    model,
    X_state,
    E_state,
    pred_X0,
    pred_E0,
):
    node_probability = shifted_clean_to_unit_probability(
        pred_X0, model.node_scale, model.node_shift
    )
    edge_probability = shifted_clean_to_unit_probability(
        pred_E0, model.edge_scale, model.edge_shift
    )[0, :, :, 0]

    generated, upper_mask = graph_from_prediction(
        edge_probability, model.edge_threshold
    )

    labels = (
        values["E_discrete"][0, :, :, 0][upper_mask]
        .long().cpu()
    )
    scores = edge_probability[upper_mask].float().cpu()

    state_stats = state_statistics(
        model,
        E_state,
        model.edge_scale,
        model.edge_shift,
        upper_mask,
    )

    predicted_classes = (
        node_probability[0].argmax(dim=-1).cpu()
    )
    generated_stats = base.graph_stats(generated)

    return {
        "mode": mode,
        "record_id": record_id,
        "time": float(time_value),
        "nodes": values["n"],
        "condition_y": values["condition_class"],
        "state_probability_min": state_stats["min"],
        "state_probability_mean": state_stats["mean"],
        "state_probability_max": state_stats["max"],
        "state_probability_std": state_stats["std"],
        "state_ge_threshold_rate": (
            state_stats["ge_threshold_rate"]
        ),
        "state_nonfinite_count": (
            state_stats["nonfinite_count"]
        ),
        "predicted_edge_probability_min": float(
            scores.min()
        ),
        "predicted_edge_probability_mean": float(
            scores.mean()
        ),
        "predicted_edge_probability_max": float(
            scores.max()
        ),
        "predicted_edge_probability_std": float(
            scores.std(unbiased=False)
        ),
        "predicted_edge_rate": float(
            (scores > model.edge_threshold)
            .float().mean()
        ),
        "predicted_edges": generated.number_of_edges(),
        "predicted_components": (
            generated_stats["components"]
        ),
        "predicted_clustering": (
            generated_stats["clustering"]
        ),
        "predicted_mean_degree": (
            generated_stats["mean_degree"]
        ),
        "predicted_max_degree": (
            generated_stats["max_degree"]
        ),
        "node_accuracy_against_reference": float(
            (
                predicted_classes
                == torch.tensor(values["classes"])
            ).float().mean()
        ),
        "condition_majority_match": int(
            base.majority(predicted_classes.tolist())
            == values["condition_class"]
        ),
        "prediction_nonfinite_count": int(
            (~torch.isfinite(node_probability)).sum().cpu()
            + (~torch.isfinite(edge_probability)).sum().cpu()
        ),
        "_labels": labels,
        "_scores": scores,
    }


@torch.inference_mode()
def run_reverse(model, record, mode, record_id, device):
    values = build_real_tensors(
        record, model, device
    )

    n = values["n"]
    node_mask = values["node_mask"]
    condition_y = values["condition_y"]
    original_condition = condition_y.clone()

    eta_X, eta_E = model._eta_tensors(
        1, n, values["X0"]
    )

    alpha_T = model._alpha(
        torch.ones((1, 1), device=device)
    )

    experiment_seed = SEED + record_id * 1000

    X_state, E_state, gamma_count = initialise_terminal(
        mode,
        values,
        model,
        eta_X,
        eta_E,
        alpha_T,
        experiment_seed,
    )

    # Reset before reverse so every mode starts the reverse loop
    # from the same RNG seed. Gamma rejection sampling can still
    # consume different internal draws after states diverge.
    seed_everything(experiment_seed + 1)

    times = torch.linspace(
        1.0, 0.0, model.T + 1, device=device
    )

    capture_steps = {
        int(round((1.0 - fraction) * model.T)): fraction
        for fraction in CAPTURE_FRACTIONS
    }

    trajectory = []
    pred_X0 = values["X0"]
    pred_E0 = values["E0"]

    for step in range(model.T):
        t = times[step].expand(1, 1)
        s = times[step + 1].expand(1, 1)
        alpha_t = model._alpha(t)
        alpha_s = model._alpha(s)

        pred_X0, pred_E0 = model._predict_clean(
            X_state,
            E_state,
            condition_y,
            t,
            alpha_t,
            eta_X,
            eta_E,
            node_mask,
        )

        if step in capture_steps:
            trajectory.append(
                prediction_snapshot(
                    mode,
                    record_id,
                    capture_steps[step],
                    values,
                    model,
                    X_state,
                    E_state,
                    pred_X0,
                    pred_E0,
                )
            )

        if model.diffuse_X:
            X_state, X_diag = reverse_beta_step(
                X_state,
                pred_X0,
                alpha_t.view(-1, 1, 1),
                alpha_s.view(-1, 1, 1),
                eta_X,
                model.beta_eps,
                model.input_space,
            )
            gamma_count += int(
                X_diag.gamma_clamp_count.cpu()
            )

        E_state, E_diag = reverse_beta_step(
            E_state,
            pred_E0,
            alpha_t.view(-1, 1, 1, 1),
            alpha_s.view(-1, 1, 1, 1),
            eta_E,
            model.beta_eps,
            model.input_space,
        )
        gamma_count += int(
            E_diag.gamma_clamp_count.cpu()
        )

        X_state, E_state = apply_graph_masks(
            X_state, E_state, node_mask
        )

        if not torch.equal(condition_y, original_condition):
            raise AssertionError(
                "Condition y changed during reverse chain"
            )

    node_probability = shifted_clean_to_unit_probability(
        pred_X0, model.node_scale, model.node_shift
    )
    edge_probability = shifted_clean_to_unit_probability(
        pred_E0, model.edge_scale, model.edge_shift
    )[0, :, :, 0]

    final_graph, upper_mask = graph_from_prediction(
        edge_probability, model.edge_threshold
    )

    real_graph = values["graph"]
    real_stats = base.graph_stats(real_graph)
    final_stats = base.graph_stats(final_graph)

    components = list(nx.connected_components(final_graph))
    largest = max((len(c) for c in components), default=0)

    predicted_classes = (
        node_probability[0].argmax(dim=-1).cpu()
    )

    final_labels = (
        values["E_discrete"][0, :, :, 0][upper_mask]
        .long().cpu()
    )
    final_scores = (
        edge_probability[upper_mask].float().cpu()
    )

    final_row = {
        "mode": mode,
        "record_id": record_id,
        "condition_y": values["condition_class"],
        "nodes": n,
        "real_edges": real_graph.number_of_edges(),
        "generated_edges": final_graph.number_of_edges(),
        "edge_ratio": safe_div(
            final_graph.number_of_edges(),
            real_graph.number_of_edges(),
        ),
        "real_clustering": real_stats["clustering"],
        "generated_clustering": (
            final_stats["clustering"]
        ),
        "real_mean_degree": real_stats["mean_degree"],
        "generated_mean_degree": (
            final_stats["mean_degree"]
        ),
        "real_max_degree": real_stats["max_degree"],
        "generated_max_degree": (
            final_stats["max_degree"]
        ),
        "generated_components": (
            final_stats["components"]
        ),
        "generated_connected": int(
            final_stats["components"] == 1
        ),
        "lcc_retention": safe_div(largest, n),
        "node_accuracy_against_reference": float(
            (
                predicted_classes
                == torch.tensor(values["classes"])
            ).float().mean()
        ),
        "condition_majority_match": int(
            base.majority(predicted_classes.tolist())
            == values["condition_class"]
        ),
        "probability_min": float(final_scores.min()),
        "probability_mean": float(final_scores.mean()),
        "probability_max": float(final_scores.max()),
        "probability_std": float(
            final_scores.std(unbiased=False)
        ),
        "probability_ge_threshold_rate": float(
            (final_scores > model.edge_threshold)
            .float().mean()
        ),
        "gamma_clamp_count": gamma_count,
        "nonfinite_count": int(
            (~torch.isfinite(node_probability)).sum().cpu()
            + (~torch.isfinite(edge_probability)).sum().cpu()
            + (~torch.isfinite(X_state)).sum().cpu()
            + (~torch.isfinite(E_state)).sum().cpu()
        ),
        "_labels": final_labels,
        "_scores": final_scores,
    }

    return final_row, trajectory


def summarize_final(rows, threshold):
    metric = teacher.binary_metrics(
        [row["_labels"] for row in rows],
        [row["_scores"] for row in rows],
        threshold,
    )

    metric.update({
        "graph_count": len(rows),
        "generated_edges_mean": mean(
            [row["generated_edges"] for row in rows]
        ),
        "edge_ratio_mean": mean(
            [row["edge_ratio"] for row in rows]
        ),
        "connected_rate": mean(
            [row["generated_connected"] for row in rows]
        ),
        "components_mean": mean(
            [row["generated_components"] for row in rows]
        ),
        "lcc_retention_mean": mean(
            [row["lcc_retention"] for row in rows]
        ),
        "real_clustering_mean": mean(
            [row["real_clustering"] for row in rows]
        ),
        "generated_clustering_mean": mean(
            [row["generated_clustering"] for row in rows]
        ),
        "real_max_degree_mean": mean(
            [row["real_max_degree"] for row in rows]
        ),
        "generated_max_degree_mean": mean(
            [row["generated_max_degree"] for row in rows]
        ),
        "node_accuracy_mean": mean(
            [
                row["node_accuracy_against_reference"]
                for row in rows
            ]
        ),
        "condition_majority_match_rate": mean(
            [
                row["condition_majority_match"]
                for row in rows
            ]
        ),
        "gamma_clamp_count": sum(
            row["gamma_clamp_count"] for row in rows
        ),
        "nonfinite_count": sum(
            row["nonfinite_count"] for row in rows
        ),
    })
    return metric


def summarize_trajectory(rows, threshold):
    metric = teacher.binary_metrics(
        [row["_labels"] for row in rows],
        [row["_scores"] for row in rows],
        threshold,
    )
    metric.update({
        "graph_count": len(rows),
        "state_probability_mean": mean(
            [row["state_probability_mean"] for row in rows]
        ),
        "state_probability_std_mean": mean(
            [row["state_probability_std"] for row in rows]
        ),
        "state_ge_threshold_rate_mean": mean(
            [row["state_ge_threshold_rate"] for row in rows]
        ),
        "predicted_components_mean": mean(
            [row["predicted_components"] for row in rows]
        ),
        "predicted_clustering_mean": mean(
            [row["predicted_clustering"] for row in rows]
        ),
        "predicted_edge_rate_mean": mean(
            [row["predicted_edge_rate"] for row in rows]
        ),
        "condition_majority_match_rate": mean(
            [row["condition_majority_match"] for row in rows]
        ),
        "nonfinite_count": sum(
            row["state_nonfinite_count"]
            + row["prediction_nonfinite_count"]
            for row in rows
        ),
    })
    return metric


def public_rows(rows):
    return [
        {
            key: value for key, value in row.items()
            if not key.startswith("_")
        }
        for row in rows
    ]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    selected, selection_histogram = (
        base.extract_validation_endpoints()
    )

    checkpoint = base._load_checkpoint_cpu(CHECKPOINT)
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

    node_prior = (
        model.node_channel_prior.detach().float().cpu()
    )
    node_prior = node_prior / node_prior.sum()
    edge_prior = float(
        model.edge_channel_prior.detach()
        .reshape(-1)[0].cpu()
    )

    print("checkpoint:", CHECKPOINT)
    print("epoch:", checkpoint.get("epoch"))
    print("global_step:", checkpoint.get("global_step"))
    print("device:", device)
    print("selected endpoints:", len(selected))
    print("modes:", MODES)
    print("node prior:", node_prior.tolist())
    print("edge prior:", edge_prior)
    print("edge threshold:", model.edge_threshold)

    final_rows = []
    trajectory_rows = []

    for mode in MODES:
        print(f"\nRunning mode: {mode}")
        for record_id, record in enumerate(selected):
            final_row, trajectory = run_reverse(
                model,
                record,
                mode,
                record_id,
                device,
            )
            final_rows.append(final_row)
            trajectory_rows.extend(trajectory)

            print(
                f"\r{mode}: {record_id + 1}/{len(selected)}",
                end="",
                flush=True,
            )
        print()

    final_summary = {}
    for mode in MODES:
        current = [
            row for row in final_rows
            if row["mode"] == mode
        ]
        final_summary[mode] = summarize_final(
            current, model.edge_threshold
        )

    trajectory_summary = defaultdict(dict)
    for mode in MODES:
        for time_value in CAPTURE_FRACTIONS:
            current = [
                row for row in trajectory_rows
                if row["mode"] == mode
                and abs(row["time"] - time_value) < 1e-9
            ]
            trajectory_summary[mode][str(time_value)] = (
                summarize_trajectory(
                    current, model.edge_threshold
                )
            )

    final_public = public_rows(final_rows)
    trajectory_public = public_rows(trajectory_rows)

    final_csv = OUTPUT_DIR / "final_results.csv"
    with final_csv.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(final_public[0].keys()),
        )
        writer.writeheader()
        writer.writerows(final_public)

    trajectory_csv = OUTPUT_DIR / "reverse_trajectory.csv"
    with trajectory_csv.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(trajectory_public[0].keys()),
        )
        writer.writeheader()
        writer.writerows(trajectory_public)

    report = {
        "checkpoint": str(CHECKPOINT),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "seed": SEED,
        "selected_count": len(selected),
        "selection_histogram": selection_histogram,
        "modes": list(MODES),
        "capture_times": list(CAPTURE_FRACTIONS),
        "node_prior": node_prior.tolist(),
        "edge_prior": edge_prior,
        "edge_threshold": model.edge_threshold,
        "final_summary": final_summary,
        "trajectory_summary": dict(trajectory_summary),
        "interpretation_note": (
            "Exact edge AUROC/AUPRC for current_zero_base and "
            "empirical_prior_base compares an independent generative "
            "sample to one reference adjacency and is diagnostic only, "
            "not a formal generative metric."
        ),
    }

    json_path = OUTPUT_DIR / "summary.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 150)
    print("Final reverse result by terminal initialisation")
    print("=" * 150)
    print(
        f"{'mode':>24} {'edges':>9} {'ratio':>8} "
        f"{'connected':>10} {'components':>11} "
        f"{'LCC':>8} {'clustering':>11} "
        f"{'maxdeg':>9} {'node acc':>9} "
        f"{'cond match':>11} {'nonfinite':>10}"
    )
    print("-" * 150)

    for mode in MODES:
        item = final_summary[mode]
        print(
            f"{mode:>24} "
            f"{item['generated_edges_mean']:9.2f} "
            f"{item['edge_ratio_mean']:8.3f} "
            f"{item['connected_rate']:10.3f} "
            f"{item['components_mean']:11.3f} "
            f"{item['lcc_retention_mean']:8.3f} "
            f"{item['generated_clustering_mean']:11.4f} "
            f"{item['generated_max_degree_mean']:9.2f} "
            f"{item['node_accuracy_mean']:9.3f} "
            f"{item['condition_majority_match_rate']:11.3f} "
            f"{item['nonfinite_count']:10d}"
        )

    print("\n" + "=" * 170)
    print("Reverse trajectory")
    print("=" * 170)
    print(
        f"{'mode':>24} {'t':>6} "
        f"{'state mean':>11} {'state >.5':>10} "
        f"{'pred mean':>10} {'AUROC':>8} "
        f"{'AUPRC':>8} {'edge rate':>10} "
        f"{'components':>11} {'clustering':>11} "
        f"{'cond match':>11}"
    )
    print("-" * 170)

    for mode in MODES:
        for time_value in CAPTURE_FRACTIONS:
            item = trajectory_summary[mode][str(time_value)]
            print(
                f"{mode:>24} "
                f"{time_value:6.2f} "
                f"{item['state_probability_mean']:11.5f} "
                f"{item['state_ge_threshold_rate_mean']:10.5f} "
                f"{item['probability_mean']:10.5f} "
                f"{item['auroc']:8.4f} "
                f"{item['auprc']:8.4f} "
                f"{item['predicted_positive_rate']:10.5f} "
                f"{item['predicted_components_mean']:11.3f} "
                f"{item['predicted_clustering_mean']:11.4f} "
                f"{item['condition_majority_match_rate']:11.3f}"
            )

    print("\nFinal CSV:", final_csv)
    print("Trajectory CSV:", trajectory_csv)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
