from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data2/wenb/ylu/HiGGs-GBD/HiGGs-main")
DIAGNOSTICS = ROOT / "tools" / "diagnostics"
DGD = ROOT / "dgd"

sys.path.insert(0, str(DIAGNOSTICS))
sys.path.insert(0, str(DGD))

import facebook_beta_h1_fixed_validation as base

from diffusion.beta_diffusion_utils import (
    apply_graph_masks,
    forward_beta_sample,
    shifted_clean_to_unit_probability,
    unit_probability_to_shifted_state,
)

CHECKPOINT = ROOT / "outputs/2026-07-30/10-58-34/checkpoints/fb_beta_h1_r10_e500/last.ckpt"
OUTPUT_DIR = (
    ROOT
    / "higgs/outputs"
    / "facebook_beta_h1_teacher_forced_epoch499_seed20260827"
)

SEED = 20260827
TIMES = [0.01, 0.25, 0.50, 0.75, 1.00]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_div(a, b):
    return float(a / b) if b else None


def mean(values):
    return float(np.mean(values)) if values else None


def std(values):
    return float(np.std(values)) if values else None


def binary_metrics(labels, scores, threshold):
    labels = torch.cat(labels).double()
    scores = torch.cat(scores).double()

    predicted = scores > threshold
    positive = labels == 1
    negative = labels == 0

    tp = int((predicted & positive).sum())
    fp = int((predicted & negative).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & negative).sum())

    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]

    cumulative_tp = torch.cumsum(sorted_labels, dim=0)
    cumulative_fp = torch.cumsum(1.0 - sorted_labels, dim=0)

    positives = float(positive.sum())
    negatives = float(negative.sum())

    if positives and negatives:
        tpr = torch.cat((
            torch.zeros(1, dtype=torch.double),
            cumulative_tp / positives,
        ))
        fpr = torch.cat((
            torch.zeros(1, dtype=torch.double),
            cumulative_fp / negatives,
        ))
        auroc = float(torch.trapz(tpr, fpr))
    else:
        auroc = None

    if positives:
        precision_curve = cumulative_tp / torch.arange(
            1, len(sorted_labels) + 1, dtype=torch.double
        )
        auprc = float(
            (precision_curve * sorted_labels).sum() / positives
        )
    else:
        auprc = None

    positive_scores = scores[positive]
    negative_scores = scores[negative]

    return {
        "candidate_count": int(labels.numel()),
        "true_positive_count": int(positive.sum()),
        "true_positive_rate": float(labels.mean()),
        "predicted_positive_count": int(predicted.sum()),
        "predicted_positive_rate": float(predicted.double().mean()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_at_threshold": safe_div(tp, tp + fp),
        "recall_at_threshold": safe_div(tp, tp + fn),
        "auroc": auroc,
        "auprc": auprc,
        "positive_probability_mean": (
            float(positive_scores.mean()) if positive_scores.numel() else None
        ),
        "positive_probability_std": (
            float(positive_scores.std(unbiased=False))
            if positive_scores.numel() else None
        ),
        "negative_probability_mean": (
            float(negative_scores.mean()) if negative_scores.numel() else None
        ),
        "negative_probability_std": (
            float(negative_scores.std(unbiased=False))
            if negative_scores.numel() else None
        ),
        "probability_min": float(scores.min()),
        "probability_mean": float(scores.mean()),
        "probability_max": float(scores.max()),
        "probability_std": float(scores.std(unbiased=False)),
    }


def size_group(n):
    if n <= 50:
        return "n<=50"
    if n <= 100:
        return "51-100"
    if n <= 150:
        return "101-150"
    return "n>150"


@torch.inference_mode()
def evaluate_endpoint(model, record, t_value, record_id, device):
    graph = record["graph"]
    classes = [int(value) for value in record["classes"]]
    condition_class = int(record["condition_y"])
    n = graph.number_of_nodes()

    node_mask = torch.ones((1, n), dtype=torch.bool, device=device)

    X_discrete = F.one_hot(
        torch.tensor(classes, dtype=torch.long, device=device),
        num_classes=model.Xdim_output,
    ).float().unsqueeze(0)

    adjacency = torch.zeros(
        (1, n, n, 1), dtype=torch.float, device=device
    )
    for left, right in graph.edges():
        adjacency[0, left, right, 0] = 1.0
        adjacency[0, right, left, 0] = 1.0

    condition_y = F.one_hot(
        torch.tensor([condition_class], device=device),
        num_classes=model.ydim_output,
    ).float()

    X0 = unit_probability_to_shifted_state(
        X_discrete, model.node_scale, model.node_shift
    )
    E0 = unit_probability_to_shifted_state(
        adjacency, model.edge_scale, model.edge_shift
    )
    X0, E0 = apply_graph_masks(X0, E0, node_mask)

    eta_X, eta_E = model._eta_tensors(1, n, X0)

    t = torch.tensor([[t_value]], dtype=X0.dtype, device=device)
    alpha_t = model._alpha(t)

    seed_everything(SEED + record_id * 1000 + int(t_value * 100))

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
    X_t, E_t = apply_graph_masks(X_t, E_t, node_mask)

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

    node_probability = shifted_clean_to_unit_probability(
        pred_X0, model.node_scale, model.node_shift
    )
    edge_probability = shifted_clean_to_unit_probability(
        pred_E0, model.edge_scale, model.edge_shift
    )[0, :, :, 0]

    noisy_edge_probability = model._state_to_unit(
        E_t, model.edge_scale, model.edge_shift
    )[0, :, :, 0]

    upper = torch.triu(
        torch.ones((n, n), dtype=torch.bool, device=device),
        diagonal=1,
    )
    labels = adjacency[0, :, :, 0][upper].long().cpu()
    scores = edge_probability[upper].float().cpu()

    predicted_upper = edge_probability[upper] > model.edge_threshold
    upper_indices = upper.nonzero(as_tuple=False)

    generated = nx.Graph()
    generated.add_nodes_from(range(n))
    for index in predicted_upper.nonzero(as_tuple=False).flatten():
        left, right = upper_indices[index].tolist()
        generated.add_edge(left, right)

    predicted_classes = node_probability[0].argmax(dim=-1).cpu()
    real_stats = base.graph_stats(graph)
    predicted_stats = base.graph_stats(generated)

    noisy_upper = noisy_edge_probability[upper]
    asymmetry = float(
        (edge_probability - edge_probability.transpose(0, 1))
        .abs().max().cpu()
    )

    return {
        "record_id": record_id,
        "condition_y": condition_class,
        "time": t_value,
        "size_group": size_group(n),
        "nodes": n,
        "real_edges": graph.number_of_edges(),
        "predicted_edges": generated.number_of_edges(),
        "edge_ratio": safe_div(
            generated.number_of_edges(), graph.number_of_edges()
        ),
        "real_clustering": real_stats["clustering"],
        "predicted_clustering": predicted_stats["clustering"],
        "real_max_degree": real_stats["max_degree"],
        "predicted_max_degree": predicted_stats["max_degree"],
        "predicted_components": predicted_stats["components"],
        "predicted_connected": int(predicted_stats["components"] == 1),
        "node_accuracy": float(
            (predicted_classes == torch.tensor(classes)).float().mean()
        ),
        "condition_majority_match": int(
            base.majority(predicted_classes.tolist()) == condition_class
        ),
        "noisy_edge_ge_threshold_rate": float(
            (noisy_upper > model.edge_threshold).float().mean().cpu()
        ),
        "probability_asymmetry_max": asymmetry,
        "nonfinite_count": int(
            (~torch.isfinite(node_probability)).sum().cpu()
            + (~torch.isfinite(edge_probability)).sum().cpu()
        ),
        "gamma_clamp_count": int(
            X_diag.gamma_clamp_count.cpu()
            + E_diag.gamma_clamp_count.cpu()
        ),
        "_labels": labels,
        "_scores": scores,
    }


def summarize_time(rows, threshold):
    metrics = binary_metrics(
        [row["_labels"] for row in rows],
        [row["_scores"] for row in rows],
        threshold,
    )
    metrics.update({
        "graph_count": len(rows),
        "connected_rate": mean(
            [row["predicted_connected"] for row in rows]
        ),
        "components_mean": mean(
            [row["predicted_components"] for row in rows]
        ),
        "node_accuracy_mean": mean(
            [row["node_accuracy"] for row in rows]
        ),
        "condition_majority_match_rate": mean(
            [row["condition_majority_match"] for row in rows]
        ),
        "real_clustering_mean": mean(
            [row["real_clustering"] for row in rows]
        ),
        "predicted_clustering_mean": mean(
            [row["predicted_clustering"] for row in rows]
        ),
        "edge_ratio_mean": mean(
            [row["edge_ratio"] for row in rows]
        ),
        "noisy_edge_ge_threshold_rate_mean": mean(
            [row["noisy_edge_ge_threshold_rate"] for row in rows]
        ),
        "probability_asymmetry_max": max(
            row["probability_asymmetry_max"] for row in rows
        ),
        "nonfinite_count": sum(
            row["nonfinite_count"] for row in rows
        ),
        "gamma_clamp_count": sum(
            row["gamma_clamp_count"] for row in rows
        ),
    })
    return metrics


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    selected, selection_histogram = base.extract_validation_endpoints()

    checkpoint = base._load_checkpoint_cpu(CHECKPOINT)
    model = base.ConditionalGraphBetaDiffusionH1.load_from_checkpoint(
        str(CHECKPOINT),
        map_location="cpu",
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print("checkpoint:", CHECKPOINT)
    print("epoch:", checkpoint.get("epoch"))
    print("global_step:", checkpoint.get("global_step"))
    print("device:", device)
    print("selected endpoints:", len(selected))
    print("threshold:", model.edge_threshold)
    print("times:", TIMES)

    rows = []
    for t_value in TIMES:
        print(f"\nEvaluating teacher-forced t={t_value:.2f}")
        for record_id, record in enumerate(selected):
            rows.append(
                evaluate_endpoint(
                    model, record, t_value, record_id, device
                )
            )

    summaries = {}
    for t_value in TIMES:
        current = [
            row for row in rows if row["time"] == t_value
        ]
        summaries[str(t_value)] = summarize_time(
            current, model.edge_threshold
        )

    size_summaries = {}
    for t_value in TIMES:
        size_summaries[str(t_value)] = {}
        for group in ("n<=50", "51-100", "101-150", "n>150"):
            current = [
                row for row in rows
                if row["time"] == t_value
                and row["size_group"] == group
            ]
            if current:
                size_summaries[str(t_value)][group] = {
                    "graph_count": len(current),
                    "connected_rate": mean(
                        [row["predicted_connected"] for row in current]
                    ),
                    "components_mean": mean(
                        [row["predicted_components"] for row in current]
                    ),
                    "node_accuracy_mean": mean(
                        [row["node_accuracy"] for row in current]
                    ),
                    "condition_match_rate": mean(
                        [row["condition_majority_match"] for row in current]
                    ),
                    "real_clustering_mean": mean(
                        [row["real_clustering"] for row in current]
                    ),
                    "predicted_clustering_mean": mean(
                        [row["predicted_clustering"] for row in current]
                    ),
                    "edge_ratio_mean": mean(
                        [row["edge_ratio"] for row in current]
                    ),
                }

    public_rows = []
    for row in rows:
        public_rows.append({
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        })

    csv_path = OUTPUT_DIR / "per_endpoint_teacher_forced.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(public_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(public_rows)

    report = {
        "checkpoint": str(CHECKPOINT),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "seed": SEED,
        "times": TIMES,
        "selected_count": len(selected),
        "selection_histogram": selection_histogram,
        "edge_threshold": model.edge_threshold,
        "summaries": summaries,
        "size_summaries": size_summaries,
    }

    json_path = OUTPUT_DIR / "summary.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 130)
    print("Facebook Beta-H1 teacher-forced single-step diagnostic")
    print("=" * 130)
    print(
        f"{'t':>6} {'AUROC':>9} {'AUPRC':>9} "
        f"{'pos prob':>10} {'neg prob':>10} "
        f"{'edge rate':>10} {'components':>11} "
        f"{'connected':>10} {'node acc':>9} "
        f"{'cond match':>11} {'clustering':>11}"
    )
    print("-" * 130)

    for t_value in TIMES:
        item = summaries[str(t_value)]
        print(
            f"{t_value:6.2f} "
            f"{item['auroc']:9.4f} "
            f"{item['auprc']:9.4f} "
            f"{item['positive_probability_mean']:10.4f} "
            f"{item['negative_probability_mean']:10.4f} "
            f"{item['predicted_positive_rate']:10.4f} "
            f"{item['components_mean']:11.3f} "
            f"{item['connected_rate']:10.3f} "
            f"{item['node_accuracy_mean']:9.3f} "
            f"{item['condition_majority_match_rate']:11.3f} "
            f"{item['predicted_clustering_mean']:11.4f}"
        )

    print("\nCSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
