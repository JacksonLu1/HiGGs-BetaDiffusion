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

import networkx as nx
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader

import utils
import higgs.sampling as sampling

from dgd.diffusion.beta_diffusion_utils import (
    apply_graph_masks,
    fixed_h2_condition,
    shifted_clean_to_unit_probability,
    unit_probability_to_shifted_state,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260805,
    )

    return parser.parse_args()


def load_graphs(path: Path):
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(obj, tuple) or len(obj) < 2:
        raise TypeError(
            f"Expected PyG collated tuple, got {type(obj)}"
        )

    collated = obj[0]
    slices = obj[1]

    if isinstance(collated, dict):
        collated = Data.from_dict(collated)

    if not isinstance(slices, dict):
        raise TypeError("Second tuple element is not slices")

    for key in ("x", "edge_index", "edge_attr", "y"):
        if key in slices:
            graph_count = len(slices[key]) - 1
            break
    else:
        raise RuntimeError(
            f"Cannot determine graph count from {list(slices)}"
        )

    graphs = []

    for index in range(graph_count):
        graphs.append(
            separate(
                cls=collated.__class__,
                batch=collated,
                idx=index,
                slice_dict=slices,
                inc_dict=None,
                decrement=False,
            )
        )

    return graphs


def safe_div(numerator: float, denominator: float):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def graph_stats(adjacency: np.ndarray):
    graph = nx.from_numpy_array(
        adjacency.astype(np.uint8)
    )

    return {
        "edges": int(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "connected": int(nx.is_connected(graph)),
        "isolates": int(nx.number_of_isolates(graph)),
        "components": int(
            nx.number_connected_components(graph)
        ),
        "clustering": float(
            nx.average_clustering(graph)
        ),
        "transitivity": float(
            nx.transitivity(graph)
        ),
    }


def main():
    args = parse_args()

    for path in (args.config, args.test_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = OmegaConf.load(args.config)

    cfg_h2 = cfg.copy()
    cfg_h2["dataset"]["h"] = 2
    cfg_h2["model"]["extra_features"] = "all"

    cfg_h2, model = sampling.get_samplers(cfg_h2)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        and int(cfg.general.gpus) > 0
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    graphs = load_graphs(args.test_file)

    loader = DataLoader(
        graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    t_values = [
        0.01,
        0.10,
        0.25,
        0.40,
        0.50,
        0.75,
        1.00,
    ]

    rows = []

    print("=" * 120)
    print("Cora H2 teacher-forced denoising audit")
    print("=" * 120)
    print("Checkpoint :", cfg.general.h2_model)
    print("Test file  :", args.test_file)
    print("Graphs     :", len(graphs))
    print("Device     :", device)
    print("Repeats    :", args.repeats)
    print("t values   :", t_values)
    print("=" * 120)

    with torch.inference_mode():
        for t_value in t_values:
            alpha_value = float(
                model._alpha(
                    torch.tensor(
                        [[t_value]],
                        device=device,
                    )
                ).item()
            )

            print()
            print(
                f"t={t_value:.2f}, "
                f"alpha={alpha_value:.9f}"
            )

            graph_offset = 0

            for repeat in range(args.repeats):
                for batch in loader:
                    batch = batch.to(device)

                    dense_data, node_mask = utils.to_dense(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                    )

                    dense_data = dense_data.mask(node_mask)

                    X_discrete = dense_data.X.float()
                    E_discrete = dense_data.E.float()

                    if E_discrete.size(-1) < 2:
                        raise RuntimeError(
                            "Expected non-edge and edge channels"
                        )

                    true_edge = (
                        E_discrete[..., 1] > 0.5
                    )

                    X0 = unit_probability_to_shifted_state(
                        X_discrete,
                        model.node_scale,
                        model.node_shift,
                    )

                    E0 = unit_probability_to_shifted_state(
                        E_discrete[..., 1:],
                        model.edge_scale,
                        model.edge_shift,
                    )

                    X0, E0 = apply_graph_masks(
                        X0,
                        E0,
                        node_mask,
                    )

                    eta_X, eta_E = model._eta_tensors(
                        E_discrete[..., 1:],
                        node_mask,
                    )

                    batch_size = int(batch.num_graphs)

                    t = torch.full(
                        (batch_size, 1),
                        t_value,
                        device=device,
                        dtype=X0.dtype,
                    )

                    alpha_t = model._alpha(t)

                    X_t, E_t, _ = model._forward_noise(
                        X0,
                        E0,
                        alpha_t,
                        eta_X,
                        eta_E,
                        node_mask,
                    )

                    condition_y = fixed_h2_condition(
                        model.ydim_output
                    ).to(
                        device=device,
                        dtype=X0.dtype,
                    ).expand(batch_size, -1)

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

                    pred_X_probability = (
                        shifted_clean_to_unit_probability(
                            pred_X0,
                            model.node_scale,
                            model.node_shift,
                        )
                    )

                    pred_E_probability = (
                        shifted_clean_to_unit_probability(
                            pred_E0,
                            model.edge_scale,
                            model.edge_shift,
                        )[..., 0]
                    )

                    pred_edge = (
                        pred_E_probability
                        > model.edge_threshold
                    )

                    pred_node = (
                        pred_X_probability.argmax(dim=-1)
                    )
                    true_node = X_discrete.argmax(dim=-1)

                    for local_index in range(batch_size):
                        n = int(
                            node_mask[local_index]
                            .sum()
                            .item()
                        )

                        upper = torch.triu(
                            torch.ones(
                                (n, n),
                                device=device,
                                dtype=torch.bool,
                            ),
                            diagonal=1,
                        )

                        true_upper = true_edge[
                            local_index,
                            :n,
                            :n,
                        ][upper]

                        pred_upper = pred_edge[
                            local_index,
                            :n,
                            :n,
                        ][upper]

                        probability_upper = (
                            pred_E_probability[
                                local_index,
                                :n,
                                :n,
                            ][upper]
                        )

                        tp = int(
                            (true_upper & pred_upper)
                            .sum()
                            .item()
                        )
                        fp = int(
                            ((~true_upper) & pred_upper)
                            .sum()
                            .item()
                        )
                        fn = int(
                            (true_upper & (~pred_upper))
                            .sum()
                            .item()
                        )
                        tn = int(
                            ((~true_upper) & (~pred_upper))
                            .sum()
                            .item()
                        )

                        precision = safe_div(
                            tp,
                            tp + fp,
                        )
                        recall = safe_div(
                            tp,
                            tp + fn,
                        )
                        f1 = safe_div(
                            2 * precision * recall,
                            precision + recall,
                        )

                        adjacency = np.zeros(
                            (n, n),
                            dtype=np.uint8,
                        )

                        upper_np = np.triu_indices(
                            n,
                            k=1,
                        )

                        adjacency[upper_np] = (
                            pred_upper.detach()
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )

                        adjacency = (
                            adjacency + adjacency.T
                        )

                        structural = graph_stats(
                            adjacency
                        )

                        true_edges = int(
                            true_upper.sum().item()
                        )

                        node_accuracy = float(
                            (
                                pred_node[
                                    local_index,
                                    :n,
                                ]
                                == true_node[
                                    local_index,
                                    :n,
                                ]
                            )
                            .float()
                            .mean()
                            .item()
                        )

                        rows.append(
                            {
                                "t": t_value,
                                "alpha": alpha_value,
                                "repeat": repeat,
                                "graph_index": (
                                    graph_offset
                                    + local_index
                                ),
                                "nodes": n,

                                "true_edges": true_edges,
                                "pred_edges": (
                                    structural["edges"]
                                ),

                                "tp": tp,
                                "fp": fp,
                                "fn": fn,
                                "tn": tn,

                                "precision": precision,
                                "recall": recall,
                                "f1": f1,
                                "node_accuracy": (
                                    node_accuracy
                                ),

                                "pred_density": (
                                    structural["density"]
                                ),
                                "pred_connected": (
                                    structural["connected"]
                                ),
                                "pred_isolates": (
                                    structural["isolates"]
                                ),
                                "pred_components": (
                                    structural["components"]
                                ),
                                "pred_clustering": (
                                    structural["clustering"]
                                ),
                                "pred_transitivity": (
                                    structural[
                                        "transitivity"
                                    ]
                                ),

                                "probability_mean": float(
                                    probability_upper
                                    .mean()
                                    .item()
                                ),
                                "probability_mid_fraction": float(
                                    (
                                        (
                                            probability_upper
                                            > 0.01
                                        )
                                        & (
                                            probability_upper
                                            < 0.99
                                        )
                                    )
                                    .float()
                                    .mean()
                                    .item()
                                ),
                                "probability_near_zero_fraction": float(
                                    (
                                        probability_upper
                                        <= 0.001
                                    )
                                    .float()
                                    .mean()
                                    .item()
                                ),
                                "probability_near_one_fraction": float(
                                    (
                                        probability_upper
                                        >= 0.99
                                    )
                                    .float()
                                    .mean()
                                    .item()
                                ),
                            }
                        )

                    graph_offset += batch_size

                graph_offset = 0

            current = pd.DataFrame(
                [
                    row
                    for row in rows
                    if row["t"] == t_value
                ]
            )

            tp = current["tp"].sum()
            fp = current["fp"].sum()
            fn = current["fn"].sum()

            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(
                2 * precision * recall,
                precision + recall,
            )

            print(
                f"  edge F1={f1:.6f}, "
                f"precision={precision:.6f}, "
                f"recall={recall:.6f}, "
                f"pred_edges_mean="
                f"{current['pred_edges'].mean():.3f}, "
                f"connected_rate="
                f"{current['pred_connected'].mean():.3f}, "
                f"isolate_graph_rate="
                f"{(current['pred_isolates'] > 0).mean():.3f}, "
                f"node_acc="
                f"{current['node_accuracy'].mean():.6f}"
            )

    frame = pd.DataFrame(rows)

    summary_rows = []

    for t_value, group in frame.groupby("t"):
        tp = float(group["tp"].sum())
        fp = float(group["fp"].sum())
        fn = float(group["fn"].sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(
            2 * precision * recall,
            precision + recall,
        )

        summary_rows.append(
            {
                "t": float(t_value),
                "alpha": float(
                    group["alpha"].iloc[0]
                ),

                "edge_precision_micro": precision,
                "edge_recall_micro": recall,
                "edge_f1_micro": f1,

                "edge_f1_graph_mean": float(
                    group["f1"].mean()
                ),
                "node_accuracy_mean": float(
                    group["node_accuracy"].mean()
                ),

                "true_edges_mean": float(
                    group["true_edges"].mean()
                ),
                "pred_edges_mean": float(
                    group["pred_edges"].mean()
                ),
                "pred_density_mean": float(
                    group["pred_density"].mean()
                ),

                "pred_connected_rate": float(
                    group["pred_connected"].mean()
                ),
                "pred_isolate_graph_rate": float(
                    (
                        group["pred_isolates"] > 0
                    ).mean()
                ),
                "pred_components_mean": float(
                    group["pred_components"].mean()
                ),

                "pred_clustering_mean": float(
                    group["pred_clustering"].mean()
                ),
                "pred_transitivity_mean": float(
                    group[
                        "pred_transitivity"
                    ].mean()
                ),

                "probability_mid_fraction": float(
                    group[
                        "probability_mid_fraction"
                    ].mean()
                ),
                "probability_near_zero_fraction": float(
                    group[
                        "probability_near_zero_fraction"
                    ].mean()
                ),
                "probability_near_one_fraction": float(
                    group[
                        "probability_near_one_fraction"
                    ].mean()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        "t"
    )

    per_graph_path = (
        args.output_dir
        / "cora_h2_teacher_forced_per_graph.csv"
    )
    summary_path = (
        args.output_dir
        / "cora_h2_teacher_forced_summary.csv"
    )
    json_path = (
        args.output_dir
        / "cora_h2_teacher_forced_summary.json"
    )

    frame.to_csv(per_graph_path, index=False)
    summary.to_csv(summary_path, index=False)

    json_path.write_text(
        json.dumps(
            summary.to_dict(orient="records"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 150)
    print("Teacher-forced H2 summary")
    print("=" * 150)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("Per graph:", per_graph_path)
    print("Summary  :", summary_path)
    print("JSON     :", json_path)
    print("=" * 150)


if __name__ == "__main__":
    main()
