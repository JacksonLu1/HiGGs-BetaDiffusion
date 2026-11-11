#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

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

import higgs.sampling as sampling
from higgs.interface_ablation import seed_sampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--real-meta-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)

    parser.add_argument("--threshold-start", type=float, default=0.40)
    parser.add_argument("--threshold-stop", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.01)

    return parser.parse_args()


def graph_from_scores(
    scores: np.ndarray,
    threshold: float,
) -> nx.Graph:
    n = scores.shape[0]

    upper = np.triu(
        scores > threshold,
        k=1,
    )

    adjacency = np.logical_or(
        upper,
        upper.T,
    )

    graph = nx.from_numpy_array(
        adjacency.astype(np.uint8)
    )

    graph.add_nodes_from(range(n))
    return graph


def basic_stats(graph: nx.Graph) -> dict[str, Any]:
    degrees = np.asarray(
        [degree for _, degree in graph.degree()],
        dtype=float,
    )

    components = list(nx.connected_components(graph))

    return {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "components": int(len(components)),
        "isolates": int(nx.number_of_isolates(graph)),
        "connected": int(nx.is_connected(graph)),
        "min_degree": float(degrees.min()),
        "median_degree": float(np.median(degrees)),
        "mean_degree": float(degrees.mean()),
        "max_degree": float(degrees.max()),
        "density": float(nx.density(graph)),
    }


def repair_connectivity(
    graph: nx.Graph,
    scores: np.ndarray,
) -> tuple[nx.Graph, int, list[dict[str, Any]]]:
    """
    Repeatedly add the highest-score edge joining two different
    connected components.

    Adds exactly components-1 edges and never removes existing edges.
    """
    repaired = graph.copy()
    additions: list[dict[str, Any]] = []

    while not nx.is_connected(repaired):
        components = list(nx.connected_components(repaired))

        component_id = {}
        for index, component in enumerate(components):
            for node in component:
                component_id[node] = index

        best_score = -math.inf
        best_pair = None

        n = repaired.number_of_nodes()

        for left in range(n):
            for right in range(left + 1, n):
                if component_id[left] == component_id[right]:
                    continue

                score = float(scores[left, right])

                if score > best_score:
                    best_score = score
                    best_pair = (left, right)

        if best_pair is None:
            raise RuntimeError(
                "Could not find an edge joining different components"
            )

        left, right = best_pair
        repaired.add_edge(left, right)

        additions.append(
            {
                "left": int(left),
                "right": int(right),
                "score": float(best_score),
            }
        )

    return repaired, len(additions), additions


def describe(values: pd.Series) -> dict[str, float]:
    array = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    return {
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> int:
    args = parse_args()

    for path in (
        args.config,
        args.real_meta_csv,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds = np.arange(
        args.threshold_start,
        args.threshold_stop + args.threshold_step / 2,
        args.threshold_step,
    )

    thresholds = np.round(thresholds, 6)

    cfg = OmegaConf.load(args.config)

    cfg_h2 = cfg.copy()
    cfg_h2["dataset"]["h"] = 2
    cfg_h2["model"]["extra_features"] = "all"

    print("=" * 112)
    print("Loading Beta-H2 for continuous edge-score scan")
    print("=" * 112)
    print("Config       :", args.config)
    print(
        "Checkpoint   :",
        OmegaConf.select(cfg, "general.h2_model"),
    )
    print(
        "Checkpoint threshold:",
        OmegaConf.select(cfg, "model.edge_threshold"),
    )
    print(
        "Threshold scan:",
        float(thresholds[0]),
        "to",
        float(thresholds[-1]),
        "step",
        args.threshold_step,
    )

    cfg_h2, model = sampling.get_samplers(cfg_h2)

    use_cuda = (
        torch.cuda.is_available()
        and int(cfg.general.gpus) > 0
    )

    device = torch.device(
        "cuda" if use_cuda else "cpu"
    )

    model = model.to(device)
    model.eval()

    if model.training:
        raise RuntimeError("Model is not in eval mode")

    fixed_condition = sampling.fixed_h2_condition(
        model.ydim_output
    ).to(device)

    # sampling_batch_h2 imported discretize_graph into the model module.
    # Replace that module-global symbol temporarily to capture continuous
    # probabilities before thresholding.
    model_module = sys.modules[
        model.__class__.__module__
    ]

    original_discretize = model_module.discretize_graph
    capture: dict[str, torch.Tensor | float] = {}

    def capture_discretize(
        node_values: torch.Tensor,
        edge_values: torch.Tensor,
        node_mask: torch.Tensor,
        edge_threshold: float,
    ):
        capture["node_values"] = (
            node_values.detach().cpu().clone()
        )
        capture["edge_values"] = (
            edge_values.detach().cpu().clone()
        )
        capture["node_mask"] = (
            node_mask.detach().cpu().clone()
        )
        capture["original_threshold"] = float(
            edge_threshold
        )

        return original_discretize(
            node_values,
            edge_values,
            node_mask,
            edge_threshold,
        )

    model_module.discretize_graph = capture_discretize

    per_seed_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    node_score_rows: list[dict[str, Any]] = []

    try:
        for offset in range(args.count):
            seed = args.start_seed + offset
            started = time.perf_counter()

            seed_sampling(seed)
            capture.clear()

            with torch.inference_mode():
                predicted_graphs, _ = (
                    model.sampling_batch_h2(
                        batch_id=0,
                        batch_size=1,
                        keep_chain=0,
                        number_chain_steps=40,
                        num_nodes=None,
                        figures=False,
                        y=fixed_condition.clone(),
                        graph_number=seed,
                    )
                )

            if "edge_values" not in capture:
                raise RuntimeError(
                    f"No continuous edge scores captured for seed {seed}"
                )

            node_mask = capture["node_mask"][0]
            n_nodes = int(node_mask.sum().item())

            edge_values = capture["edge_values"]

            # Exact quantity used by discretize_graph:
            # max_edge, _ = edge_values.max(dim=-1)
            edge_scores = (
                edge_values.max(dim=-1).values[
                    0, :n_nodes, :n_nodes
                ]
                .numpy()
                .astype(float)
            )

            # Preserve the same upper-triangle convention used by
            # discretize_graph. No averaging of opposite directions.
            upper_scores = edge_scores[
                np.triu_indices(n_nodes, k=1)
            ]

            node_max_scores = []

            for node in range(n_nodes):
                incident = np.concatenate(
                    (
                        edge_scores[node, :node],
                        edge_scores[node, node + 1 :],
                    )
                )

                node_max = float(incident.max())
                node_max_scores.append(node_max)

                node_score_rows.append(
                    {
                        "seed": seed,
                        "node": node,
                        "node_max_incident_score": node_max,
                        "isolated_at_0_5": int(
                            node_max <= 0.5
                        ),
                    }
                )

            per_seed_rows.append(
                {
                    "seed": seed,
                    "nodes": n_nodes,
                    "candidate_edges": len(upper_scores),
                    "edge_score_min": float(
                        upper_scores.min()
                    ),
                    "edge_score_p05": float(
                        np.quantile(upper_scores, 0.05)
                    ),
                    "edge_score_p25": float(
                        np.quantile(upper_scores, 0.25)
                    ),
                    "edge_score_median": float(
                        np.median(upper_scores)
                    ),
                    "edge_score_mean": float(
                        upper_scores.mean()
                    ),
                    "edge_score_p75": float(
                        np.quantile(upper_scores, 0.75)
                    ),
                    "edge_score_p95": float(
                        np.quantile(upper_scores, 0.95)
                    ),
                    "edge_score_max": float(
                        upper_scores.max()
                    ),
                    "node_max_score_min": float(
                        min(node_max_scores)
                    ),
                    "node_max_score_median": float(
                        np.median(node_max_scores)
                    ),
                    "node_max_score_max": float(
                        max(node_max_scores)
                    ),
                    "elapsed_seconds": (
                        time.perf_counter() - started
                    ),
                }
            )

            for threshold in thresholds:
                raw_graph = graph_from_scores(
                    edge_scores,
                    float(threshold),
                )

                raw = basic_stats(raw_graph)

                (
                    repaired_graph,
                    repair_edges,
                    repair_additions,
                ) = repair_connectivity(
                    raw_graph,
                    edge_scores,
                )

                repaired = basic_stats(repaired_graph)

                minimum_repair_score = (
                    min(
                        item["score"]
                        for item in repair_additions
                    )
                    if repair_additions
                    else float("nan")
                )

                maximum_repair_score = (
                    max(
                        item["score"]
                        for item in repair_additions
                    )
                    if repair_additions
                    else float("nan")
                )

                threshold_rows.append(
                    {
                        "seed": seed,
                        "threshold": float(threshold),

                        "raw_nodes": raw["nodes"],
                        "raw_edges": raw["edges"],
                        "raw_components": raw["components"],
                        "raw_isolates": raw["isolates"],
                        "raw_connected": raw["connected"],
                        "raw_min_degree": raw["min_degree"],
                        "raw_median_degree": raw["median_degree"],
                        "raw_mean_degree": raw["mean_degree"],
                        "raw_max_degree": raw["max_degree"],
                        "raw_density": raw["density"],

                        "repair_edges_added": repair_edges,
                        "repair_min_added_score": (
                            minimum_repair_score
                        ),
                        "repair_max_added_score": (
                            maximum_repair_score
                        ),

                        "repaired_edges": repaired["edges"],
                        "repaired_components": (
                            repaired["components"]
                        ),
                        "repaired_isolates": (
                            repaired["isolates"]
                        ),
                        "repaired_connected": (
                            repaired["connected"]
                        ),
                        "repaired_min_degree": (
                            repaired["min_degree"]
                        ),
                        "repaired_density": (
                            repaired["density"]
                        ),
                    }
                )

            baseline_graph = graph_from_scores(
                edge_scores,
                0.5,
            )
            baseline = basic_stats(baseline_graph)

            print(
                f"[{offset + 1:03d}/{args.count:03d}] "
                f"seed={seed:3d} "
                f"nodes={n_nodes:2d} "
                f"t0.5={baseline['edges']:3d}E "
                f"components={baseline['components']} "
                f"isolates={baseline['isolates']} "
                f"min-node-max-score="
                f"{min(node_max_scores):.6f} "
                f"time={time.perf_counter() - started:.2f}s",
                flush=True,
            )

    finally:
        model_module.discretize_graph = (
            original_discretize
        )

    per_seed = pd.DataFrame(per_seed_rows)
    per_threshold = pd.DataFrame(threshold_rows)
    node_scores = pd.DataFrame(node_score_rows)

    real = pd.read_csv(args.real_meta_csv)

    real_edge_mean = float(real["edges"].mean())
    real_edge_median = float(real["edges"].median())
    real_density_mean = float(real["density"].mean())
    real_density_median = float(real["density"].median())

    grouped = (
        per_threshold.groupby("threshold")
        .agg(
            samples=("seed", "size"),

            raw_edge_mean=("raw_edges", "mean"),
            raw_edge_median=("raw_edges", "median"),
            raw_density_mean=("raw_density", "mean"),
            raw_density_median=("raw_density", "median"),
            raw_connected_rate=("raw_connected", "mean"),
            raw_isolate_graph_rate=(
                "raw_isolates",
                lambda values: float((values > 0).mean()),
            ),
            raw_isolates_mean=("raw_isolates", "mean"),
            raw_components_mean=("raw_components", "mean"),
            raw_min_degree_mean=("raw_min_degree", "mean"),

            repair_edges_mean=("repair_edges_added", "mean"),
            repair_edges_median=("repair_edges_added", "median"),
            repair_edges_p95=(
                "repair_edges_added",
                lambda values: float(
                    np.quantile(values, 0.95)
                ),
            ),

            repaired_edge_mean=("repaired_edges", "mean"),
            repaired_density_mean=(
                "repaired_density",
                "mean",
            ),
            repaired_connected_rate=(
                "repaired_connected",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["raw_edge_mean_error"] = (
        grouped["raw_edge_mean"] - real_edge_mean
    ).abs()

    grouped["raw_density_mean_error"] = (
        grouped["raw_density_mean"]
        - real_density_mean
    ).abs()

    grouped["repaired_edge_mean_error"] = (
        grouped["repaired_edge_mean"]
        - real_edge_mean
    ).abs()

    grouped["repaired_density_mean_error"] = (
        grouped["repaired_density_mean"]
        - real_density_mean
    ).abs()

    baseline_mismatches = []

    if (
        args.baseline_csv is not None
        and args.baseline_csv.is_file()
    ):
        baseline = pd.read_csv(args.baseline_csv)

        threshold_05 = per_threshold[
            np.isclose(
                per_threshold["threshold"],
                0.5,
            )
        ][
            [
                "seed",
                "raw_nodes",
                "raw_edges",
                "raw_components",
                "raw_isolates",
            ]
        ]

        expected = baseline[
            [
                "seed",
                "raw_nodes",
                "raw_edges",
                "raw_components",
                "raw_isolates",
            ]
        ]

        merged = threshold_05.merge(
            expected,
            on="seed",
            suffixes=("_scan", "_baseline"),
        )

        for column in (
            "raw_nodes",
            "raw_edges",
            "raw_components",
            "raw_isolates",
        ):
            mismatch = merged[
                merged[f"{column}_scan"]
                != merged[f"{column}_baseline"]
            ]

            if not mismatch.empty:
                baseline_mismatches.append(
                    {
                        "column": column,
                        "count": int(len(mismatch)),
                        "seeds": mismatch["seed"]
                        .astype(int)
                        .tolist(),
                    }
                )

    per_seed_path = (
        args.output_dir
        / "cora_h2_edge_score_per_seed.csv"
    )
    threshold_path = (
        args.output_dir
        / "cora_h2_threshold_per_seed.csv"
    )
    grouped_path = (
        args.output_dir
        / "cora_h2_threshold_summary.csv"
    )
    node_score_path = (
        args.output_dir
        / "cora_h2_node_max_scores.csv"
    )

    per_seed.to_csv(per_seed_path, index=False)
    per_threshold.to_csv(threshold_path, index=False)
    grouped.to_csv(grouped_path, index=False)
    node_scores.to_csv(node_score_path, index=False)

    best_raw_density = grouped.loc[
        grouped["raw_density_mean_error"].idxmin()
    ]

    best_raw_edges = grouped.loc[
        grouped["raw_edge_mean_error"].idxmin()
    ]

    best_repaired_density = grouped.loc[
        grouped[
            "repaired_density_mean_error"
        ].idxmin()
    ]

    high_connectivity = grouped[
        grouped["raw_connected_rate"] >= 0.95
    ]

    best_high_connectivity = None

    if not high_connectivity.empty:
        best_high_connectivity = (
            high_connectivity.loc[
                high_connectivity[
                    "raw_density_mean_error"
                ].idxmin()
            ]
        )

    isolated_node_scores = node_scores[
        node_scores["isolated_at_0_5"] == 1
    ]["node_max_incident_score"]

    summary = {
        "config": str(args.config),
        "checkpoint": str(
            OmegaConf.select(
                cfg,
                "general.h2_model",
            )
        ),
        "checkpoint_threshold": float(
            OmegaConf.select(
                cfg,
                "model.edge_threshold",
            )
        ),
        "start_seed": args.start_seed,
        "count": args.count,
        "thresholds": thresholds.tolist(),

        "real": {
            "edge_mean": real_edge_mean,
            "edge_median": real_edge_median,
            "density_mean": real_density_mean,
            "density_median": real_density_median,
        },

        "baseline_mismatches": baseline_mismatches,

        "all_node_max_scores": describe(
            node_scores["node_max_incident_score"]
        ),

        "isolated_at_0_5_node_max_scores": (
            describe(isolated_node_scores)
            if len(isolated_node_scores) > 0
            else None
        ),

        "best_raw_density_threshold": (
            best_raw_density.to_dict()
        ),
        "best_raw_edge_threshold": (
            best_raw_edges.to_dict()
        ),
        "best_repaired_density_threshold": (
            best_repaired_density.to_dict()
        ),
        "best_raw_threshold_with_connected_rate_ge_0_95": (
            best_high_connectivity.to_dict()
            if best_high_connectivity is not None
            else None
        ),
    }

    summary_path = (
        args.output_dir
        / "cora_h2_threshold_scan_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=float,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 145)
    print("Cora Beta-H2 continuous threshold scan")
    print("=" * 145)
    print(
        "Baseline reproducibility mismatches:",
        baseline_mismatches,
    )
    print(
        "Real edges mean/median:",
        real_edge_mean,
        real_edge_median,
    )
    print(
        "Real density mean/median:",
        real_density_mean,
        real_density_median,
    )

    if len(isolated_node_scores) > 0:
        print(
            "Node max-score distribution for nodes "
            "isolated at threshold 0.5:"
        )
        print(describe(isolated_node_scores))

    print("\nThreshold summary:")
    print(
        grouped[
            [
                "threshold",
                "raw_edge_mean",
                "raw_density_mean",
                "raw_connected_rate",
                "raw_isolate_graph_rate",
                "repair_edges_mean",
                "repaired_edge_mean",
                "repaired_density_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nBest threshold by raw density:")
    print(best_raw_density.to_dict())

    print("\nBest threshold by raw edge count:")
    print(best_raw_edges.to_dict())

    print("\nBest threshold after connectivity repair:")
    print(best_repaired_density.to_dict())

    print("\nBest raw threshold with connected rate >= 0.95:")
    if best_high_connectivity is None:
        print("NONE")
    else:
        print(best_high_connectivity.to_dict())

    print("\nCSV per seed       :", per_seed_path)
    print("CSV per threshold  :", threshold_path)
    print("CSV summary        :", grouped_path)
    print("CSV node max scores:", node_score_path)
    print("JSON summary       :", summary_path)
    print("=" * 145)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
