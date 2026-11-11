#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# HiGGs retains legacy top-level imports such as:
#     from higgs_utils import *
# Make the diagnostic script independent of the caller's PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

for search_path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "higgs",
    PROJECT_ROOT / "dgd",
):
    value = str(search_path)

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
    parser = argparse.ArgumentParser(
        description=(
            "Sample only Beta-H2 over multiple seeds and audit "
            "connectivity and meta-graph statistics."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Hydra config.yaml from a validated Cora sampling run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--real-meta-csv",
        type=Path,
        default=None,
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
    parser.add_argument(
        "--save-graphs",
        action="store_true",
    )

    return parser.parse_args()


def graph_stats(graph: nx.Graph) -> dict[str, Any]:
    graph = nx.Graph(graph)
    graph.remove_edges_from(
        list(nx.selfloop_edges(graph))
    )

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    if node_count == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "components": 0,
            "isolates": 0,
            "connected": 0,
            "lcc_nodes": 0,
            "min_degree": 0.0,
            "p05_degree": 0.0,
            "p25_degree": 0.0,
            "median_degree": 0.0,
            "mean_degree": 0.0,
            "p75_degree": 0.0,
            "p95_degree": 0.0,
            "max_degree": 0.0,
            "degree_cv": 0.0,
            "density": 0.0,
            "core_density": 0.0,
            "clustering": 0.0,
            "transitivity": 0.0,
        }

    degrees = np.asarray(
        [degree for _, degree in graph.degree()],
        dtype=float,
    )

    components = list(
        nx.connected_components(graph)
    )

    lcc_nodes = max(
        len(component)
        for component in components
    )

    mean_degree = float(degrees.mean())

    degree_cv = (
        float(degrees.std() / mean_degree)
        if mean_degree > 0
        else 0.0
    )

    non_isolated = [
        node
        for node, degree in graph.degree()
        if degree > 0
    ]

    if len(non_isolated) > 1:
        core = graph.subgraph(
            non_isolated
        ).copy()
        core_density = float(
            nx.density(core)
        )
    else:
        core_density = 0.0

    return {
        "nodes": int(node_count),
        "edges": int(edge_count),
        "components": int(len(components)),
        "isolates": int(
            nx.number_of_isolates(graph)
        ),
        "connected": int(
            nx.is_connected(graph)
        ),
        "lcc_nodes": int(lcc_nodes),
        "min_degree": float(degrees.min()),
        "p05_degree": float(
            np.quantile(degrees, 0.05)
        ),
        "p25_degree": float(
            np.quantile(degrees, 0.25)
        ),
        "median_degree": float(
            np.median(degrees)
        ),
        "mean_degree": mean_degree,
        "p75_degree": float(
            np.quantile(degrees, 0.75)
        ),
        "p95_degree": float(
            np.quantile(degrees, 0.95)
        ),
        "max_degree": float(degrees.max()),
        "degree_cv": degree_cv,
        "density": float(
            nx.density(graph)
        ),
        "core_density": core_density,
        "clustering": float(
            nx.average_clustering(graph)
        ),
        "transitivity": float(
            nx.transitivity(graph)
        ),
    }


def distribution(values: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(values) == 0:
        return {
            "min": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    return {
        "min": float(values.min()),
        "p05": float(
            np.quantile(values, 0.05)
        ),
        "p25": float(
            np.quantile(values, 0.25)
        ),
        "median": float(
            np.median(values)
        ),
        "mean": float(values.mean()),
        "p75": float(
            np.quantile(values, 0.75)
        ),
        "p95": float(
            np.quantile(values, 0.95)
        ),
        "max": float(values.max()),
    }


def add_prefixed(
    target: dict[str, Any],
    prefix: str,
    values: dict[str, Any],
) -> None:
    for key, value in values.items():
        target[f"{prefix}_{key}"] = value


def main() -> int:
    args = parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_graph_dir = (
        args.output_dir / "graphs_raw"
    )
    used_graph_dir = (
        args.output_dir / "graphs_used"
    )

    if args.save_graphs:
        raw_graph_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        used_graph_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    cfg = OmegaConf.load(args.config)

    cfg_h2 = cfg.copy()
    cfg_h2["dataset"]["h"] = 2
    cfg_h2["model"]["extra_features"] = "all"

    print("=" * 112)
    print("Loading Beta-H2 checkpoint once")
    print("=" * 112)
    print("Config     :", args.config)
    print(
        "Checkpoint :",
        OmegaConf.select(
            cfg,
            "general.h2_model",
        ),
    )
    print(
        "Family     :",
        OmegaConf.select(
            cfg,
            "general.h2_model_family",
            default="unknown",
        ),
    )

    cfg_h2, model_h2 = sampling.get_samplers(
        cfg_h2
    )

    use_cuda = (
        torch.cuda.is_available()
        and int(cfg.general.gpus) > 0
    )

    device = torch.device(
        "cuda" if use_cuda else "cpu"
    )

    model_h2 = model_h2.to(device)
    model_h2.eval()

    if model_h2.training:
        raise RuntimeError(
            "Beta-H2 is not in eval mode"
        )

    dropout_training = sum(
        int(module.training)
        for module in model_h2.modules()
        if isinstance(
            module,
            (
                torch.nn.Dropout,
                torch.nn.Dropout1d,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
            ),
        )
    )

    if dropout_training != 0:
        raise RuntimeError(
            "Dropout remains active in Beta-H2"
        )

    options = sampling.interface_options(cfg)
    component_policy = options[
        "h2_component_policy"
    ]

    is_beta_h2 = isinstance(
        model_h2,
        sampling.ConditionalGraphBetaDiffusionH2,
    )

    print("Device     :", device)
    print("Model      :", model_h2.__class__.__name__)
    print("Beta-H2    :", is_beta_h2)
    print("Policy     :", component_policy)
    print("Start seed :", args.start_seed)
    print("Count      :", args.count)
    print("=" * 112)

    fixed_condition = None

    if is_beta_h2:
        fixed_condition = (
            sampling.fixed_h2_condition(
                model_h2.ydim_output
            )
            .to(device)
        )

    rows: list[dict[str, Any]] = []
    failed_seeds: list[int] = []

    for offset in range(args.count):
        seed = args.start_seed + offset
        start_time = time.perf_counter()

        row: dict[str, Any] = {
            "seed": seed,
            "status": "success",
            "component_policy": (
                component_policy
            ),
        }

        try:
            seed_sampling(seed)

            sampling_kwargs: dict[str, Any] = {
                "batch_id": 0,
                "batch_size": 1,
                "keep_chain": 0,
                "number_chain_steps": 40,
                "num_nodes": None,
                "figures": False,
            }

            if is_beta_h2:
                sampling_kwargs["y"] = (
                    fixed_condition.clone()
                )
                sampling_kwargs[
                    "graph_number"
                ] = seed

            with torch.inference_mode():
                predicted_graphs, _ = (
                    model_h2.sampling_batch_h2(
                        **sampling_kwargs
                    )
                )

            if len(predicted_graphs) != 1:
                raise RuntimeError(
                    "Expected one H2 graph, got "
                    f"{len(predicted_graphs)}"
                )

            full_nodes = predicted_graphs[0][0]
            full_adjacency = predicted_graphs[0][1]

            (
                used_nodes,
                used_adjacency,
                component_audit,
            ) = sampling.apply_component_policy(
                full_nodes,
                full_adjacency,
                component_policy,
                label="H2",
            )

            raw_graph = sampling.tensor_graph(
                full_nodes,
                full_adjacency,
            )

            used_graph = sampling.tensor_graph(
                used_nodes,
                used_adjacency,
            )

            raw_stats = graph_stats(raw_graph)
            used_stats = graph_stats(used_graph)

            add_prefixed(
                row,
                "raw",
                raw_stats,
            )
            add_prefixed(
                row,
                "used",
                used_stats,
            )

            row["nodes_removed_by_policy"] = (
                raw_stats["nodes"]
                - used_stats["nodes"]
            )
            row["edges_removed_by_policy"] = (
                raw_stats["edges"]
                - used_stats["edges"]
            )

            row["component_audit"] = json.dumps(
                component_audit,
                default=str,
                sort_keys=True,
            )

            if args.save_graphs:
                raw_path = (
                    raw_graph_dir
                    / f"seed_{seed:04d}.pkl"
                )
                used_path = (
                    used_graph_dir
                    / f"seed_{seed:04d}.pkl"
                )

                with raw_path.open("wb") as handle:
                    pickle.dump(
                        raw_graph,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                with used_path.open("wb") as handle:
                    pickle.dump(
                        used_graph,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

            elapsed = (
                time.perf_counter()
                - start_time
            )
            row["elapsed_seconds"] = elapsed

            print(
                f"[{offset + 1:03d}/{args.count:03d}] "
                f"seed={seed:4d} "
                f"raw={raw_stats['nodes']}N/"
                f"{raw_stats['edges']}E "
                f"components={raw_stats['components']} "
                f"isolates={raw_stats['isolates']} "
                f"min_degree={raw_stats['min_degree']:.0f} "
                f"density={raw_stats['density']:.6f} "
                f"used={used_stats['nodes']}N/"
                f"{used_stats['edges']}E "
                f"time={elapsed:.2f}s",
                flush=True,
            )

        except Exception:
            failed_seeds.append(seed)
            row["status"] = "error"
            row["error"] = traceback.format_exc()
            row["elapsed_seconds"] = (
                time.perf_counter()
                - start_time
            )

            print(
                f"[ERROR] seed={seed}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()

        rows.append(row)

    all_results = pd.DataFrame(rows)

    results_csv = (
        args.output_dir
        / "cora_h2_multiseed_results.csv"
    )

    all_results.to_csv(
        results_csv,
        index=False,
    )

    successful = all_results[
        all_results["status"] == "success"
    ].copy()

    if successful.empty:
        raise RuntimeError(
            "All H2-only samples failed"
        )

    raw_connected_rate = float(
        successful["raw_connected"].mean()
    )
    raw_isolate_graph_rate = float(
        (
            successful["raw_isolates"] > 0
        ).mean()
    )

    used_connected_rate = float(
        successful["used_connected"].mean()
    )
    used_isolate_graph_rate = float(
        (
            successful["used_isolates"] > 0
        ).mean()
    )

    raw_disconnected_seeds = (
        successful.loc[
            successful["raw_connected"] == 0,
            "seed",
        ]
        .astype(int)
        .tolist()
    )

    raw_isolate_seeds = (
        successful.loc[
            successful["raw_isolates"] > 0,
            "seed",
        ]
        .astype(int)
        .tolist()
    )

    metrics = [
        "nodes",
        "edges",
        "components",
        "isolates",
        "min_degree",
        "median_degree",
        "mean_degree",
        "max_degree",
        "degree_cv",
        "density",
        "core_density",
        "clustering",
        "transitivity",
    ]

    generated_summary = {
        metric: distribution(
            successful[f"raw_{metric}"]
        )
        for metric in metrics
    }

    summary: dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(
            OmegaConf.select(
                cfg,
                "general.h2_model",
            )
        ),
        "device": str(device),
        "physical_cuda_visible_devices": (
            __import__("os").environ.get(
                "CUDA_VISIBLE_DEVICES",
                "",
            )
        ),
        "component_policy": component_policy,
        "start_seed": args.start_seed,
        "requested_count": args.count,
        "successful_count": int(
            len(successful)
        ),
        "failed_seeds": failed_seeds,
        "raw_connected_rate": (
            raw_connected_rate
        ),
        "raw_isolate_graph_rate": (
            raw_isolate_graph_rate
        ),
        "raw_disconnected_seeds": (
            raw_disconnected_seeds
        ),
        "raw_isolate_seeds": (
            raw_isolate_seeds
        ),
        "used_connected_rate": (
            used_connected_rate
        ),
        "used_isolate_graph_rate": (
            used_isolate_graph_rate
        ),
        "generated_raw": generated_summary,
    }

    real_df = None

    if (
        args.real_meta_csv is not None
        and args.real_meta_csv.is_file()
    ):
        real_df = pd.read_csv(
            args.real_meta_csv
        )

        summary["real_meta_csv"] = str(
            args.real_meta_csv
        )
        summary["real_count"] = int(
            len(real_df)
        )
        summary["real_connected_rate"] = float(
            real_df["connected"].mean()
        )
        summary[
            "real_isolate_graph_rate"
        ] = float(
            (
                real_df["isolates"] > 0
            ).mean()
        )

        summary["real"] = {
            metric: distribution(
                real_df[metric]
            )
            for metric in metrics
            if metric in real_df.columns
        }

    summary_path = (
        args.output_dir
        / "cora_h2_multiseed_summary.json"
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
    print("=" * 126)
    print("Cora Beta-H2 multiseed summary")
    print("=" * 126)
    print(
        "Successful samples       :",
        len(successful),
        "/",
        args.count,
    )
    print(
        "Failed seeds             :",
        failed_seeds,
    )
    print(
        "Raw connected rate       :",
        f"{raw_connected_rate:.6f}",
    )
    print(
        "Raw isolate-graph rate   :",
        f"{raw_isolate_graph_rate:.6f}",
    )
    print(
        "Raw disconnected seeds   :",
        raw_disconnected_seeds,
    )
    print(
        "Raw isolate seeds        :",
        raw_isolate_seeds,
    )
    print(
        "Used connected rate      :",
        f"{used_connected_rate:.6f}",
    )
    print(
        "Used isolate-graph rate  :",
        f"{used_isolate_graph_rate:.6f}",
    )

    if real_df is not None:
        print(
            "Real connected rate      :",
            f"{real_df['connected'].mean():.6f}",
        )
        print(
            "Real isolate-graph rate  :",
            f"{(real_df['isolates'] > 0).mean():.6f}",
        )

        print()
        print("Raw Beta-H2 vs real H2 meta graphs")
        print("-" * 126)
        print(
            f"{'Metric':<18}"
            f"{'Gen p05':>12}"
            f"{'Gen median':>14}"
            f"{'Gen mean':>12}"
            f"{'Gen p95':>12}"
            f"{'Real p05':>12}"
            f"{'Real median':>14}"
            f"{'Real mean':>12}"
            f"{'Real p95':>12}"
        )

        for metric in metrics:
            if metric not in real_df.columns:
                continue

            generated = generated_summary[
                metric
            ]
            real = distribution(
                real_df[metric]
            )

            print(
                f"{metric:<18}"
                f"{generated['p05']:>12.6f}"
                f"{generated['median']:>14.6f}"
                f"{generated['mean']:>12.6f}"
                f"{generated['p95']:>12.6f}"
                f"{real['p05']:>12.6f}"
                f"{real['median']:>14.6f}"
                f"{real['mean']:>12.6f}"
                f"{real['p95']:>12.6f}"
            )

    print("-" * 126)
    print("CSV  :", results_csv)
    print("JSON :", summary_path)
    print("=" * 126)

    return 2 if failed_seeds else 0


if __name__ == "__main__":
    raise SystemExit(main())
