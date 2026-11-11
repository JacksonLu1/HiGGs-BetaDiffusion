"""Offline SBM MMD evaluation aligned with the HiGGs metric implementations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DGD_ROOT = REPO_ROOT / "dgd"
if str(DGD_ROOT) not in sys.path:
    sys.path.insert(0, str(DGD_ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

from analysis import dist_helper, spectre_utils  # noqa: E402
from metrics.mmd import calculators as mmd_calculators  # noqa: E402


EXPECTED_GRAPH_COUNT = 40
SPLIT_SEED = 1234
SELF_CHECK_TOLERANCE = 1e-8
SAMPLE_PATTERN = re.compile(r"^sampled_graph_(\d+)\.pkl$")

PAPER_HIGGS = {
    "Nodes": 0.547,
    "Degree": 0.286,
    "Clustering": 0.411,
    "Eccentricity": 0.0401,
    "Spectre": 0.314,
}
PAPER_REAL_SBM = {
    "Nodes": 107.0,
    "Edges": 522.0,
    "Diameter": 5.46,
    "Clustering": 0.276,
    "Density": 0.099,
    "Transitivity": 0.276,
}
PAPER_COMPATIBLE_MODE = "tv"
PAPER_MODE_REASON = (
    "The active HiGGs SBM evaluation call chain constructs SBMSamplingMetrics and "
    "FullSampleMetrics with compute_emd=False. Nodes always use gaussian_tv; the "
    "active SBM radius call also keeps its default compute_emd=False. The unused "
    "PostProcessSampleMetrics class is not evidence for the reported SBM path."
)

STAT_FIELDS = (
    "nodes",
    "edges",
    "average_degree",
    "diameter",
    "average_clustering",
    "density",
    "transitivity",
    "connected_components",
)

METRIC_FUNCTIONS = {
    "Nodes": "analysis.dist_helper.compute_mmd",
    "Degree": "analysis.spectre_utils.degree_stats",
    "Clustering": "analysis.spectre_utils.clustering_stats",
    "Eccentricity": "metrics.mmd.calculators.radius_stats",
    "Spectre": "analysis.spectre_utils.spectral_stats",
}

METRIC_PARAMETERS = {
    "common": {
        "histogram_normalization": (
            "Each histogram is normalized by sum + 1e-6 inside the project's "
            "compute_mmd implementation."
        ),
        "parallelism": (
            "is_parallel=True, using the project's ThreadPoolExecutor paths. "
            "radius_stats must use this path because its serial branch does not "
            "implement the same shortest-path histogram."
        ),
    },
    "Nodes": {
        "histogram": "one aggregate np.bincount histogram per graph set",
        "tv_kernel": "analysis.dist_helper.gaussian_tv",
        "emd_kernel": "analysis.dist_helper.gaussian_emd",
        "sigma": 1.0,
        "emd_distance_scaling": 1.0,
    },
    "Degree": {
        "histogram": "nx.degree_histogram for every graph",
        "tv_kernel": "analysis.dist_helper.gaussian_tv",
        "emd_kernel": "analysis.dist_helper.gaussian_emd",
        "sigma": 1.0,
        "emd_distance_scaling": 1.0,
    },
    "Clustering": {
        "histogram": "np.histogram(nx.clustering values, bins=100, range=(0, 1))",
        "bins": 100,
        "tv_kernel": "analysis.dist_helper.gaussian_tv",
        "emd_kernel": "analysis.dist_helper.gaussian_emd",
        "sigma": 0.1,
        "emd_distance_scaling": 100.0,
    },
    "Eccentricity": {
        "histogram": (
            "50-bin histogram of all-pairs shortest-path lengths, including "
            "zero self-distances, after radius_worker selects the largest "
            "connected component"
        ),
        "bins": 50,
        "tv_kernel": "metrics.mmd.kernels.gaussian_tv",
        "emd_kernel": (
            "metrics.mmd.kernels.gaussian (the project's actual "
            "radius_stats(compute_emd=True) implementation)"
        ),
        "sigma": 1.0,
    },
    "Spectre": {
        "histogram": (
            "200-bin normalized-Laplacian eigenvalue histogram over "
            "range=(-1e-5, 2) with n_eigvals=-1"
        ),
        "bins": 200,
        "laplacian": "networkx.normalized_laplacian_matrix",
        "tv_kernel": "analysis.dist_helper.gaussian_tv",
        "emd_kernel": "analysis.dist_helper.gaussian_emd",
        "sigma": 1.0,
        "emd_distance_scaling": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and evaluate 40 generated SBM graphs against the HiGGs test split."
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        required=True,
        help="Directory recursively containing sampled_graph_0.pkl through sampled_graph_39.pkl.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Project data directory containing sbm/raw/sbm_200.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which the new mode-specific audit files will be written.",
    )
    parser.add_argument(
        "--mode",
        choices=("tv", "emd", "both", "paper"),
        default="both",
        help="Metric kernel mode. 'paper' selects the mode proven by the HiGGs call chain.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run only order-invariance/identity checks; do not write formal MMD result files.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sbm_mmd_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "sbm_mmd_audit.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_reproducible_seeds() -> None:
    random.seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    torch.manual_seed(SPLIT_SEED)


def discover_sample_files(samples_dir: Path) -> list[tuple[int, Path]]:
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Samples directory does not exist: {samples_dir}")

    numbered_paths: dict[int, Path] = {}
    duplicates: dict[int, list[Path]] = {}
    for path in samples_dir.rglob("sampled_graph_*.pkl"):
        match = SAMPLE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        graph_number = int(match.group(1))
        if graph_number in numbered_paths:
            duplicates.setdefault(graph_number, [numbered_paths[graph_number]]).append(path)
        else:
            numbered_paths[graph_number] = path

    if duplicates:
        details = "; ".join(
            f"{number}: {[str(path) for path in paths]}"
            for number, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate sampled graph numbers found: {details}")

    expected = set(range(EXPECTED_GRAPH_COUNT))
    found = set(numbered_paths)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    if missing or unexpected or len(numbered_paths) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            "Expected exactly sampled_graph_0.pkl through sampled_graph_39.pkl; "
            f"missing={missing}, unexpected={unexpected}, found={len(numbered_paths)}"
        )

    return [(number, numbered_paths[number]) for number in sorted(numbered_paths)]


def load_pickle_graph(path: Path) -> nx.Graph:
    try:
        with path.open("rb") as handle:
            graph = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to load generated graph {path}: {exc}") from exc

    graph_types = (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
    if not isinstance(graph, graph_types):
        raise TypeError(
            f"Generated object in {path} is {type(graph).__name__}, not a NetworkX graph"
        )
    if graph.number_of_nodes() == 0:
        raise ValueError(f"Generated graph is empty: {path}")
    return graph


def prepare_graph(graph: nx.Graph, label: str) -> tuple[nx.Graph, int, int, int]:
    """Apply the same non-destructive graph conversion to real and generated data."""
    simple_graph = nx.Graph(graph)
    simple_graph.remove_edges_from(list(nx.selfloop_edges(simple_graph)))
    source_nodes = simple_graph.number_of_nodes()
    source_edges = simple_graph.number_of_edges()
    if source_nodes == 0:
        raise ValueError(f"Graph is empty after conversion: {label}")

    component_count = nx.number_connected_components(simple_graph)
    return simple_graph, component_count, source_nodes, source_edges


def diameter_on_largest_component(graph: nx.Graph) -> tuple[float, bool]:
    if nx.is_connected(graph):
        return float(nx.diameter(graph)), False
    largest_nodes = max(nx.connected_components(graph), key=len)
    largest_component = graph.subgraph(largest_nodes)
    return float(nx.diameter(largest_component)), True


def graph_statistics(
    graph: nx.Graph,
    graph_number: int,
    component_count: int,
    source_nodes: int,
    source_edges: int,
    source_path: str | None = None,
    test_order: int | None = None,
) -> dict[str, Any]:
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    diameter, diameter_used_lcc = diameter_on_largest_component(graph)
    values: dict[str, Any] = {
        "graph_number": graph_number,
        "nodes": nodes,
        "edges": edges,
        "average_degree": 2.0 * edges / nodes,
        "diameter": diameter,
        "average_clustering": nx.average_clustering(graph),
        "density": nx.density(graph),
        "transitivity": nx.transitivity(graph),
        "connected_components": component_count,
        "source_nodes": source_nodes,
        "source_edges": source_edges,
        "graph_metrics_used_largest_connected_component": False,
        "diameter_stat_used_largest_connected_component": diameter_used_lcc,
    }
    if source_path is not None:
        values["source_path"] = source_path
    if test_order is not None:
        values["test_order"] = test_order

    for field in STAT_FIELDS:
        value = float(values[field])
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite statistic {field}={values[field]} for graph {graph_number}"
            )
    return values


def load_generated_graphs(
    numbered_paths: Sequence[tuple[int, Path]], logger: logging.Logger
) -> tuple[list[nx.Graph], list[dict[str, Any]]]:
    graphs: list[nx.Graph] = []
    rows: list[dict[str, Any]] = []
    for graph_number, path in numbered_paths:
        loaded = load_pickle_graph(path)
        graph, components, source_nodes, source_edges = prepare_graph(loaded, str(path))
        graphs.append(graph)
        rows.append(
            graph_statistics(
                graph,
                graph_number,
                components,
                source_nodes,
                source_edges,
                source_path=str(path.resolve()),
            )
        )
        logger.info(
            "Generated graph %d: nodes=%d edges=%d components=%d (full graph retained)",
            graph_number,
            graph.number_of_nodes(),
            graph.number_of_edges(),
            components,
        )
    return graphs, rows


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prescribed_test_indices(dataset_size: int) -> tuple[list[int], tuple[int, int, int]]:
    test_len = int(round(dataset_size * 0.2))
    train_len = int(round((dataset_size - test_len) * 0.8))
    val_len = dataset_size - train_len - test_len
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    test_indices = permutation[train_len + val_len :]
    return test_indices, (train_len, val_len, test_len)


def adjacency_to_graph(adjacency: Any, label: str) -> nx.Graph:
    if not torch.is_tensor(adjacency):
        raise TypeError(f"Adjacency {label} is {type(adjacency).__name__}, not a tensor")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency {label} is not square: shape={tuple(adjacency.shape)}")
    if adjacency.shape[0] == 0:
        raise ValueError(f"Adjacency {label} is empty")
    if not torch.isfinite(adjacency).all():
        raise ValueError(f"Adjacency {label} contains NaN or Inf")
    array = adjacency.detach().cpu().numpy()
    return nx.from_numpy_array(array, create_using=nx.Graph)


def load_real_test_graphs(
    data_root: Path, logger: logging.Logger
) -> tuple[list[nx.Graph], list[dict[str, Any]], list[int]]:
    raw_path = data_root.resolve() / "sbm" / "raw" / "sbm_200.pt"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Official SBM data file does not exist: {raw_path}")

    payload = torch_load_cpu(raw_path)
    if not isinstance(payload, (list, tuple)) or len(payload) != 8:
        raise ValueError(
            f"Unexpected SBM payload in {raw_path}: expected an 8-item list/tuple"
        )
    adjacencies = payload[0]
    if not isinstance(adjacencies, (list, tuple)):
        raise TypeError("The first SBM payload item is not an adjacency sequence")
    if len(adjacencies) != 200:
        raise ValueError(f"Expected 200 official SBM graphs, found {len(adjacencies)}")

    test_indices, split_lengths = prescribed_test_indices(len(adjacencies))
    if split_lengths != (128, 32, 40) or len(test_indices) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Unexpected prescribed split lengths={split_lengths}, test={len(test_indices)}"
        )

    graphs: list[nx.Graph] = []
    rows: list[dict[str, Any]] = []
    for test_order, dataset_index in enumerate(test_indices):
        loaded = adjacency_to_graph(
            adjacencies[dataset_index], f"official SBM index {dataset_index}"
        )
        graph, components, source_nodes, source_edges = prepare_graph(
            loaded, f"official SBM index {dataset_index}"
        )
        graphs.append(graph)
        rows.append(
            graph_statistics(
                graph,
                dataset_index,
                components,
                source_nodes,
                source_edges,
                source_path=str(raw_path),
                test_order=test_order,
            )
        )
        logger.info(
            "Real test graph index %d: nodes=%d edges=%d components=%d (full graph retained)",
            dataset_index,
            graph.number_of_nodes(),
            graph.number_of_edges(),
            components,
        )
    return graphs, rows, test_indices


def validate_histogram(histogram: np.ndarray, label: str) -> None:
    if histogram.size == 0 or float(np.sum(histogram)) <= 0:
        raise ValueError(f"Empty histogram for {label}")
    if not np.isfinite(histogram).all():
        raise ValueError(f"Histogram for {label} contains NaN or Inf")


def finite_metric(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric {label} is NaN or Inf: {result}")
    return result


def node_count_mmd(
    real_graphs: Sequence[nx.Graph],
    generated_graphs: Sequence[nx.Graph],
    mode: str,
    logger: logging.Logger | None = None,
) -> float:
    real_counts = [graph.number_of_nodes() for graph in real_graphs]
    generated_counts = [graph.number_of_nodes() for graph in generated_graphs]
    real_histogram = np.bincount(real_counts)
    generated_histogram = np.bincount(generated_counts)
    validate_histogram(real_histogram, "real node counts")
    validate_histogram(generated_histogram, "generated node counts")

    kernel = dist_helper.gaussian_tv if mode == "tv" else dist_helper.gaussian_emd
    if logger is not None:
        logger.info(
            "Nodes/%s via analysis.dist_helper.compute_mmd: real_counts=%s generated_counts=%s",
            mode,
            real_counts,
            generated_counts,
        )
        logger.info(
            "Nodes/%s aggregate histogram lengths: real=%d generated=%d; kernel=%s",
            mode,
            len(real_histogram),
            len(generated_histogram),
            "analysis.dist_helper.gaussian_tv"
            if mode == "tv"
            else "analysis.dist_helper.gaussian_emd",
        )
    return finite_metric(
        dist_helper.compute_mmd(
            [real_histogram], [generated_histogram], kernel=kernel
        ),
        f"Nodes/{mode}",
    )


def compute_metric_set(
    real_graphs: Sequence[nx.Graph],
    generated_graphs: Sequence[nx.Graph],
    mode: str,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    if mode not in {"tv", "emd"}:
        raise ValueError(f"Unsupported internal metric mode: {mode}")
    compute_emd = mode == "emd"
    metrics = {
        "Nodes": node_count_mmd(real_graphs, generated_graphs, mode, logger),
        "Degree": spectre_utils.degree_stats(
            real_graphs,
            generated_graphs,
            is_parallel=True,
            compute_emd=compute_emd,
        ),
        "Clustering": spectre_utils.clustering_stats(
            real_graphs,
            generated_graphs,
            bins=100,
            is_parallel=True,
            compute_emd=compute_emd,
        ),
        "Eccentricity": mmd_calculators.radius_stats(
            real_graphs,
            generated_graphs,
            compute_emd=compute_emd,
            is_parallel=True,
            PRINT_TIME=False,
        ),
        "Spectre": spectre_utils.spectral_stats(
            real_graphs,
            generated_graphs,
            is_parallel=True,
            n_eigvals=-1,
            compute_emd=compute_emd,
        ),
    }
    return {
        name: finite_metric(value, f"{name}/{mode}")
        for name, value in metrics.items()
    }


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in STAT_FIELDS:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Cannot summarize invalid field: {field}")
        summary[field] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def real_baseline_comparison(
    real_summary: dict[str, dict[str, float]], logger: logging.Logger
) -> dict[str, dict[str, float]]:
    field_map = {
        "Nodes": "nodes",
        "Edges": "edges",
        "Diameter": "diameter",
        "Clustering": "average_clustering",
        "Density": "density",
        "Transitivity": "transitivity",
    }
    comparison: dict[str, dict[str, float]] = {}
    logger.info("Real SBM test means compared with the paper's real-data statistics:")
    for label, field in field_map.items():
        current = real_summary[field]["mean"]
        paper = PAPER_REAL_SBM[label]
        comparison[label] = {
            "current_test_mean": current,
            "paper_real_sbm": paper,
            "difference": current - paper,
            "absolute_difference": abs(current - paper),
        }
        logger.info(
            "  %-14s current=%10.6f paper=%10.6f difference=%+10.6f",
            label,
            current,
            paper,
            current - paper,
        )
    return comparison


def metric_rows(metrics: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {
            "metric": name,
            "generated_result": metrics[name],
            "paper_higgs": PAPER_HIGGS[name],
            "absolute_difference": abs(metrics[name] - PAPER_HIGGS[name]),
        }
        for name in PAPER_HIGGS
    ]


def write_metric_output(
    output_dir: Path,
    stem: str,
    mode: str,
    metrics: dict[str, float],
) -> list[str]:
    rows = metric_rows(metrics)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "mode": mode,
        "paper_compatible_mode": PAPER_COMPATIBLE_MODE,
        "paper_mode_reason": PAPER_MODE_REASON,
        "metrics": rows,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return [str(csv_path), str(json_path)]


def write_audit(output_dir: Path, payload: dict[str, Any]) -> str:
    path = output_dir / "sbm_mmd_audit.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return str(path)


def print_metric_table(
    title: str, rows: Iterable[dict[str, Any]], logger: logging.Logger
) -> None:
    logger.info("%s", title)
    logger.info("%-15s %12s %15s %14s", "Metric", "Current", "Paper HiGGs", "Difference")
    for row in rows:
        logger.info(
            "%-15s %12.6f %15.6f %14.6f",
            row["metric"],
            row["generated_result"],
            row["paper_higgs"],
            row["absolute_difference"],
        )


def computation_modes(requested_mode: str) -> list[str]:
    if requested_mode == "both":
        return ["tv", "emd"]
    if requested_mode == "paper":
        return [PAPER_COMPATIBLE_MODE]
    return [requested_mode]


def run_self_checks(
    real_graphs: Sequence[nx.Graph],
    generated_graphs: Sequence[nx.Graph],
    modes: Sequence[str],
    logger: logging.Logger,
) -> dict[str, Any]:
    shuffled_generated = list(generated_graphs)
    random.Random(SPLIT_SEED).shuffle(shuffled_generated)
    checks: dict[str, Any] = {}
    for mode in modes:
        mode_checks: dict[str, dict[str, float]] = {}
        cases = {
            "real_vs_real": (real_graphs, real_graphs),
            "generated_vs_generated": (generated_graphs, generated_graphs),
            "generated_vs_shuffled_generated": (
                generated_graphs,
                shuffled_generated,
            ),
        }
        for case_name, (reference, candidate) in cases.items():
            values = compute_metric_set(reference, candidate, mode, logger=None)
            failures = {
                metric: value
                for metric, value in values.items()
                if abs(value) > SELF_CHECK_TOLERANCE
            }
            if failures:
                raise AssertionError(
                    f"Self-check failed for {mode}/{case_name}; "
                    f"tolerance={SELF_CHECK_TOLERANCE}, values={failures}"
                )
            mode_checks[case_name] = values
            logger.info(
                "Self-check passed: mode=%s case=%s tolerance=%g values=%s",
                mode,
                case_name,
                SELF_CHECK_TOLERANCE,
                values,
            )
        checks[mode] = mode_checks
    return {
        "tolerance": SELF_CHECK_TOLERANCE,
        "checks": checks,
        "formal_result_files_written": False,
    }


def log_audit_configuration(logger: logging.Logger) -> None:
    logger.info("Paper-compatible mode: %s", PAPER_COMPATIBLE_MODE)
    logger.info("Paper-compatible selection evidence: %s", PAPER_MODE_REASON)
    for metric, function_name in METRIC_FUNCTIONS.items():
        logger.info("Metric function: %-15s %s", metric, function_name)
    logger.info("Metric parameters: %s", json.dumps(METRIC_PARAMETERS, ensure_ascii=False))
    logger.info(
        "Preprocessing: nx.Graph conversion and self-loop removal only; preserve "
        "isolated nodes, labels, and disconnected components for every graph."
    )
    logger.info(
        "Largest connected component policy: only radius_worker and the standalone "
        "diameter summary select the LCC; no graph passed to MMD is globally truncated."
    )


def run(args: argparse.Namespace, logger: logging.Logger) -> None:
    set_reproducible_seeds()
    log_audit_configuration(logger)

    numbered_paths = discover_sample_files(args.samples_dir.resolve())
    generated_graphs, generated_rows = load_generated_graphs(numbered_paths, logger)
    real_graphs, real_rows, test_indices = load_real_test_graphs(
        args.data_root.resolve(), logger
    )
    if len(real_graphs) != EXPECTED_GRAPH_COUNT:
        raise ValueError(f"Real test set must contain 40 graphs, found {len(real_graphs)}")
    if len(generated_graphs) != EXPECTED_GRAPH_COUNT:
        raise ValueError(
            f"Generated set must contain 40 graphs, found {len(generated_graphs)}"
        )

    logger.info("Real test graphs: %d", len(real_graphs))
    logger.info("Generated graphs: %d", len(generated_graphs))
    logger.info("Split seed: %d; lengths: train=128 validation=32 test=40", SPLIT_SEED)
    logger.info("Exact test indices: %s", test_indices)

    generated_summary = summarize_rows(generated_rows)
    real_summary = summarize_rows(real_rows)
    baseline_comparison = real_baseline_comparison(real_summary, logger)
    modes = computation_modes(args.mode)
    output_dir = args.output_dir.resolve()

    audit_payload: dict[str, Any] = {
        "requested_mode": args.mode,
        "paper_compatible_mode": PAPER_COMPATIBLE_MODE,
        "paper_compatible_selection_evidence": {
            "reason": PAPER_MODE_REASON,
            "active_call_chain": [
                "dgd/main.py -> analysis.spectre_utils.SBMSamplingMetrics",
                "analysis.spectre_utils.SBMSamplingMetrics -> SpectreSamplingMetrics(compute_emd=False)",
                "higgs/higgs_utils.py -> analysis.spectre_utils.FullSampleMetrics(compute_emd=False)",
            ],
            "nodes_behavior": "SpectreSamplingMetrics always calls compute_mmd with gaussian_tv for Nodes.",
            "radius_behavior": "The active SBM radius_stats call omits compute_emd, retaining False.",
            "excluded_evidence": (
                "PostProcessSampleMetrics hard-codes compute_emd=True but has no active "
                "repository call site and therefore does not select the paper mode."
            ),
        },
        "metric_functions": METRIC_FUNCTIONS,
        "metric_parameters": METRIC_PARAMETERS,
        "preprocessing": {
            "real_and_generated_identical": True,
            "steps": [
                "convert to networkx.Graph",
                "remove self-loops",
                "retain all nodes including isolates",
                "retain all connected components",
                "do not relabel nodes",
            ],
            "global_largest_connected_component": False,
            "radius_largest_connected_component": True,
            "diameter_summary_largest_connected_component_if_disconnected": True,
        },
        "test_split": {
            "dataset_size": 200,
            "seed": SPLIT_SEED,
            "algorithm": "torch.randperm with torch.Generator().manual_seed(1234)",
            "lengths": {"train": 128, "validation": 32, "test": 40},
            "test_indices": test_indices,
        },
        "graph_counts": {
            "real_test": len(real_graphs),
            "generated": len(generated_graphs),
        },
        "real_test_summary": real_summary,
        "generated_summary": generated_summary,
        "paper_real_sbm_comparison": baseline_comparison,
        "real_test_per_graph_stats": real_rows,
        "generated_per_graph_stats": generated_rows,
        "self_check_requested": args.self_check,
    }

    if args.self_check:
        audit_payload["self_check"] = run_self_checks(
            real_graphs, generated_graphs, modes, logger
        )
        audit_path = write_audit(output_dir, audit_payload)
        logger.info("Self-check only: no formal MMD result files were written.")
        logger.info("Audit saved to: %s", audit_path)
        logger.info(
            "Final selected paper-compatible mode=%s; reason=%s",
            PAPER_COMPATIBLE_MODE,
            PAPER_MODE_REASON,
        )
        return

    results: dict[str, dict[str, float]] = {}
    for mode in modes:
        logger.info("Computing complete five-metric %s result set", mode.upper())
        results[mode] = compute_metric_set(
            real_graphs, generated_graphs, mode, logger=logger
        )

    output_files: list[str] = []
    if "tv" in results and args.mode in {"tv", "both"}:
        output_files.extend(
            write_metric_output(output_dir, "sbm_mmd_results_tv", "tv", results["tv"])
        )
    if "emd" in results:
        output_files.extend(
            write_metric_output(output_dir, "sbm_mmd_results_emd", "emd", results["emd"])
        )
    if "tv" in results:
        output_files.extend(
            write_metric_output(
                output_dir,
                "sbm_mmd_results_paper_compatible",
                PAPER_COMPATIBLE_MODE,
                results[PAPER_COMPATIBLE_MODE],
            )
        )

    audit_payload["results"] = {
        mode: metric_rows(values) for mode, values in results.items()
    }
    audit_payload["formal_output_files"] = output_files
    audit_path = write_audit(output_dir, audit_payload)

    if args.mode == "paper":
        print_metric_table(
            "Paper-compatible MMD results (TV)",
            metric_rows(results[PAPER_COMPATIBLE_MODE]),
            logger,
        )
    else:
        for mode in modes:
            print_metric_table(
                f"{mode.upper()} MMD results",
                metric_rows(results[mode]),
                logger,
            )

    logger.info("New result files: %s", output_files)
    logger.info("Audit saved to: %s", audit_path)
    logger.info(
        "Final selected paper-compatible mode=%s; reason=%s",
        PAPER_COMPATIBLE_MODE,
        PAPER_MODE_REASON,
    )


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.output_dir.resolve())
    try:
        run(args, logger)
    except Exception:
        logger.exception("SBM MMD audit/evaluation failed")
        raise


if __name__ == "__main__":
    main()
