#!/usr/bin/env python3
"""Summarize saved HiGGs interface-ablation intermediates.

The input run artifacts are treated as read-only.  This tool only writes the
four ``interface_summary*`` files in each explicitly supplied run directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np


SAMPLE_PATTERN = re.compile(r"^sample_(\d+)$")
COMMUNITY_PATTERN = re.compile(r"^community_(\d+)\.pkl$")
OUTPUT_NAMES = (
    "interface_summary_samples.csv",
    "interface_summary_h1.csv",
    "interface_summary.json",
    "interface_summary.txt",
)
REAL_H2_OUTPUT_NAMES = (
    "real_h2_r1_summary.json",
    "real_h2_r1_summary.txt",
    "real_h2_r1_graphs.csv",
)


class SummaryError(RuntimeError):
    """Raised when an input run is incomplete or internally inconsistent."""


def load_graph(path: Path) -> nx.Graph:
    """Load and strictly validate one saved NetworkX undirected graph."""
    if not path.is_file():
        raise FileNotFoundError(f"Required graph file is missing: {path}")
    with path.open("rb") as handle:
        graph = pickle.load(handle)
    if not isinstance(graph, nx.Graph) or graph.is_multigraph() or graph.is_directed():
        raise TypeError(
            f"Expected an undirected networkx.Graph (not MultiGraph), got "
            f"{type(graph).__module__}.{type(graph).__qualname__}: {path}"
        )
    return graph


def complete_status(node_count: int, edge_count: int) -> tuple[bool | None, bool]:
    """Return (is_complete, is_applicable); n <= 1 is recorded separately."""
    if node_count <= 1:
        return None, False
    possible_edges = node_count * (node_count - 1) // 2
    return edge_count == possible_edges, True


def graph_stats(graph: nx.Graph) -> dict[str, Any]:
    """Calculate topology statistics for one validated graph."""
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    component_sizes = sorted(
        (len(component) for component in nx.connected_components(graph)),
        reverse=True,
    )
    is_complete, complete_applicable = complete_status(node_count, edge_count)
    return {
        "nodes": node_count,
        "edges": edge_count,
        "components": len(component_sizes),
        "component_sizes": component_sizes,
        "density": float(nx.density(graph)),
        "is_complete": is_complete,
        "complete_applicable": complete_applicable,
    }


def _edge_signature(graph: nx.Graph) -> frozenset[frozenset[Any]]:
    return frozenset(frozenset((source, target)) for source, target in graph.edges())


def graphs_differ(full_graph: nx.Graph, used_graph: nx.Graph) -> bool:
    """Compare the saved node and edge identities without graph isomorphism."""
    return (
        frozenset(full_graph.nodes()) != frozenset(used_graph.nodes())
        or _edge_signature(full_graph) != _edge_signature(used_graph)
    )


def compare_graphs(full_graph: nx.Graph, used_graph: nx.Graph) -> dict[str, Any]:
    """Return the requested full-versus-used topology fields."""
    full = graph_stats(full_graph)
    used = graph_stats(used_graph)
    removed_nodes = full["nodes"] - used["nodes"]
    removed_edges = full["edges"] - used["edges"]
    if removed_nodes < 0 or removed_edges < 0:
        raise SummaryError(
            "The used graph is larger than the corresponding full graph: "
            f"nodes {full['nodes']}->{used['nodes']}, "
            f"edges {full['edges']}->{used['edges']}"
        )
    return {
        "full_nodes": full["nodes"],
        "used_nodes": used["nodes"],
        "full_edges": full["edges"],
        "used_edges": used["edges"],
        "full_components": full["components"],
        "used_components": used["components"],
        "full_component_sizes": full["component_sizes"],
        "used_component_sizes": used["component_sizes"],
        "full_density": full["density"],
        "used_density": used["density"],
        "removed_nodes": removed_nodes,
        "removed_edges": removed_edges,
        "lcc_triggered": graphs_differ(full_graph, used_graph),
        "is_complete_full": full["is_complete"],
        "is_complete_used": used["is_complete"],
        "complete_full_applicable": full["complete_applicable"],
        "complete_used_applicable": used["complete_applicable"],
    }


def _adjacency_numpy(adjacency: Any) -> np.ndarray:
    """Convert a tensor/array-like adjacency to a validated binary matrix."""
    if hasattr(adjacency, "detach") and hasattr(adjacency, "cpu"):
        adjacency = adjacency.detach().cpu().numpy()
    array = np.asarray(adjacency)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise SummaryError(
            f"Original adjacency must be a square 2-D matrix, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise SummaryError("Original adjacency contains non-finite values")
    if not np.all(np.logical_or(array == 0, array == 1)):
        unique = np.unique(array).tolist()
        raise SummaryError(f"Original adjacency is not binary; unique values: {unique}")
    if not np.array_equal(array, array.T):
        raise SummaryError("Original adjacency is not symmetric")
    if np.count_nonzero(np.diag(array)):
        raise SummaryError("Original adjacency contains self-loops")
    return array.astype(np.uint8, copy=False)


def build_h2_meta_graph(
    adjacency: Any,
    communities: Sequence[Iterable[int]],
    *,
    global_node_offset: int,
) -> nx.Graph:
    """Build one H2 graph from a saved partition without rerunning Louvain.

    Community order is the stable H2 node order.  Every community is explicitly
    inserted before inter-community edges, so isolated H2 nodes are preserved.
    """
    array = _adjacency_numpy(adjacency)
    node_count = int(array.shape[0])
    if not isinstance(communities, (list, tuple)):
        raise SummaryError(
            "Each saved partition must be a list/tuple of communities, got "
            f"{type(communities).__module__}.{type(communities).__qualname__}"
        )
    if not communities:
        raise SummaryError("A saved partition must contain at least one community")

    node_to_community: dict[int, int] = {}
    normalized_communities: list[set[int]] = []
    for community_id, members in enumerate(communities):
        member_set = set(members)
        if not member_set:
            raise SummaryError(f"Saved community {community_id} is empty")
        if any(not isinstance(node, (int, np.integer)) for node in member_set):
            raise SummaryError(
                f"Saved community {community_id} contains a non-integer node ID"
            )
        normalized = {int(node) for node in member_set}
        overlap = set(node_to_community).intersection(normalized)
        if overlap:
            raise SummaryError(
                f"Saved partition contains nodes in multiple communities: "
                f"{sorted(overlap)[:10]}"
            )
        for node in normalized:
            node_to_community[node] = community_id
        normalized_communities.append(normalized)

    expected_nodes = set(
        range(global_node_offset, global_node_offset + node_count)
    )
    actual_nodes = set(node_to_community)
    if actual_nodes != expected_nodes:
        missing = sorted(expected_nodes - actual_nodes)
        extra = sorted(actual_nodes - expected_nodes)
        raise SummaryError(
            "Saved partition does not exactly cover the original graph node range "
            f"[{global_node_offset}, {global_node_offset + node_count}): "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    meta_graph = nx.Graph()
    meta_graph.add_nodes_from(
        (
            community_id,
            {"community_id": community_id, "original_nodes": sorted(members)},
        )
        for community_id, members in enumerate(normalized_communities)
    )

    upper_edges = np.argwhere(np.triu(array, k=1) == 1)
    for local_source, local_target in upper_edges:
        source = int(local_source) + global_node_offset
        target = int(local_target) + global_node_offset
        source_community = node_to_community[source]
        target_community = node_to_community[target]
        if source_community != target_community:
            meta_graph.add_edge(source_community, target_community)

    if meta_graph.number_of_nodes() != len(normalized_communities):
        raise SummaryError(
            "H2 construction lost a saved community: "
            f"{meta_graph.number_of_nodes()} != {len(normalized_communities)}"
        )
    if nx.number_of_selfloops(meta_graph):
        raise SummaryError("H2 construction unexpectedly produced a self-loop")
    return meta_graph


def pair_h1_files(sample_dir: Path) -> list[tuple[int, Path, Path]]:
    """Pair H1 full/used files by the exact same community filename."""
    full_dir = sample_dir / "h1_full"
    used_dir = sample_dir / "h1_used"
    if not full_dir.is_dir():
        raise FileNotFoundError(f"Required H1 directory is missing: {full_dir}")
    if not used_dir.is_dir():
        raise FileNotFoundError(f"Required H1 directory is missing: {used_dir}")

    full_files = {path.name: path for path in full_dir.glob("community_*.pkl")}
    used_files = {path.name: path for path in used_dir.glob("community_*.pkl")}
    if not full_files:
        raise FileNotFoundError(f"No community_*.pkl files found in: {full_dir}")

    missing_used = sorted(set(full_files) - set(used_files))
    missing_full = sorted(set(used_files) - set(full_files))
    if missing_used or missing_full:
        details = []
        if missing_used:
            details.append(f"missing from h1_used: {missing_used}")
        if missing_full:
            details.append(f"missing from h1_full: {missing_full}")
        raise FileNotFoundError(
            f"H1 community files are not paired by identical filename in "
            f"{sample_dir}: {'; '.join(details)}"
        )

    paired: list[tuple[int, Path, Path]] = []
    for filename in full_files:
        match = COMMUNITY_PATTERN.fullmatch(filename)
        if match is None:
            raise SummaryError(f"Invalid H1 community filename: {full_files[filename]}")
        paired.append((int(match.group(1)), full_files[filename], used_files[filename]))
    paired.sort(key=lambda item: item[0])
    return paired


def discover_samples(run_dir: Path) -> list[tuple[int, Path]]:
    intermediates = run_dir / "sampling" / "intermediates"
    if not intermediates.is_dir():
        raise FileNotFoundError(
            f"Required intermediates directory is missing: {intermediates}"
        )
    samples: list[tuple[int, Path]] = []
    for path in intermediates.iterdir():
        if not path.is_dir():
            continue
        match = SAMPLE_PATTERN.fullmatch(path.name)
        if match is not None:
            samples.append((int(match.group(1)), path))
    if not samples:
        raise FileNotFoundError(f"No sample_NNN directories found in: {intermediates}")
    samples.sort(key=lambda item: item[0])
    sample_ids = [sample_id for sample_id, _ in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise SummaryError(f"Duplicate numeric sample IDs found in: {intermediates}")
    return samples


def read_audit(sample_dir: Path) -> dict[str, Any]:
    path = sample_dir / "interface_audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"Required audit JSON is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
    except json.JSONDecodeError as error:
        raise SummaryError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(audit, dict):
        raise SummaryError(f"Audit JSON top level must be an object: {path}")
    return audit


def audit_value(
    audit: dict[str, Any],
    keys: Sequence[str],
    warnings: list[str],
    context: str,
) -> Any:
    """Read one exact JSON path; do not infer or alias unknown field names."""
    value: Any = audit
    for depth, key in enumerate(keys):
        if not isinstance(value, dict) or key not in value:
            parent = ".".join(keys[:depth]) or "<top-level>"
            available = sorted(value.keys()) if isinstance(value, dict) else []
            message = (
                f"{context}: missing audit field {'.'.join(keys)}; "
                f"available keys at {parent}: {available}. Recorded null."
            )
            warnings.append(message)
            return None
        value = value[key]
    return value


def _distribution(values: Iterable[int]) -> dict[str, int]:
    return {
        str(value): count
        for value, count in sorted(Counter(values).items(), key=lambda item: item[0])
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    known = [float(value) for value in values if value is not None]
    return fmean(known) if known else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sum_known(values: Iterable[int | float | None]) -> int | float | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _numeric_summary(values: Sequence[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    numeric = [float(value) for value in values]
    return {
        "mean": fmean(numeric),
        "std": pstdev(numeric),
        "min": min(values),
        "max": max(values),
    }


def _load_torch_payload(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required SBM tensor file is missing: {path}")
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_real_sbm_r1_inputs(raw_dir: Path) -> dict[str, Any]:
    """Load and validate the actual r=1 raw files without changing them."""
    raw_dir = raw_dir.expanduser().resolve()
    adjacency_path = raw_dir / "sbm_200.pt"
    partition_path = raw_dir / "sbm_partitions_r1.pkl"
    payload = _load_torch_payload(adjacency_path)
    if not isinstance(payload, (list, tuple)) or len(payload) != 8:
        raise SummaryError(
            "sbm_200.pt must contain the observed 8-item list/tuple structure; "
            f"got {type(payload).__name__} with "
            f"length={len(payload) if hasattr(payload, '__len__') else None}"
        )
    adjs, _, _, n_nodes, _, _, _, n_max = payload
    if not isinstance(adjs, list) or not isinstance(n_nodes, list):
        raise SummaryError(
            "sbm_200.pt observed fields adjs and n_nodes must both be lists"
        )
    if len(adjs) != len(n_nodes):
        raise SummaryError(
            f"Adjacency/node-count length mismatch: {len(adjs)} != {len(n_nodes)}"
        )

    if not partition_path.is_file():
        raise FileNotFoundError(f"Required partition pickle is missing: {partition_path}")
    with partition_path.open("rb") as handle:
        partitions = pickle.load(handle)
    if not isinstance(partitions, list):
        raise SummaryError(
            "sbm_partitions_r1.pkl must contain the observed list structure, got "
            f"{type(partitions).__module__}.{type(partitions).__qualname__}"
        )
    if len(partitions) != len(adjs):
        raise SummaryError(
            f"Partition/graph length mismatch: {len(partitions)} != {len(adjs)}"
        )

    split_ids: dict[str, list[int]] = {}
    split_headers: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        split_path = raw_dir / f"{split}_r1.csv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Required split CSV is missing: {split_path}")
        with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            split_headers[split] = list(reader.fieldnames or [])
            if "community_id" not in split_headers[split]:
                raise SummaryError(
                    f"Split CSV lacks observed 'community_id' column: {split_path}; "
                    f"headers={split_headers[split]}"
                )
            rows = list(reader)
        try:
            ids = [int(row["community_id"]) for row in rows]
        except (TypeError, ValueError) as error:
            raise SummaryError(
                f"Split CSV contains a non-integer community_id: {split_path}"
            ) from error
        if len(ids) != len(set(ids)):
            raise SummaryError(f"Split CSV contains duplicate graph IDs: {split_path}")
        invalid = [graph_id for graph_id in ids if not 0 <= graph_id < len(adjs)]
        if invalid:
            raise SummaryError(
                f"Split CSV contains out-of-range graph IDs: {split_path}: {invalid}"
            )
        split_ids[split] = ids

    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    intersections = {
        "train_val": sorted(split_sets["train"] & split_sets["val"]),
        "train_test": sorted(split_sets["train"] & split_sets["test"]),
        "val_test": sorted(split_sets["val"] & split_sets["test"]),
    }
    if any(intersections.values()):
        raise SummaryError(f"r=1 split graph IDs overlap: {intersections}")
    all_split_ids = set().union(*split_sets.values())
    expected_ids = set(range(len(adjs)))
    if all_split_ids != expected_ids:
        raise SummaryError(
            "r=1 split IDs do not cover all raw graphs exactly: "
            f"missing={sorted(expected_ids - all_split_ids)}, "
            f"extra={sorted(all_split_ids - expected_ids)}"
        )

    structures = {
        "sbm_200_type": type(payload).__name__,
        "sbm_200_item_count": len(payload),
        "adjacency_container_type": type(adjs).__name__,
        "adjacency_count": len(adjs),
        "adjacency_item_type": type(adjs[0]).__name__ if adjs else None,
        "n_nodes_container_type": type(n_nodes).__name__,
        "n_max": int(n_max),
        "partition_container_type": type(partitions).__name__,
        "partition_count": len(partitions),
        "partition_item_type": type(partitions[0]).__name__ if partitions else None,
        "community_item_types": sorted(
            {
                type(community).__name__
                for partition in partitions
                for community in partition
            }
        ),
        "split_csv_headers": split_headers,
        "split_counts": {split: len(ids) for split, ids in split_ids.items()},
        "split_intersections": intersections,
    }
    return {
        "raw_dir": raw_dir,
        "paths": {
            "sbm_200": adjacency_path,
            "partitions": partition_path,
            **{
                split: raw_dir / f"{split}_r1.csv"
                for split in ("train", "val", "test")
            },
        },
        "structures": structures,
        "adjs": adjs,
        "n_nodes": n_nodes,
        "partitions": partitions,
        "split_ids": split_ids,
    }


def _real_h2_row(
    split: str,
    original_graph_index: int,
    graph: nx.Graph,
) -> dict[str, Any]:
    stats = graph_stats(graph)
    connected = nx.is_connected(graph) if stats["nodes"] > 0 else None
    isolated_count = nx.number_of_isolates(graph)
    return {
        "split": split,
        "original_graph_index": original_graph_index,
        "h2_nodes": stats["nodes"],
        "h2_edges": stats["edges"],
        "h2_density": stats["density"],
        "is_complete": stats["is_complete"],
        "complete_applicable": stats["complete_applicable"],
        "is_connected": connected,
        "has_isolated_h2_nodes": isolated_count > 0,
        "isolated_h2_node_count": isolated_count,
        "connected_components": stats["components"],
        "component_sizes": stats["component_sizes"],
    }


def summarize_real_h2_split(
    split: str,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    node_values = [int(row["h2_nodes"]) for row in rows]
    edge_values = [int(row["h2_edges"]) for row in rows]
    density_values = [float(row["h2_density"]) for row in rows]
    complete_eligible = [row for row in rows if row["complete_applicable"]]
    complete_count = sum(row["is_complete"] is True for row in complete_eligible)
    connected_known = [row for row in rows if row["is_connected"] is not None]
    connected_count = sum(row["is_connected"] is True for row in connected_known)
    isolated_count = sum(row["has_isolated_h2_nodes"] is True for row in rows)

    by_node_count: dict[str, Any] = {}
    for node_count in sorted(set(node_values)):
        group = [row for row in rows if row["h2_nodes"] == node_count]
        group_complete = sum(row["is_complete"] is True for row in group)
        by_node_count[str(node_count)] = {
            "graph_count": len(group),
            "h2_edges": _numeric_summary(
                [int(row["h2_edges"]) for row in group]
            ),
            "h2_edge_distribution": _distribution(
                int(row["h2_edges"]) for row in group
            ),
            "h2_density": _numeric_summary(
                [float(row["h2_density"]) for row in group]
            ),
            "complete_graph_count": group_complete,
            "complete_graph_ratio": _ratio(group_complete, len(group)),
        }

    return {
        "split": split,
        "graph_count": len(rows),
        "h2_nodes": _numeric_summary(node_values),
        "h2_node_distribution": _distribution(node_values),
        "h2_edges": _numeric_summary(edge_values),
        "h2_edge_distribution": _distribution(edge_values),
        "h2_density": _numeric_summary(density_values),
        "complete_graph_count": complete_count,
        "complete_graph_ratio": _ratio(complete_count, len(complete_eligible)),
        "complete_graph_eligible_count": len(complete_eligible),
        "connected_graph_count": connected_count,
        "connected_graph_ratio": _ratio(connected_count, len(connected_known)),
        "connected_graph_known_count": len(connected_known),
        "graphs_with_isolated_h2_nodes": isolated_count,
        "graphs_with_isolated_h2_nodes_ratio": _ratio(isolated_count, len(rows)),
        "connected_components_distribution": _distribution(
            int(row["connected_components"]) for row in rows
        ),
        "by_h2_node_count": by_node_count,
        "graphs": list(rows),
    }


def load_generated_h2_comparison(summary_path: Path) -> dict[str, Any]:
    """Read generated H2 fields from an existing interface_summary.json."""
    summary_path = summary_path.expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Generated interface summary JSON is missing: {summary_path}"
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise SummaryError(
            f"Generated interface summary lacks the observed samples list: "
            f"{summary_path}"
        )
    rows = payload["samples"]
    required = (
        "full_nodes",
        "full_edges",
        "full_density",
        "is_complete_full",
        "full_components",
    )
    for index, row in enumerate(rows):
        missing = [key for key in required if key not in row]
        if missing:
            raise SummaryError(
                f"Generated H2 sample {index} lacks fields {missing}: {summary_path}"
            )
    complete_eligible = [
        row for row in rows if row["is_complete_full"] is not None
    ]
    connected = sum(int(row["full_components"]) == 1 for row in rows)
    return {
        "label": payload.get("run_name") or summary_path.parent.name,
        "summary_path": str(summary_path),
        "graph_count": len(rows),
        "mean_h2_nodes": _mean(row["full_nodes"] for row in rows),
        "mean_h2_edges": _mean(row["full_edges"] for row in rows),
        "mean_h2_density": _mean(row["full_density"] for row in rows),
        "complete_graph_ratio": _ratio(
            sum(row["is_complete_full"] is True for row in complete_eligible),
            len(complete_eligible),
        ),
        "connected_graph_ratio": _ratio(connected, len(rows)),
        "by_h2_node_count": {
            str(node_count): {
                "graph_count": len(group),
                "mean_h2_edges": _mean(row["full_edges"] for row in group),
                "mean_h2_density": _mean(row["full_density"] for row in group),
                "complete_graph_ratio": _ratio(
                    sum(row["is_complete_full"] is True for row in group),
                    len(group),
                ),
            }
            for node_count in sorted({int(row["full_nodes"]) for row in rows})
            for group in [
                [row for row in rows if int(row["full_nodes"]) == node_count]
            ]
        },
    }


def build_real_generated_comparison(
    splits: dict[str, dict[str, Any]],
    generated: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for split in ("train", "val", "test"):
        split_summary = splits[split]
        rows.append(
            {
                "data": f"{split} real H2",
                "graph_count": split_summary["graph_count"],
                "mean_h2_nodes": split_summary["h2_nodes"]["mean"],
                "mean_h2_edges": split_summary["h2_edges"]["mean"],
                "mean_h2_density": split_summary["h2_density"]["mean"],
                "complete_graph_ratio": split_summary["complete_graph_ratio"],
                "connected_graph_ratio": split_summary["connected_graph_ratio"],
            }
        )
    rows.append(
        {
            "data": f"{generated['label']} generated H2",
            "graph_count": generated["graph_count"],
            "mean_h2_nodes": generated["mean_h2_nodes"],
            "mean_h2_edges": generated["mean_h2_edges"],
            "mean_h2_density": generated["mean_h2_density"],
            "complete_graph_ratio": generated["complete_graph_ratio"],
            "connected_graph_ratio": generated["connected_graph_ratio"],
        }
    )

    near_complete_threshold = 0.95
    material_gap_threshold = 0.10
    train_ratio = splits["train"]["complete_graph_ratio"]
    test_ratio = splits["test"]["complete_graph_ratio"]
    generated_ratio = generated["complete_graph_ratio"]
    train_gap = generated_ratio - train_ratio
    test_gap = generated_ratio - test_ratio
    if train_ratio >= near_complete_threshold and test_ratio >= near_complete_threshold:
        classification = "A"
        judgment = (
            "Real train and test H2 graphs are both near 100% complete under the "
            "declared >=95% criterion; the generated complete-graph rate may match "
            "the real H2 structure."
        )
    elif max(train_gap, test_gap) >= material_gap_threshold:
        classification = "B"
        judgment = (
            "At least one real train/test complete-graph ratio is >=10 percentage "
            "points below the generated rate; Beta-H2 may have an all-edge collapse "
            "or a threshold/prior problem."
        )
    else:
        classification = "MIXED"
        judgment = (
            "The complete-graph comparison is intermediate under the declared "
            "criteria; use the reported node-conditioned edge and density "
            "distributions rather than density alone."
        )
    return {
        "rows": rows,
        "judgment": {
            "classification": classification,
            "text": judgment,
            "near_complete_threshold": near_complete_threshold,
            "material_complete_ratio_gap_threshold": material_gap_threshold,
            "generated_minus_train_complete_ratio": train_gap,
            "generated_minus_test_complete_ratio": test_gap,
        },
        "node_conditioned_real": {
            split: {
                node_count: splits[split]["by_h2_node_count"].get(str(node_count))
                for node_count in (3, 4, 5)
            }
            for split in ("train", "val", "test")
        },
        "node_conditioned_generated": {
            str(node_count): generated["by_h2_node_count"].get(str(node_count))
            for node_count in (3, 4, 5)
        },
    }


def summarize_real_sbm_r1(
    raw_dir: Path,
    generated_summary_path: Path,
) -> dict[str, Any]:
    inputs = load_real_sbm_r1_inputs(raw_dir)
    adjs = inputs["adjs"]
    n_nodes = inputs["n_nodes"]
    partitions = inputs["partitions"]

    meta_graphs: list[nx.Graph] = []
    global_offset = 0
    for graph_index, (adjacency, declared_nodes, partition) in enumerate(
        zip(adjs, n_nodes, partitions)
    ):
        array = _adjacency_numpy(adjacency)
        if int(declared_nodes) != int(array.shape[0]):
            raise SummaryError(
                f"Raw graph {graph_index} n_nodes/adjacency mismatch: "
                f"{declared_nodes} != {array.shape[0]}"
            )
        meta_graphs.append(
            build_h2_meta_graph(
                array,
                partition,
                global_node_offset=global_offset,
            )
        )
        global_offset += int(declared_nodes)

    all_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        rows = [
            _real_h2_row(split, graph_id, meta_graphs[graph_id])
            for graph_id in inputs["split_ids"][split]
        ]
        all_rows.extend(rows)
        split_summaries[split] = summarize_real_h2_split(split, rows)

    generated = load_generated_h2_comparison(generated_summary_path)
    comparison = build_real_generated_comparison(split_summaries, generated)
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now().astimezone().isoformat(),
        "input_policy": "read-only",
        "louvain_rerun": False,
        "h2_definition": {
            "partition_source": "saved sbm_partitions_r1.pkl only",
            "community_node_policy": "every saved community is an H2 node",
            "edge_policy": (
                "one undirected H2 edge iff at least one original edge joins "
                "two different saved communities"
            ),
            "self_loops": False,
            "reverse_edge_duplicates": False,
            "std_definition": "population standard deviation (ddof=0)",
        },
        "source_files": {
            key: str(path) for key, path in inputs["paths"].items()
        },
        "observed_input_structures": inputs["structures"],
        "splits": split_summaries,
        "generated_h2": generated,
        "comparison": comparison,
        "graphs": all_rows,
    }


def summarize_run(run_name: str, run_dir: Path) -> dict[str, Any]:
    """Load one run and construct all per-sample and aggregate records."""
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    sample_rows: list[dict[str, Any]] = []
    h1_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    audit_structures: dict[str, dict[str, list[str]]] = {}

    for sample_id, sample_dir in discover_samples(run_dir):
        audit = read_audit(sample_dir)
        audit_sample_id = audit_value(
            audit, ("sample_id",), warnings, f"{sample_dir.name}"
        )
        if audit_sample_id is not None and int(audit_sample_id) != sample_id:
            raise SummaryError(
                f"Directory/audit sample ID mismatch in {sample_dir}: "
                f"{sample_id} != {audit_sample_id}"
            )

        assembly = audit.get("assembly")
        audit_structures[sample_dir.name] = {
            "top_level_keys": sorted(audit.keys()),
            "assembly_keys": sorted(assembly.keys())
            if isinstance(assembly, dict)
            else [],
        }

        h2_full = load_graph(sample_dir / "h2_full.pkl")
        h2_used = load_graph(sample_dir / "h2_used.pkl")
        h2 = compare_graphs(h2_full, h2_used)
        sample_row = {
            "run": run_name,
            "sample_id": sample_id,
            **h2,
            "glue_added_edges": audit_value(
                audit,
                ("assembly", "glue_added_edge_count"),
                warnings,
                sample_dir.name,
            ),
            "final_nodes": audit_value(
                audit, ("assembly", "final_nodes"), warnings, sample_dir.name
            ),
            "final_edges": audit_value(
                audit, ("assembly", "final_edges"), warnings, sample_dir.name
            ),
            "final_components": audit_value(
                audit, ("assembly", "final_components"), warnings, sample_dir.name
            ),
            "all_assertions_passed": audit_value(
                audit, ("all_assertions_passed",), warnings, sample_dir.name
            ),
        }
        final_components = sample_row["final_components"]
        sample_row["final_connected"] = (
            final_components == 1 if final_components is not None else None
        )
        sample_rows.append(sample_row)

        for community_id, full_path, used_path in pair_h1_files(sample_dir):
            comparison = compare_graphs(load_graph(full_path), load_graph(used_path))
            h1_rows.append(
                {
                    "run": run_name,
                    "sample_id": sample_id,
                    "community_id": community_id,
                    **comparison,
                }
            )

    h2_lcc_count = sum(bool(row["lcc_triggered"]) for row in sample_rows)
    h1_lcc_count = sum(bool(row["lcc_triggered"]) for row in h1_rows)
    full_complete_eligible = [
        row for row in sample_rows if row["complete_full_applicable"]
    ]
    used_complete_eligible = [
        row for row in sample_rows if row["complete_used_applicable"]
    ]
    full_complete_count = sum(
        row["is_complete_full"] is True for row in full_complete_eligible
    )
    used_complete_count = sum(
        row["is_complete_used"] is True for row in used_complete_eligible
    )
    final_connected_known = [
        row for row in sample_rows if row["final_connected"] is not None
    ]
    final_connected_count = sum(
        row["final_connected"] is True for row in final_connected_known
    )

    aggregate = {
        "sample_count": len(sample_rows),
        "h2_lcc_triggered_sample_count": h2_lcc_count,
        "h2_lcc_triggered_ratio": _ratio(h2_lcc_count, len(sample_rows)),
        "h2_removed_nodes_total": sum(row["removed_nodes"] for row in sample_rows),
        "h2_removed_edges_total": sum(row["removed_edges"] for row in sample_rows),
        "h2_full_density_mean": _mean(row["full_density"] for row in sample_rows),
        "h2_used_density_mean": _mean(row["used_density"] for row in sample_rows),
        "h2_complete_full_count": full_complete_count,
        "h2_complete_full_ratio": _ratio(
            full_complete_count, len(full_complete_eligible)
        ),
        "h2_complete_full_eligible_count": len(full_complete_eligible),
        "h2_complete_full_n_le_1_count": len(sample_rows)
        - len(full_complete_eligible),
        "h2_complete_used_count": used_complete_count,
        "h2_complete_used_ratio": _ratio(
            used_complete_count, len(used_complete_eligible)
        ),
        "h2_complete_used_eligible_count": len(used_complete_eligible),
        "h2_complete_used_n_le_1_count": len(sample_rows)
        - len(used_complete_eligible),
        "h2_full_nodes_distribution": _distribution(
            row["full_nodes"] for row in sample_rows
        ),
        "h2_used_nodes_distribution": _distribution(
            row["used_nodes"] for row in sample_rows
        ),
        "h2_full_edges_distribution": _distribution(
            row["full_edges"] for row in sample_rows
        ),
        "h2_used_edges_distribution": _distribution(
            row["used_edges"] for row in sample_rows
        ),
        "h1_graph_count": len(h1_rows),
        "h1_lcc_triggered_graph_count": h1_lcc_count,
        "h1_lcc_triggered_ratio": _ratio(h1_lcc_count, len(h1_rows)),
        "h1_removed_nodes_total": sum(row["removed_nodes"] for row in h1_rows),
        "h1_removed_edges_total": sum(row["removed_edges"] for row in h1_rows),
        "glue_added_edges_total": _sum_known(
            row["glue_added_edges"] for row in sample_rows
        ),
        "final_connected_graph_count": final_connected_count,
        "final_connected_graph_ratio": _ratio(
            final_connected_count, len(final_connected_known)
        ),
        "final_connected_graph_known_count": len(final_connected_known),
        "final_graphs": [
            {
                "sample_id": row["sample_id"],
                "nodes": row["final_nodes"],
                "edges": row["final_edges"],
                "glue_added_edges": row["glue_added_edges"],
                "components": row["final_components"],
                "connected": row["final_connected"],
            }
            for row in sample_rows
        ],
    }

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "input_policy": "read-only",
        "audit_json_structures": audit_structures,
        "warnings": warnings,
        "summary": aggregate,
        "samples": sample_rows,
        "h1_communities": h1_rows,
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise SummaryError(f"Refusing to write a CSV with no rows: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _display(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def render_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "HiGGs Interface Ablation Summary",
        f"Run: {report['run_name']}",
        f"Path: {report['run_dir']}",
        "",
        f"Samples: {summary['sample_count']}",
        (
            "H2 LCC triggered: "
            f"{summary['h2_lcc_triggered_sample_count']} "
            f"({_display(summary['h2_lcc_triggered_ratio'])})"
        ),
        f"H2 removed nodes: {summary['h2_removed_nodes_total']}",
        f"H2 removed edges: {summary['h2_removed_edges_total']}",
        f"H2 full mean density: {_display(summary['h2_full_density_mean'])}",
        f"H2 used mean density: {_display(summary['h2_used_density_mean'])}",
        (
            "H2 complete full graphs (n>=2): "
            f"{summary['h2_complete_full_count']}/"
            f"{summary['h2_complete_full_eligible_count']} "
            f"({_display(summary['h2_complete_full_ratio'])})"
        ),
        (
            "H2 full node distribution: "
            f"{_display(summary['h2_full_nodes_distribution'])}"
        ),
        (
            "H2 full edge distribution: "
            f"{_display(summary['h2_full_edges_distribution'])}"
        ),
        f"H1 graphs: {summary['h1_graph_count']}",
        (
            "H1 LCC triggered: "
            f"{summary['h1_lcc_triggered_graph_count']} "
            f"({_display(summary['h1_lcc_triggered_ratio'])})"
        ),
        f"H1 removed nodes: {summary['h1_removed_nodes_total']}",
        f"H1 removed edges: {summary['h1_removed_edges_total']}",
        f"Glue added edges total: {_display(summary['glue_added_edges_total'])}",
        (
            "Final connected graphs: "
            f"{summary['final_connected_graph_count']}/"
            f"{summary['final_connected_graph_known_count']} "
            f"({_display(summary['final_connected_graph_ratio'])})"
        ),
        "",
        "Per-sample final graphs:",
        "sample_id\tnodes\tedges\tglue_added\tcomponents\tconnected",
    ]
    for row in summary["final_graphs"]:
        lines.append(
            "\t".join(
                _display(row[key])
                for key in (
                    "sample_id",
                    "nodes",
                    "edges",
                    "glue_added_edges",
                    "components",
                    "connected",
                )
            )
        )
    lines.extend(["", f"Warnings: {len(report['warnings'])}"])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any]) -> list[Path]:
    run_dir = Path(report["run_dir"])
    paths = [run_dir / name for name in OUTPUT_NAMES]
    write_csv(paths[0], report["samples"])
    write_csv(paths[1], report["h1_communities"])
    with paths[2].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with paths[3].open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_text(report))
    return paths


def _comparison_table_lines(comparison: dict[str, Any]) -> list[str]:
    headers = (
        "Data",
        "Graphs",
        "Mean H2 nodes",
        "Mean H2 edges",
        "Mean density",
        "Complete ratio",
        "Connected ratio",
    )
    rows = [
        (
            row["data"],
            _display(row["graph_count"]),
            _display(row["mean_h2_nodes"]),
            _display(row["mean_h2_edges"]),
            _display(row["mean_h2_density"]),
            _display(row["complete_graph_ratio"]),
            _display(row["connected_graph_ratio"]),
        )
        for row in comparison["rows"]
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    return [
        " | ".join(
            header.ljust(widths[index]) for index, header in enumerate(headers)
        ),
        "-+-".join("-" * width for width in widths),
        *[
            " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            for row in rows
        ],
    ]


def render_real_h2_text(report: dict[str, Any]) -> str:
    lines = [
        "Real SBM r=1 H2 Meta-Graph Summary",
        f"Generated at: {report['generated_at']}",
        "Louvain rerun: false",
        "Standard deviation: population (ddof=0)",
        "",
    ]
    for split in ("train", "val", "test"):
        summary = report["splits"][split]
        lines.extend(
            [
                f"[{split}]",
                f"Graphs: {summary['graph_count']}",
                (
                    "H2 nodes mean/std/min/max: "
                    f"{_display(summary['h2_nodes']['mean'])} / "
                    f"{_display(summary['h2_nodes']['std'])} / "
                    f"{_display(summary['h2_nodes']['min'])} / "
                    f"{_display(summary['h2_nodes']['max'])}"
                ),
                (
                    "H2 node distribution: "
                    f"{_display(summary['h2_node_distribution'])}"
                ),
                (
                    "H2 edges mean/std/min/max: "
                    f"{_display(summary['h2_edges']['mean'])} / "
                    f"{_display(summary['h2_edges']['std'])} / "
                    f"{_display(summary['h2_edges']['min'])} / "
                    f"{_display(summary['h2_edges']['max'])}"
                ),
                (
                    "H2 edge distribution: "
                    f"{_display(summary['h2_edge_distribution'])}"
                ),
                (
                    "H2 density mean/std/min/max: "
                    f"{_display(summary['h2_density']['mean'])} / "
                    f"{_display(summary['h2_density']['std'])} / "
                    f"{_display(summary['h2_density']['min'])} / "
                    f"{_display(summary['h2_density']['max'])}"
                ),
                (
                    "Complete graphs: "
                    f"{summary['complete_graph_count']}/"
                    f"{summary['complete_graph_eligible_count']} "
                    f"({_display(summary['complete_graph_ratio'])})"
                ),
                (
                    "Connected graphs: "
                    f"{summary['connected_graph_count']}/"
                    f"{summary['connected_graph_known_count']} "
                    f"({_display(summary['connected_graph_ratio'])})"
                ),
                (
                    "Graphs with isolated H2 nodes: "
                    f"{summary['graphs_with_isolated_h2_nodes']}/"
                    f"{summary['graph_count']} "
                    f"({_display(summary['graphs_with_isolated_h2_nodes_ratio'])})"
                ),
                (
                    "Connected-components distribution: "
                    f"{_display(summary['connected_components_distribution'])}"
                ),
                "Node-conditioned n=3/4/5:",
            ]
        )
        for node_count in (3, 4, 5):
            conditioned = summary["by_h2_node_count"].get(str(node_count))
            if conditioned is None:
                lines.append(f"  n={node_count}: null")
            else:
                lines.append(
                    f"  n={node_count}: graphs={conditioned['graph_count']}, "
                    f"mean_density={_display(conditioned['h2_density']['mean'])}, "
                    f"complete_ratio={_display(conditioned['complete_graph_ratio'])}, "
                    f"edge_distribution="
                    f"{_display(conditioned['h2_edge_distribution'])}"
                )
        lines.append("")

    lines.extend(
        [
            "[Real vs generated comparison]",
            *_comparison_table_lines(report["comparison"]),
            "",
            (
                "Judgment: "
                f"{report['comparison']['judgment']['classification']} - "
                f"{report['comparison']['judgment']['text']}"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_real_h2_report(report: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir = output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(
            f"Real H2 output directory does not exist: {output_dir}"
        )
    paths = [output_dir / name for name in REAL_H2_OUTPUT_NAMES]
    with paths[0].open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with paths[1].open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_real_h2_text(report))
    write_csv(paths[2], report["graphs"])
    return paths


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--run must use NAME=PATH syntax, received: {spec!r}"
        )
    name, raw_path = spec.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            f"--run must contain both a non-empty name and path: {spec!r}"
        )
    return name.strip(), Path(raw_path.strip())


def print_summary_table(reports: Sequence[dict[str, Any]]) -> None:
    headers = (
        "Run",
        "Samples",
        "H2 LCC",
        "H2 rm nodes",
        "H2 complete",
        "H1 LCC",
        "Glue edges",
        "Final connected",
    )
    rows: list[tuple[str, ...]] = []
    for report in reports:
        summary = report["summary"]
        rows.append(
            (
                report["run_name"],
                str(summary["sample_count"]),
                (
                    f"{summary['h2_lcc_triggered_sample_count']}/"
                    f"{summary['sample_count']}"
                ),
                str(summary["h2_removed_nodes_total"]),
                (
                    f"{summary['h2_complete_full_count']}/"
                    f"{summary['h2_complete_full_eligible_count']}"
                ),
                (
                    f"{summary['h1_lcc_triggered_graph_count']}/"
                    f"{summary['h1_graph_count']}"
                ),
                _display(summary["glue_added_edges_total"]),
                (
                    f"{summary['final_connected_graph_count']}/"
                    f"{summary['final_connected_graph_known_count']}"
                ),
            )
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(
        " | ".join(
            header.ljust(widths[index]) for index, header in enumerate(headers)
        )
    )
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(
            " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize saved HiGGs interface-ablation intermediates."
    )
    parser.add_argument(
        "--run",
        action="append",
        metavar="NAME=PATH",
        help="Run label and run directory. Repeat this option for multiple runs.",
    )
    parser.add_argument(
        "--real-sbm-r1",
        action="store_true",
        help=(
            "Also summarize real SBM r=1 H2 meta-graphs from the saved raw "
            "partition; this never reruns Louvain."
        ),
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--sbm-raw-dir",
        type=Path,
        default=project_root / "data" / "sbm" / "raw",
        help="Directory containing sbm_200.pt, saved r=1 partition, and split CSVs.",
    )
    parser.add_argument(
        "--real-output-dir",
        type=Path,
        default=project_root / "data" / "sbm",
        help="Existing directory for real_h2_r1_summary outputs.",
    )
    parser.add_argument(
        "--compare-summary",
        type=Path,
        help=(
            "Existing generated interface_summary.json used for the direct "
            "real/generated H2 comparison. If omitted, exactly one --run is "
            "required and its output is used."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run and not args.real_sbm_r1:
        parser.error("at least one --run or --real-sbm-r1 is required")
    run_specs = [parse_run_spec(spec) for spec in (args.run or [])]
    run_names = [name for name, _ in run_specs]
    duplicates = sorted(name for name, count in Counter(run_names).items() if count > 1)
    if duplicates:
        raise SummaryError(f"Duplicate --run names are not allowed: {duplicates}")

    reports: list[dict[str, Any]] = []
    for run_name, run_dir in run_specs:
        report = summarize_run(run_name, run_dir)
        output_paths = write_report(report)
        reports.append(report)
        for warning in report["warnings"]:
            print(f"WARNING: {warning}", file=sys.stderr)
        print(f"[{run_name}] wrote:")
        for path in output_paths:
            print(f"  {path}")
    if reports:
        print()
        print_summary_table(reports)

    if args.real_sbm_r1:
        if args.compare_summary is not None:
            compare_summary = args.compare_summary
        elif len(run_specs) == 1:
            compare_summary = run_specs[0][1] / "interface_summary.json"
        else:
            parser.error(
                "--real-sbm-r1 requires --compare-summary when the number of "
                "--run arguments is not exactly one"
            )
        real_report = summarize_real_sbm_r1(args.sbm_raw_dir, compare_summary)
        real_paths = write_real_h2_report(real_report, args.real_output_dir)
        print("\n[Real SBM r=1 H2] wrote:")
        for path in real_paths:
            print(f"  {path}")
        print()
        print("\n".join(_comparison_table_lines(real_report["comparison"])))
        judgment = real_report["comparison"]["judgment"]
        print(f"\nJudgment: {judgment['classification']} - {judgment['text']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, SummaryError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
