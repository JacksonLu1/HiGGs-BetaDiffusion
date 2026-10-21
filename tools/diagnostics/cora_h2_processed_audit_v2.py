#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data.separate import separate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--real-meta-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def describe(values):
    array = np.asarray(values, dtype=float)

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


def slice_count(value):
    if torch.is_tensor(value):
        return int(value.numel()) - 1

    return len(value) - 1


def load_collated_file(path):
    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(obj, tuple):
        raise TypeError(
            f"{path.name}: expected tuple, got {type(obj)}"
        )

    if len(obj) < 2:
        raise ValueError(
            f"{path.name}: tuple length is only {len(obj)}"
        )

    collated = obj[0]
    slices = obj[1]

    if isinstance(collated, dict):
        collated = Data.from_dict(collated)

    if not isinstance(slices, dict):
        raise TypeError(
            f"{path.name}: slices is not a dictionary"
        )

    preferred_keys = [
        "x",
        "edge_index",
        "edge_attr",
        "y",
    ]

    graph_count = None

    for key in preferred_keys:
        if key in slices:
            graph_count = slice_count(slices[key])
            break

    if graph_count is None:
        raise RuntimeError(
            f"{path.name}: cannot infer graph count; "
            f"slice keys={list(slices)}"
        )

    all_slice_counts = {
        key: slice_count(value)
        for key, value in slices.items()
    }

    print()
    print("Loading:", path)
    print("  tuple length :", len(obj))
    print("  collated type:", type(collated).__name__)
    print("  slice keys   :", list(slices))
    print("  slice counts :", all_slice_counts)
    print("  graph count  :", graph_count)

    graphs = []

    for index in range(graph_count):
        data = separate(
            cls=collated.__class__,
            batch=collated,
            idx=index,
            slice_dict=slices,
            inc_dict=None,
            decrement=False,
        )

        graphs.append(data)

    return graphs, all_slice_counts


def get_split(path):
    match = re.search(
        r"proc_(train|val|test)_h2",
        path.name,
    )

    if not match:
        return "unknown"

    return match.group(1)


def y_value(data):
    y = getattr(data, "y", None)

    if y is None:
        return "<none>"

    values = (
        torch.as_tensor(y)
        .detach()
        .cpu()
        .reshape(-1)
        .tolist()
    )

    return str(
        tuple(round(float(value), 8) for value in values)
    )


def inspect_graph(split, graph_index, data):
    x = getattr(data, "x", None)

    if x is not None:
        node_count = int(x.size(0))
    else:
        node_count = int(data.num_nodes)

    edge_index = (
        data.edge_index
        .detach()
        .cpu()
        .long()
    )

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))

    if edge_index.numel() > 0:
        for left, right in edge_index.t().tolist():
            left = int(left)
            right = int(right)

            if left == right:
                continue

            if not (
                0 <= left < node_count
                and 0 <= right < node_count
            ):
                raise ValueError(
                    f"{split}[{graph_index}]: "
                    f"edge ({left}, {right}) outside "
                    f"node range 0..{node_count - 1}"
                )

            graph.add_edge(left, right)

    degrees = np.asarray(
        [degree for _, degree in graph.degree()],
        dtype=float,
    )

    components = list(nx.connected_components(graph))

    largest_component = max(
        (
            len(component)
            for component in components
        ),
        default=0,
    )

    labels = []

    if x is not None:
        x = torch.as_tensor(x).detach().cpu()

        if x.ndim == 2 and x.size(-1) > 1:
            labels = x.argmax(dim=-1).long().tolist()
        else:
            labels = x.reshape(-1).long().tolist()

    return {
        "split": split,
        "graph_index": graph_index,
        "nodes": node_count,
        "edges": graph.number_of_edges(),
        "components": len(components),
        "connected": int(
            nx.is_connected(graph)
            if node_count > 0
            else False
        ),
        "isolates": nx.number_of_isolates(graph),
        "lcc_nodes": largest_component,
        "lcc_fraction": (
            largest_component / node_count
            if node_count
            else 0.0
        ),
        "min_degree": float(
            degrees.min() if len(degrees) else 0.0
        ),
        "median_degree": float(
            np.median(degrees) if len(degrees) else 0.0
        ),
        "mean_degree": float(
            degrees.mean() if len(degrees) else 0.0
        ),
        "max_degree": float(
            degrees.max() if len(degrees) else 0.0
        ),
        "density": float(nx.density(graph)),
        "clustering": float(
            nx.average_clustering(graph)
        ),
        "transitivity": float(
            nx.transitivity(graph)
        ),
        "y": y_value(data),
        "_labels": labels,
    }


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        args.processed_dir.glob(
            "proc_*_h2_r2_majority_max500_bs8.pt"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No processed H2 files in "
            f"{args.processed_dir}"
        )

    rows = []
    node_classes = Counter()
    y_distribution = Counter()
    split_counts = Counter()
    slice_metadata = {}

    for path in files:
        split = get_split(path)

        graphs, slice_counts = load_collated_file(path)

        split_counts[split] += len(graphs)
        slice_metadata[path.name] = slice_counts

        for graph_index, data in enumerate(graphs):
            row = inspect_graph(
                split,
                graph_index,
                data,
            )

            node_classes.update(
                row.pop("_labels")
            )
            y_distribution[row["y"]] += 1
            rows.append(row)

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise RuntimeError("No H2 graphs reconstructed")

    metrics = [
        "nodes",
        "edges",
        "components",
        "isolates",
        "lcc_fraction",
        "min_degree",
        "median_degree",
        "mean_degree",
        "max_degree",
        "density",
        "clustering",
        "transitivity",
    ]

    result = {
        "graph_count": int(len(frame)),
        "split_counts": dict(split_counts),
        "connected_count": int(
            frame["connected"].sum()
        ),
        "connected_rate": float(
            frame["connected"].mean()
        ),
        "graphs_with_isolates": int(
            (frame["isolates"] > 0).sum()
        ),
        "isolate_graph_rate": float(
            (frame["isolates"] > 0).mean()
        ),
        "metrics": {
            metric: describe(frame[metric])
            for metric in metrics
        },
        "y_distribution": dict(y_distribution),
        "node_class_counts": dict(
            sorted(node_classes.items())
        ),
        "slice_metadata": slice_metadata,
    }

    real = pd.read_csv(args.real_meta_csv)

    comparison_metrics = [
        "nodes",
        "edges",
        "density",
        "mean_degree",
        "clustering",
        "transitivity",
    ]

    comparison = {
        "processed_count": int(len(frame)),
        "real_count": int(len(real)),
        "processed_connected_rate": float(
            frame["connected"].mean()
        ),
        "real_connected_rate": float(
            real["connected"].mean()
        ),
        "processed_isolate_graph_rate": float(
            (frame["isolates"] > 0).mean()
        ),
        "real_isolate_graph_rate": float(
            (real["isolates"] > 0).mean()
        ),
    }

    for metric in comparison_metrics:
        processed_mean = float(
            frame[metric].mean()
        )
        real_mean = float(
            real[metric].mean()
        )

        comparison[metric] = {
            "processed_mean": processed_mean,
            "real_mean": real_mean,
            "difference": processed_mean - real_mean,
        }

    result["real_comparison"] = comparison

    csv_path = (
        args.output_dir
        / "cora_h2_processed_graphs.csv"
    )
    json_path = (
        args.output_dir
        / "cora_h2_processed_audit_v2.json"
    )

    frame.to_csv(csv_path, index=False)

    json_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 118)
    print(
        "Cora processed H2 audit "
        "(correctly separated with PyG slices)"
    )
    print("=" * 118)

    print(
        "Split graph counts      :",
        dict(split_counts),
    )
    print(
        "Total graphs            :",
        len(frame),
    )
    print(
        "Connected               :",
        f"{int(frame['connected'].sum())}/"
        f"{len(frame)}",
    )
    print(
        "Connected rate          :",
        frame["connected"].mean(),
    )
    print(
        "Graphs with isolates    :",
        int((frame["isolates"] > 0).sum()),
    )
    print(
        "Isolate-graph rate      :",
        (frame["isolates"] > 0).mean(),
    )

    for metric in metrics:
        print(
            f"{metric:23s}:",
            result["metrics"][metric],
        )

    print(
        "y distribution          :",
        dict(y_distribution),
    )
    print(
        "Node class counts       :",
        dict(sorted(node_classes.items())),
    )

    print()
    print("Processed data vs raw real H2 meta graphs:")
    print(
        "  graph counts    :",
        comparison["processed_count"],
        comparison["real_count"],
    )
    print(
        "  connected rates :",
        comparison["processed_connected_rate"],
        comparison["real_connected_rate"],
    )
    print(
        "  isolate rates   :",
        comparison["processed_isolate_graph_rate"],
        comparison["real_isolate_graph_rate"],
    )

    for metric in comparison_metrics:
        values = comparison[metric]

        print(
            f"  {metric:14s}: "
            f"processed={values['processed_mean']:.12f} "
            f"real={values['real_mean']:.12f} "
            f"diff={values['difference']:.12e}"
        )

    print()
    print("CSV :", csv_path)
    print("JSON:", json_path)
    print("=" * 118)


if __name__ == "__main__":
    main()
