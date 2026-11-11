#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import statistics
from pathlib import Path
from typing import Any

import networkx as nx


_SAMPLE_RE = re.compile(r"sampled_graph_(\d+)\.pkl$")


def sample_index(path: Path) -> int:
    match = _SAMPLE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"无法解析样本编号：{path.name}")
    return int(match.group(1))


def load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as f:
        obj: Any = pickle.load(f)

    if isinstance(obj, (nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"{path.name} 是 MultiGraph/MultiDiGraph，不符合要求。")
    if isinstance(obj, nx.DiGraph):
        raise TypeError(f"{path.name} 是有向图，不符合要求。")
    if not isinstance(obj, nx.Graph):
        raise TypeError(f"{path.name} 不是 networkx.Graph：{type(obj)!r}")

    graph = nx.Graph(obj)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    if graph.number_of_nodes() == 0:
        raise ValueError(f"{path.name} 是空图。")
    return graph


def lcc(graph: nx.Graph) -> nx.Graph:
    if nx.is_connected(graph):
        return graph
    nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(nodes).copy()


def summarize(samples_dir: Path, expected_count: int, resolution: float, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted(samples_dir.glob("sampled_graph_*.pkl"), key=sample_index)
    if len(paths) != expected_count:
        raise RuntimeError(f"预期 {expected_count} 张图，实际找到 {len(paths)} 张：{samples_dir}")

    actual_indices = [sample_index(p) for p in paths]
    expected_indices = list(range(expected_count))
    if actual_indices != expected_indices:
        raise RuntimeError(f"样本编号不连续。实际：{actual_indices}")

    rows: list[dict[str, Any]] = []
    all_community_sizes: list[int] = []

    for path in paths:
        idx = sample_index(path)
        graph = load_graph(path)
        diameter_graph = lcc(graph)

        communities = nx.community.louvain_communities(
            graph,
            resolution=resolution,
            seed=seed,
            weight="weight",
        )
        community_sizes = [len(c) for c in communities]
        all_community_sizes.extend(community_sizes)

        row = {
            "sample_id": idx,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "components": nx.number_connected_components(graph),
            "diameter_lcc": nx.diameter(diameter_graph),
            "clustering": nx.average_clustering(graph),
            "density": nx.density(graph),
            "transitivity": nx.transitivity(graph),
            "num_communities": len(communities),
            "median_community_size": statistics.median(community_sizes) if community_sizes else None,
        }
        rows.append(row)

    def mean_of(key: str) -> float:
        return float(statistics.fmean(float(r[key]) for r in rows))

    summary = {
        "sample_count": len(rows),
        "resolution": resolution,
        "louvain_seed": seed,
        "connected_graphs": sum(int(r["components"] == 1) for r in rows),
        "nodes_mean": mean_of("nodes"),
        "edges_mean": mean_of("edges"),
        "diameter_mean_lcc": mean_of("diameter_lcc"),
        "clustering_mean": mean_of("clustering"),
        "density_mean": mean_of("density"),
        "transitivity_mean": mean_of("transitivity"),
        "num_communities_mean": mean_of("num_communities"),
        "per_graph_median_community_size_mean": mean_of("median_community_size"),
        "global_median_community_size": float(statistics.median(all_community_sizes)) if all_community_sizes else None,
        "nodes_min": min(r["nodes"] for r in rows),
        "nodes_max": max(r["nodes"] for r in rows),
        "edges_min": min(r["edges"] for r in rows),
        "edges_max": max(r["edges"] for r in rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算 C3 40 张 SBM 生成图的整图结构统计。"
    )
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, summary = summarize(
        samples_dir=args.samples_dir,
        expected_count=args.expected_count,
        resolution=args.resolution,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "c3_structure_stats_samples.csv"
    json_path = args.output_dir / "c3_structure_stats_summary.json"
    txt_path = args.output_dir / "c3_structure_stats_summary.txt"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "C3 40张 SBM 整图统计",
        "=" * 48,
        f"Samples                  : {summary['sample_count']}",
        f"Connected graphs         : {summary['connected_graphs']}/{summary['sample_count']}",
        f"Nodes mean               : {summary['nodes_mean']:.6f}",
        f"Edges mean               : {summary['edges_mean']:.6f}",
        f"Diameter mean (LCC)      : {summary['diameter_mean_lcc']:.6f}",
        f"Clustering mean          : {summary['clustering_mean']:.6f}",
        f"Density mean             : {summary['density_mean']:.6f}",
        f"Transitivity mean        : {summary['transitivity_mean']:.6f}",
        f"N_C mean                 : {summary['num_communities_mean']:.6f}",
        f"|C|_med mean per graph   : {summary['per_graph_median_community_size_mean']:.6f}",
        f"|C|_med global           : {summary['global_median_community_size']:.6f}",
    ]
    text = "\n".join(lines) + "\n"
    txt_path.write_text(text, encoding="utf-8")

    print(text)
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"TXT : {txt_path}")


if __name__ == "__main__":
    main()
