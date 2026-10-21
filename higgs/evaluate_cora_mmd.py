#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import networkx as nx


PAPER_COMPATIBLE_MODE = "mixed"

PAPER_METRIC_MODES = {
    "whole_graph": {
        "degree": "emd",
        "clustering": "emd",
        "eccentricity": "emd",
    },
    "communities": {
        "nodes": "tv",
        "degree": "emd",
        "clustering": "emd",
        "eccentricity": "emd",
        "spectre": "emd",
    },
}

BACKEND_EVALUATOR = Path(r"/data2/wenb/ylu/HiGGs-GBD/HiGGs-main/higgs/evaluate_cora_mmd.py.backend_20260802_143907")


def load_clean_graph(path: Path) -> nx.Graph:
    with path.open("rb") as file:
        obj: Any = pickle.load(file)

    if isinstance(obj, (nx.MultiGraph, nx.MultiDiGraph)):
        graph = nx.Graph(obj)
    elif isinstance(obj, nx.DiGraph):
        graph = nx.Graph(obj.to_undirected())
    elif isinstance(obj, nx.Graph):
        graph = nx.Graph(obj)
    else:
        raise TypeError(
            f"{path} 不是 NetworkX 图，实际类型为 {type(obj)!r}"
        )

    graph.remove_edges_from(list(nx.selfloop_edges(graph)))

    if graph.number_of_nodes() == 0:
        raise ValueError(f"{path} 是空图")

    return graph


def largest_connected_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        raise ValueError("不能对空图计算最大连通分量")

    if nx.is_connected(graph):
        return graph.copy()

    nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(nodes).copy()


def table3_statistics(
    graph: nx.Graph,
    *,
    resolution: float,
    seed: int,
) -> dict[str, Any]:
    # 表 3：所有结构统计都在最大连通分量上计算
    lcc = largest_connected_component(graph)

    communities = nx.community.louvain_communities(
        lcc,
        resolution=resolution,
        seed=seed,
        weight="weight",
    )

    community_sizes = [len(nodes) for nodes in communities]

    return {
        "nodes": int(lcc.number_of_nodes()),
        "edges": int(lcc.number_of_edges()),
        "diameter": int(nx.diameter(lcc)),
        "communities": int(len(communities)),
        "median_community_size": (
            float(statistics.median(community_sizes))
            if community_sizes
            else None
        ),
        "clustering": float(nx.average_clustering(lcc)),
        "density": float(nx.density(lcc)),
        "transitivity": float(nx.transitivity(lcc)),
        "input_full_nodes": int(graph.number_of_nodes()),
        "input_full_edges": int(graph.number_of_edges()),
        "input_connected_components": int(
            nx.number_connected_components(graph)
        ),
        "statistics_graph": "largest_connected_component",
    }


def save_table3(
    output_dir: Path,
    real_stats: dict[str, Any],
    generated_stats: dict[str, Any],
) -> None:
    json_path = output_dir / "cora_table3_statistics.json"
    csv_path = output_dir / "cora_table3_statistics.csv"

    payload = {
        "protocol": (
            "Table 3; largest connected component selected independently "
            "for the real and generated graph"
        ),
        "real": real_stats,
        "generated": generated_stats,
    }

    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metric_names = [
        "nodes",
        "edges",
        "diameter",
        "communities",
        "median_community_size",
        "clustering",
        "density",
        "transitivity",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "real", "generated", "protocol"],
        )
        writer.writeheader()

        for metric in metric_names:
            writer.writerow(
                {
                    "metric": metric,
                    "real": real_stats[metric],
                    "generated": generated_stats[metric],
                    "protocol": "independent LCC",
                }
            )


def find_result_candidate(
    output_dir: Path,
    *,
    extension: str,
    required_terms: tuple[str, ...],
    excluded_name: str,
) -> Path | None:
    candidates: list[Path] = []

    for path in output_dir.glob(f"*{extension}"):
        lower_name = path.name.lower()

        if path.name == excluded_name:
            continue

        if all(term in lower_name for term in required_terms):
            candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def create_result_aliases(output_dir: Path) -> None:
    aliases = [
        (
            "cora_whole_mmd_paper_compatible.csv",
            ".csv",
            ("whole", "mmd"),
        ),
        (
            "cora_whole_mmd_paper_compatible.json",
            ".json",
            ("whole", "mmd"),
        ),
        (
            "cora_communities_mmd_paper_compatible.csv",
            ".csv",
            ("communit", "mmd"),
        ),
        (
            "cora_communities_mmd_paper_compatible.json",
            ".json",
            ("communit", "mmd"),
        ),
    ]

    for target_name, extension, terms in aliases:
        target = output_dir / target_name

        if target.is_file():
            print(f"结果文件已存在：{target}")
            continue

        source = find_result_candidate(
            output_dir,
            extension=extension,
            required_terms=terms,
            excluded_name=target_name,
        )

        if source is None:
            print(
                f"警告：没有找到可复制为 {target_name} 的后端结果文件"
            )
            continue

        shutil.copy2(source, target)
        print(f"结果别名：{source.name} -> {target.name}")


def update_audit(
    output_dir: Path,
    *,
    real_path: Path,
    sample_path: Path,
    resolution: float,
    seed: int,
    real_stats: dict[str, Any],
    generated_stats: dict[str, Any],
) -> None:
    audit_path = output_dir / "cora_mmd_audit.json"

    audit: dict[str, Any] = {}

    if audit_path.is_file():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except Exception:
            audit = {}

    audit.update(
        {
            "paper_compatible_mode": PAPER_COMPATIBLE_MODE,
            "paper_metric_modes": PAPER_METRIC_MODES,
            "real_path": str(real_path.resolve()),
            "sample_path": str(sample_path.resolve()),
            "louvain_resolution": resolution,
            "louvain_seed": seed,
            "whole_graph_mmd_policy": (
                "largest connected component, as implemented by the "
                "existing paper-table backend"
            ),
            "communities_mmd_louvain_input": (
                "complete cleaned graph; no whole-graph LCC truncation "
                "before Louvain"
            ),
            "table3_policy": (
                "real and generated graphs independently reduced to "
                "their largest connected component"
            ),
            "table3_real": real_stats,
            "table3_generated": generated_stats,
            "backend_evaluator": str(BACKEND_EVALUATOR),
        }
    )

    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cora paper-compatible evaluation: mixed MMD kernels, "
            "full cleaned graph for community Louvain, independent LCC "
            "for all Table 3 statistics."
        )
    )

    parser.add_argument("--real-path", type=Path, required=True)
    parser.add_argument("--sample-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["paper", "paper-table", "tv", "emd"],
        default="paper",
    )
    parser.add_argument(
        "--louvain-resolution",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--louvain-seed",
        type=int,
        default=0,
    )

    args, unknown = parser.parse_known_args()

    if not BACKEND_EVALUATOR.is_file():
        raise FileNotFoundError(
            f"找不到备份后端脚本：{BACKEND_EVALUATOR}"
        )

    if not args.real_path.is_file():
        raise FileNotFoundError(f"真实图不存在：{args.real_path}")

    if not args.sample_path.is_file():
        raise FileNotFoundError(f"生成图不存在：{args.sample_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 新的 --mode paper 直接对应旧脚本的 mixed paper-table 协议
    backend_mode = (
        "paper"
        if args.mode in {"paper", "paper-table"}
        else args.mode
    )

    command = [
        sys.executable,
        str(BACKEND_EVALUATOR),
        "--real-path",
        str(args.real_path),
        "--sample-path",
        str(args.sample_path),
        "--output-dir",
        str(args.output_dir),
        "--mode",
        backend_mode,
        "--louvain-resolution",
        str(args.louvain_resolution),
        "--louvain-seed",
        str(args.louvain_seed),
        *unknown,
    ]

    print("=" * 88)
    print(f"PAPER_COMPATIBLE_MODE = {PAPER_COMPATIBLE_MODE}")
    print(f"Backend mode = {backend_mode}")
    print(
        "Whole Graph: Degree/Clustering/Eccentricity = EMD"
    )
    print(
        "Communities: Nodes = TV; "
        "Degree/Clustering/Eccentricity/Spectre = EMD"
    )
    print(
        "Communities Louvain input = complete cleaned graph"
    )
    print(
        "Table 3 input = independent LCC for real and generated graph"
    )
    print("=" * 88)

    subprocess.run(command, check=True)

    real_graph = load_clean_graph(args.real_path)
    generated_graph = load_clean_graph(args.sample_path)

    real_stats = table3_statistics(
        real_graph,
        resolution=args.louvain_resolution,
        seed=args.louvain_seed,
    )
    generated_stats = table3_statistics(
        generated_graph,
        resolution=args.louvain_resolution,
        seed=args.louvain_seed,
    )

    save_table3(
        args.output_dir,
        real_stats,
        generated_stats,
    )

    create_result_aliases(args.output_dir)

    update_audit(
        args.output_dir,
        real_path=args.real_path,
        sample_path=args.sample_path,
        resolution=args.louvain_resolution,
        seed=args.louvain_seed,
        real_stats=real_stats,
        generated_stats=generated_stats,
    )

    print("=" * 88)
    print("Cora Table 3 statistics")
    print(
        f"Real LCC: nodes={real_stats['nodes']} "
        f"edges={real_stats['edges']} "
        f"diameter={real_stats['diameter']}"
    )
    print(
        f"Generated LCC: nodes={generated_stats['nodes']} "
        f"edges={generated_stats['edges']} "
        f"diameter={generated_stats['diameter']}"
    )
    print(f"Output directory: {args.output_dir.resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()
