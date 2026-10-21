import csv
import json
import sys
from pathlib import Path
from statistics import mean

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components

ROOT = Path("/data2/wenb/ylu/HiGGs-GBD/HiGGs-main")
sys.path.insert(0, str(ROOT / "tools/diagnostics"))
sys.path.insert(0, str(ROOT / "dgd"))

import cora_beta_h1_teacher_forced as teacher

base = teacher.base
SEED = 20260815

CHECKPOINTS = {
    "epoch649_validation_best": (
        ROOT
        / "outputs/2026-07-29/21-46-00/checkpoints"
        / "cora_beta_h1_r2_e1000/epoch=649.ckpt"
    ),
    "epoch849_second_best": (
        ROOT
        / "outputs/2026-07-29/21-46-00/checkpoints"
        / "cora_beta_h1_r2_e1000/epoch=849.ckpt"
    ),
    "epoch999_last": (
        ROOT
        / "outputs/2026-07-29/21-46-00/checkpoints"
        / "cora_beta_h1_r2_e1000/last.ckpt"
    ),
}

OUTPUT = (
    ROOT
    / "higgs/outputs"
    / "cora_beta_h1_checkpoint_full_reverse_validation_seed20260815"
)


def average(rows, key):
    return float(mean(row[key] for row in rows))


def graph_from_adjacency(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.int64)

    if adjacency.ndim != 2:
        raise ValueError("Adjacency is not 2-D")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency is not square")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("Adjacency is asymmetric")
    if np.any(np.diag(adjacency) != 0):
        raise ValueError("Adjacency has a self-loop")
    if not np.isin(adjacency, [0, 1]).all():
        raise ValueError("Adjacency is not binary")

    return nx.from_numpy_array(adjacency)


def upstream_lcc(node_types, adjacency):
    node_types = np.asarray(node_types, dtype=np.int64)
    adjacency = np.asarray(adjacency, dtype=np.int64)

    count, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    sizes = np.bincount(labels, minlength=count)
    selected_label = int(np.argmax(sizes))
    indices = np.flatnonzero(labels == selected_label)

    return (
        node_types[indices],
        adjacency[np.ix_(indices, indices)],
    )


def evaluate(model, record, record_id, checkpoint_name):
    real_graph = record["graph"]
    real_classes = np.asarray(
        record["classes"], dtype=np.int64
    )
    condition = int(record["condition_y"])
    n = real_graph.number_of_nodes()

    y = F.one_hot(
        torch.tensor(
            [condition],
            dtype=torch.long,
            device=model.device,
        ),
        num_classes=model.ydim_output,
    ).float()

    # The same endpoint receives the same seed for all checkpoints.
    teacher.seed_everything(SEED + record_id)

    graphs, _ = model.sampling_batch_h1(
        batch_id=record_id,
        batch_size=1,
        keep_chain=0,
        number_chain_steps=0,
        num_nodes=n,
        y=y,
        figures=False,
    )

    node_types = (
        graphs[0][0].detach().long().cpu().numpy()
    )
    adjacency = (
        graphs[0][1].detach().long().cpu().numpy()
    )

    if len(node_types) != n:
        raise AssertionError(
            f"Node count changed: {n} -> {len(node_types)}"
        )

    full_graph = graph_from_adjacency(adjacency)

    used_types, used_adjacency = upstream_lcc(
        node_types, adjacency
    )
    used_graph = graph_from_adjacency(used_adjacency)

    real_stats = base.graph_stats(real_graph)
    full_stats = base.graph_stats(full_graph)
    used_stats = base.graph_stats(used_graph)

    return {
        "checkpoint": checkpoint_name,
        "record_id": record_id,
        "condition_y": condition,
        "real_nodes": n,
        "real_edges": real_graph.number_of_edges(),
        "real_clustering": real_stats["clustering"],
        "real_max_degree": real_stats["max_degree"],
        "full_nodes": full_graph.number_of_nodes(),
        "full_edges": full_graph.number_of_edges(),
        "full_edge_ratio": (
            full_graph.number_of_edges()
            / max(real_graph.number_of_edges(), 1)
        ),
        "full_components": full_stats["components"],
        "full_connected": int(
            full_stats["components"] == 1
        ),
        "full_clustering": full_stats["clustering"],
        "full_max_degree": full_stats["max_degree"],
        "full_mean_degree": full_stats["mean_degree"],
        "lcc_nodes": used_graph.number_of_nodes(),
        "lcc_edges": used_graph.number_of_edges(),
        "lcc_retention": (
            used_graph.number_of_nodes()
            / max(full_graph.number_of_nodes(), 1)
        ),
        "lcc_clustering": used_stats["clustering"],
        "lcc_max_degree": used_stats["max_degree"],
        "lcc_mean_degree": used_stats["mean_degree"],
        "node_accuracy": float(
            np.mean(node_types == real_classes)
        ),
        "full_condition_match": int(
            base.majority(node_types.tolist())
            == condition
        ),
        "lcc_condition_match": int(
            base.majority(used_types.tolist())
            == condition
        ),
    }


def summarize(rows):
    return {
        "graph_count": len(rows),
        "real_edges_mean": average(
            rows, "real_edges"
        ),
        "full_edges_mean": average(
            rows, "full_edges"
        ),
        "full_edge_ratio_mean": average(
            rows, "full_edge_ratio"
        ),
        "full_connected_rate": average(
            rows, "full_connected"
        ),
        "full_components_mean": average(
            rows, "full_components"
        ),
        "lcc_retention_mean": average(
            rows, "lcc_retention"
        ),
        "real_clustering_mean": average(
            rows, "real_clustering"
        ),
        "full_clustering_mean": average(
            rows, "full_clustering"
        ),
        "lcc_clustering_mean": average(
            rows, "lcc_clustering"
        ),
        "real_max_degree_mean": average(
            rows, "real_max_degree"
        ),
        "full_max_degree_mean": average(
            rows, "full_max_degree"
        ),
        "lcc_max_degree_mean": average(
            rows, "lcc_max_degree"
        ),
        "full_mean_degree_mean": average(
            rows, "full_mean_degree"
        ),
        "lcc_mean_degree_mean": average(
            rows, "lcc_mean_degree"
        ),
        "node_accuracy_mean": average(
            rows, "node_accuracy"
        ),
        "full_condition_match_rate": average(
            rows, "full_condition_match"
        ),
        "lcc_condition_match_rate": average(
            rows, "lcc_condition_match"
        ),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    selected, histogram = (
        base.extract_validation_endpoints()
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    all_rows = []
    report = {
        "seed": SEED,
        "selected_count": len(selected),
        "selection_histogram": histogram,
        "validation_only": True,
        "checkpoints": {},
    }

    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = base._load_checkpoint_cpu(
            checkpoint_path
        )
        model = (
            base.ConditionalGraphBetaDiffusionH1
            .load_from_checkpoint(
                str(checkpoint_path),
                map_location="cpu",
            )
        )
        model = model.to(device)
        model.eval()

        print()
        print("=" * 100)
        print("checkpoint:", checkpoint_name)
        print("epoch:", checkpoint.get("epoch"))
        print("global_step:", checkpoint.get("global_step"))
        print("device:", device)

        rows = []

        for record_id, record in enumerate(selected):
            rows.append(
                evaluate(
                    model,
                    record,
                    record_id,
                    checkpoint_name,
                )
            )
            print(
                f"\r{checkpoint_name}: "
                f"{record_id + 1}/{len(selected)}",
                end="",
                flush=True,
            )

        print()

        report["checkpoints"][checkpoint_name] = {
            "path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
            "summary": summarize(rows),
        }
        all_rows.extend(rows)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = OUTPUT / "per_endpoint_results.csv"
    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = OUTPUT / "summary.json"
    json_path.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 190)
    print("Beta-H1 validation full-reverse comparison")
    print("=" * 190)
    print(
        f"{'checkpoint':>26} "
        f"{'edges':>9} {'ratio':>8} "
        f"{'connected':>10} {'components':>11} "
        f"{'LCC':>8} {'full clust':>11} "
        f"{'LCC clust':>11} {'maxdeg':>9} "
        f"{'node acc':>9} {'cond full':>10} "
        f"{'cond LCC':>10}"
    )
    print("-" * 190)

    for name, checkpoint in report["checkpoints"].items():
        item = checkpoint["summary"]
        print(
            f"{name:>26} "
            f"{item['full_edges_mean']:9.2f} "
            f"{item['full_edge_ratio_mean']:8.3f} "
            f"{item['full_connected_rate']:10.3f} "
            f"{item['full_components_mean']:11.3f} "
            f"{item['lcc_retention_mean']:8.3f} "
            f"{item['full_clustering_mean']:11.4f} "
            f"{item['lcc_clustering_mean']:11.4f} "
            f"{item['full_max_degree_mean']:9.2f} "
            f"{item['node_accuracy_mean']:9.3f} "
            f"{item['full_condition_match_rate']:10.3f} "
            f"{item['lcc_condition_match_rate']:10.3f}"
        )

    reference = next(iter(
        report["checkpoints"].values()
    ))["summary"]

    print()
    print("Reference real edges:", reference["real_edges_mean"])
    print(
        "Reference real clustering:",
        reference["real_clustering_mean"],
    )
    print(
        "Reference real max degree:",
        reference["real_max_degree_mean"],
    )
    print("CSV:", csv_path)
    print("JSON:", json_path)


main()
