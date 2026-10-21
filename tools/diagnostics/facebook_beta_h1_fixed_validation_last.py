#!/usr/bin/env python3

import csv
import hashlib
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data.separate import separate
from tqdm import tqdm


ROOT = Path("/data2/wenb/ylu/HiGGs-GBD/HiGGs-main")

CHECKPOINT = (
    ROOT
    / "outputs/2026-07-30/10-58-34/checkpoints/fb_beta_h1_r10_e500/last.ckpt"
)

VAL_CACHE = (
    ROOT
    / "data/fb/processed/"
    "proc_val_h1_r10_majority_max400_bs8.pt"
)

OUTPUT_DIR = (
    ROOT
    / "higgs/outputs/"
    "facebook_beta_h1_fixed_validation_last_seed20260825"
)

BASE_SEED = 20260825
PER_CLASS = 5

sys.path.insert(0, str(ROOT / "dgd"))
sys.path.insert(0, str(ROOT / "higgs"))

from higgs_utils import (  # noqa: E402
    _load_checkpoint_cpu,
    register_checkpoint_pickle_compatibility,
)
from diffusion_model_beta_h1 import (  # noqa: E402
    ConditionalGraphBetaDiffusionH1,
)
from interface_ablation import apply_upstream_lcc  # noqa: E402


FIELDS = [
    "record_id",
    "validation_pair_index",
    "side",
    "reverse_seed",
    "condition_y",

    "real_nodes",
    "real_edges",
    "real_density",
    "real_mean_degree",
    "real_max_degree",
    "real_clustering",
    "real_components",

    "full_nodes",
    "full_edges",
    "full_density",
    "full_mean_degree",
    "full_max_degree",
    "full_clustering",
    "full_components",

    "used_nodes",
    "used_edges",
    "used_density",
    "used_mean_degree",
    "used_max_degree",
    "used_clustering",
    "used_components",

    "removed_nodes",
    "lcc_retention",
    "condition_majority_match",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def node_classes(x):
    if x.ndim == 1:
        return x.long()
    if x.ndim != 2:
        raise RuntimeError(f"Unsupported X shape: {tuple(x.shape)}")
    if x.shape[1] == 1:
        return x[:, 0].round().long()
    return x.argmax(dim=1).long()


def majority(values):
    counts = Counter(int(value) for value in values)
    maximum = max(counts.values())
    return min(
        class_index
        for class_index, count in counts.items()
        if count == maximum
    )


def graph_stats(graph):
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    return {
        "nodes": n,
        "edges": m,
        "density": nx.density(graph) if n > 1 else 0.0,
        "mean_degree": 2.0 * m / n if n else 0.0,
        "max_degree": max((degree for _, degree in graph.degree()), default=0),
        "clustering": nx.average_clustering(graph) if n else 0.0,
        "components": nx.number_connected_components(graph) if n else 0,
    }


def endpoint_signature(graph, classes):
    payload = {
        "classes": [int(value) for value in classes],
        "edges": [
            [int(min(left, right)), int(max(left, right))]
            for left, right in sorted(
                (min(a, b), max(a, b))
                for a, b in graph.edges()
            )
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def extract_validation_endpoints():
    loaded = torch.load(VAL_CACHE, map_location="cpu")

    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError(
            "Expected Facebook H1 validation cache as "
            "collated (Data, slices)"
        )

    data, slices = loaded

    if "x" not in slices:
        raise RuntimeError("Validation cache has no x slices")

    graph_count = len(slices["x"]) - 1
    by_class = defaultdict(list)
    seen = set()
    majority_mismatches = 0
    disconnected_real_graphs = 0

    for graph_index in range(graph_count):
        sample = separate(
            cls=data.__class__,
            batch=data,
            idx=graph_index,
            slice_dict=slices,
            decrement=False,
        )

        classes = node_classes(sample.x)
        n = int(classes.shape[0])

        if n <= 0:
            raise RuntimeError(
                f"Validation H1 graph {graph_index} is empty"
            )

        edge_index = sample.edge_index.detach().cpu().long()

        if edge_index.numel():
            minimum = int(edge_index.min().item())
            maximum = int(edge_index.max().item())
            if minimum < 0 or maximum >= n:
                raise RuntimeError(
                    f"Validation H1 graph {graph_index} has invalid "
                    f"edge indices: min={minimum}, max={maximum}, n={n}"
                )

        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(
            (int(left), int(right))
            for left, right in edge_index.t().tolist()
            if int(left) != int(right)
        )

        components = nx.number_connected_components(graph)
        if components != 1:
            disconnected_real_graphs += 1
            component_sizes = sorted(
                (
                    len(component)
                    for component in nx.connected_components(graph)
                ),
                reverse=True,
            )
            print(
                "[Facebook H1 diagnostic] WARNING: real validation "
                f"graph_index={graph_index} is disconnected: "
                f"nodes={n} edges={graph.number_of_edges()} "
                f"components={components} "
                f"component_sizes={component_sizes}"
            )

        if not hasattr(sample, "y") or sample.y.numel() == 0:
            raise RuntimeError(
                f"Validation H1 graph {graph_index} has no y condition"
            )

        condition = int(
            sample.y.reshape(1, -1).argmax(dim=1).item()
        )
        computed_majority = majority(classes.tolist())

        if condition != computed_majority:
            majority_mismatches += 1

        class_values = [int(value) for value in classes.tolist()]
        signature = endpoint_signature(graph, class_values)

        if signature in seen:
            continue

        seen.add(signature)
        by_class[condition].append({
            "pair_index": graph_index,
            "side": "whole_h1",
            "condition_y": condition,
            "graph": graph,
            "classes": class_values,
            "signature": signature,
        })

    if majority_mismatches:
        raise RuntimeError(
            f"Facebook H1 validation cache contains "
            f"{majority_mismatches} y/majority mismatches"
        )

    if disconnected_real_graphs:
        print(
            "[Facebook H1 diagnostic] WARNING: preserving "
            f"{disconnected_real_graphs} disconnected real validation "
            "H1 graph(s) exactly as stored; the cache is not modified."
        )

    rng = random.Random(BASE_SEED)
    selected = []

    for class_index in sorted(by_class):
        candidates = list(by_class[class_index])
        rng.shuffle(candidates)
        selected.extend(candidates[:PER_CLASS])

    selected.sort(
        key=lambda row: (
            row["condition_y"],
            row["graph"].number_of_nodes(),
            row["signature"],
        )
    )

    for record_id, row in enumerate(selected):
        row["record_id"] = record_id

    return selected, {
        str(class_index): len(rows)
        for class_index, rows in sorted(by_class.items())
    }


def prefix_stats(prefix, stats):
    return {
        f"{prefix}_{key}": value
        for key, value in stats.items()
    }


def typed_row(row):
    integer_fields = {
        "record_id",
        "validation_pair_index",
        "reverse_seed",
        "condition_y",
        "real_nodes",
        "real_edges",
        "real_max_degree",
        "real_components",
        "full_nodes",
        "full_edges",
        "full_max_degree",
        "full_components",
        "used_nodes",
        "used_edges",
        "used_max_degree",
        "used_components",
        "removed_nodes",
        "condition_majority_match",
    }

    result = {}

    for key, value in row.items():
        if key in integer_fields:
            result[key] = int(float(value))
        elif key == "side":
            result[key] = value
        else:
            result[key] = float(value)

    return result


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def summarize(rows, checkpoint_metadata, selection_histogram):
    features = [
        "edges",
        "density",
        "mean_degree",
        "max_degree",
        "clustering",
    ]

    summary = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_global_step": checkpoint_metadata.get("global_step"),
        "validation_cache": str(VAL_CACHE),
        "validation_cache_sha256": sha256(VAL_CACHE),
        "base_seed": BASE_SEED,
        "per_class": PER_CLASS,
        "selected_count": len(rows),
        "available_unique_endpoints_by_condition": selection_histogram,
        "selected_condition_histogram": dict(sorted(Counter(
            row["condition_y"] for row in rows
        ).items())),
        "full_connected_rate": mean(
            row["full_components"] == 1 for row in rows
        ),
        "used_connected_rate": mean(
            row["used_components"] == 1 for row in rows
        ),
        "mean_lcc_retention": mean(
            row["lcc_retention"] for row in rows
        ),
        "condition_majority_match_rate": mean(
            row["condition_majority_match"] for row in rows
        ),
        "means": {},
        "mean_absolute_errors": {
            "full": {},
            "used": {},
        },
    }

    for feature in (
        "nodes",
        "edges",
        "density",
        "mean_degree",
        "max_degree",
        "clustering",
        "components",
    ):
        summary["means"][feature] = {
            "real": mean(row[f"real_{feature}"] for row in rows),
            "full": mean(row[f"full_{feature}"] for row in rows),
            "used": mean(row[f"used_{feature}"] for row in rows),
        }

    for feature in features:
        summary["mean_absolute_errors"]["full"][feature] = mean(
            abs(row[f"full_{feature}"] - row[f"real_{feature}"])
            for row in rows
        )
        summary["mean_absolute_errors"]["used"][feature] = mean(
            abs(row[f"used_{feature}"] - row[f"real_{feature}"])
            for row in rows
        )

    return summary


def main():
    for path in (CHECKPOINT, VAL_CACHE):
        if not path.is_file():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "per_endpoint_results.csv"
    json_path = OUTPUT_DIR / "summary.json"
    txt_path = OUTPUT_DIR / "summary.txt"

    selected, selection_histogram = extract_validation_endpoints()

    print("Available unique endpoints:", selection_histogram)
    print("Selected endpoints:", len(selected))
    print(
        "Selected conditions:",
        dict(sorted(Counter(
            row["condition_y"] for row in selected
        ).items())),
    )

    register_checkpoint_pickle_compatibility()
    checkpoint_metadata = _load_checkpoint_cpu(CHECKPOINT)

    model = ConditionalGraphBetaDiffusionH1.load_from_checkpoint(
        str(CHECKPOINT),
        map_location="cpu",
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    dropout_training = sum(
        isinstance(module, torch.nn.Dropout) and module.training
        for module in model.modules()
    )

    if model.training or dropout_training:
        raise RuntimeError(
            "Beta-H1 model did not enter eval mode"
        )

    print(
        "Using Facebook Beta-H1 last checkpoint: "
        f"epoch={checkpoint_metadata.get('epoch')} "
        f"global_step={checkpoint_metadata.get('global_step')} "
        f"device={device} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    existing_rows = []
    completed = set()

    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            existing_rows = [
                typed_row(row)
                for row in csv.DictReader(handle)
            ]
        completed = {
            row["record_id"]
            for row in existing_rows
        }
        print("Resuming completed endpoint count:", len(completed))

    mode = "a" if csv_path.exists() else "w"

    with csv_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)

        if mode == "w":
            writer.writeheader()
            handle.flush()

        for record in tqdm(
            selected,
            desc="Fixed-n fixed-y Beta-H1 validation",
        ):
            record_id = record["record_id"]

            if record_id in completed:
                continue

            graph = record["graph"]
            condition = record["condition_y"]
            n_nodes = graph.number_of_nodes()
            reverse_seed = BASE_SEED + record_id

            seed_everything(reverse_seed)

            y = F.one_hot(
                torch.tensor([condition], device=device),
                num_classes=model.ydim_output,
            ).float()

            with torch.inference_mode():
                graphs, _ = model.sampling_batch_h1(
                    batch_id=record_id,
                    batch_size=1,
                    keep_chain=0,
                    number_chain_steps=40,
                    num_nodes=n_nodes,
                    y=y,
                    figures=False,
                )

            generated_nodes = graphs[0][0].long().cpu()
            generated_adj = graphs[0][1].long().cpu()

            if generated_nodes.shape[0] != n_nodes:
                raise RuntimeError("Fixed node count was not preserved")
            if generated_adj.shape != (n_nodes, n_nodes):
                raise RuntimeError("Generated adjacency shape mismatch")
            if not torch.equal(generated_adj, generated_adj.T):
                raise RuntimeError("Generated adjacency is asymmetric")
            if torch.count_nonzero(
                torch.diagonal(generated_adj)
            ).item() != 0:
                raise RuntimeError("Generated adjacency has self-loops")
            if not torch.all(
                (generated_adj == 0) | (generated_adj == 1)
            ):
                raise RuntimeError("Generated adjacency is not binary")

            used_nodes, used_adj = apply_upstream_lcc(
                generated_nodes,
                generated_adj,
            )

            full_graph = nx.from_numpy_array(
                generated_adj.numpy()
            )
            used_graph = nx.from_numpy_array(
                used_adj.numpy()
            )

            real_stats = graph_stats(graph)
            full_stats = graph_stats(full_graph)
            used_stats = graph_stats(used_graph)

            generated_majority = majority(
                generated_nodes.tolist()
            )

            row = {
                "record_id": record_id,
                "validation_pair_index": record["pair_index"],
                "side": record["side"],
                "reverse_seed": reverse_seed,
                "condition_y": condition,
                **prefix_stats("real", real_stats),
                **prefix_stats("full", full_stats),
                **prefix_stats("used", used_stats),
                "removed_nodes": n_nodes - used_graph.number_of_nodes(),
                "lcc_retention": (
                    used_graph.number_of_nodes() / n_nodes
                ),
                "condition_majority_match": int(
                    generated_majority == condition
                ),
            }

            writer.writerow(row)
            handle.flush()
            existing_rows.append(row)

    existing_rows.sort(key=lambda row: row["record_id"])

    summary = summarize(
        existing_rows,
        checkpoint_metadata,
        selection_histogram,
    )

    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "Facebook Beta-H1 fixed-n/fixed-y validation diagnostic",
        f"checkpoint: {CHECKPOINT}",
        f"epoch: {summary['checkpoint_epoch']}",
        f"selected endpoints: {summary['selected_count']}",
        f"full connected rate: {summary['full_connected_rate']}",
        f"mean LCC retention: {summary['mean_lcc_retention']}",
        (
            "condition majority match rate: "
            f"{summary['condition_majority_match_rate']}"
        ),
        "",
        "Feature means:",
    ]

    for feature, values in summary["means"].items():
        lines.append(
            f"{feature}: "
            f"real={values['real']} "
            f"full={values['full']} "
            f"used={values['used']}"
        )

    lines.extend([
        "",
        "Full graph mean absolute errors:",
        json.dumps(
            summary["mean_absolute_errors"]["full"],
            indent=2,
        ),
        "",
        "Used/LCC graph mean absolute errors:",
        json.dumps(
            summary["mean_absolute_errors"]["used"],
            indent=2,
        ),
    ])

    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(lines))
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
