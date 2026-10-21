from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-forced test for the Cora Original Glue checkpoint "
            "using real r2 community-pair inputs."
        )
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def scalar(value) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)

    if array.size == 0:
        return {"count": 0}

    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def extract_real_pair(
    storage,
    slices,
    graph_index: int,
) -> tuple[Data, torch.Tensor, dict]:
    x_start = scalar(slices["x"][graph_index])
    x_end = scalar(slices["x"][graph_index + 1])

    edge_start = scalar(slices["edge_index"][graph_index])
    edge_end = scalar(slices["edge_index"][graph_index + 1])

    y_start = scalar(slices["y"][graph_index])
    y_end = scalar(slices["y"][graph_index + 1])

    pos_start = scalar(slices["pos"][graph_index])
    pos_end = scalar(slices["pos"][graph_index + 1])

    x = storage.x[x_start:x_end].clone()
    edge_index = storage.edge_index[:, edge_start:edge_end].clone()
    edge_attr = storage.edge_attr[edge_start:edge_end].clone()
    y = storage.y[y_start:y_end].clone()
    pos = storage.pos[pos_start:pos_end].clone()

    node_count = int(x.shape[0])

    if edge_index.numel() > 0:
        edge_min = int(edge_index.min().item())
        edge_max = int(edge_index.max().item())

        if edge_min < 0 or edge_max >= node_count:
            raise RuntimeError(
                f"Pair {graph_index}: edge_index is not local. "
                f"min={edge_min}, max={edge_max}, nodes={node_count}"
            )

    if pos.ndim != 2:
        raise RuntimeError(
            f"Pair {graph_index}: pos shape={tuple(pos.shape)}"
        )

    if pos.shape[0] < node_count or pos.shape[1] < node_count:
        raise RuntimeError(
            f"Pair {graph_index}: pos={tuple(pos.shape)}, "
            f"nodes={node_count}"
        )

    pos_actual = pos[:node_count, :node_count].bool()

    source = edge_index[0].long()
    target = edge_index[1].long()

    # True means the two nodes belong to the same H1 community.
    intra_mask = pos_actual[source, target]

    directed_true_cross = int((~intra_mask).sum().item())

    if directed_true_cross % 2 != 0:
        raise RuntimeError(
            f"Pair {graph_index}: true cross-edge count is not symmetric"
        )

    true_cross_edges = directed_true_cross // 2

    # Teacher forcing:
    # retain the two real H1 internal structures but remove all true
    # cross-community edges before calling the Glue sampler.
    input_edge_index = edge_index[:, intra_mask].contiguous()
    input_edge_attr = edge_attr[intra_mask].contiguous()

    input_internal_edges = int(input_edge_index.shape[1]) // 2

    candidate_cross_edges = int(
        torch.triu(
            ~pos_actual,
            diagonal=1,
        ).sum().item()
    )

    community_1_nodes = int(pos_actual[0].sum().item())
    community_2_nodes = node_count - community_1_nodes

    y_vector = y.reshape(-1, y.shape[-1])[0]
    y_class = int(y_vector.argmax().item())

    data = Data(
        x=x,
        edge_index=input_edge_index,
        edge_attr=input_edge_attr,
        y=y,
        pos=pos,
    )

    metadata = {
        "graph_index": graph_index,
        "node_count": node_count,
        "community_1_nodes": community_1_nodes,
        "community_2_nodes": community_2_nodes,
        "input_internal_edges": input_internal_edges,
        "true_cross_edges": true_cross_edges,
        "candidate_cross_edges": candidate_cross_edges,
        "y_class": y_class,
    }

    return data, pos_actual, metadata


def count_generated_edges(
    graph: nx.Graph,
    pos_actual: torch.Tensor,
    node_count: int,
) -> tuple[int, int]:
    generated_internal = 0
    generated_cross = 0

    for raw_left, raw_right in graph.edges():
        left = int(raw_left)
        right = int(raw_right)

        if left == right:
            continue

        if not (
            0 <= left < node_count
            and 0 <= right < node_count
        ):
            raise RuntimeError(
                "Generated Glue graph contains an invalid node edge: "
                f"{(left, right)}, node_count={node_count}"
            )

        if bool(pos_actual[left, right]):
            generated_internal += 1
        else:
            generated_cross += 1

    return generated_internal, generated_cross


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    processed_path = Path(args.processed).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path in (
        root,
        config_path,
        checkpoint_path,
        processed_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / "dgd"))
    sys.path.insert(0, str(root / "higgs"))
    sys.path.insert(0, str(root))

    # Import only after project paths have been added.
    from higgs_utils import get_samplers

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = OmegaConf.load(config_path)

    with open_dict(cfg):
        cfg.dataset.h = 1.5
        cfg.general.glue_model = str(checkpoint_path)
        cfg.general.glue_model_family = "original_glue"
        cfg.general.gpus = 1 if torch.cuda.is_available() else 0
        cfg.general.wandb = "disabled"
        cfg.general.name = "cora_teacher_forced_glue"
        cfg.general.force_eval = True

    cfg, model = get_samplers(cfg)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and int(cfg.general.gpus) > 0
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    dropout_training = sum(
        int(module.training)
        for module in model.modules()
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

    if model.training or dropout_training != 0:
        raise RuntimeError(
            "Glue model is not in strict evaluation mode"
        )

    loaded = torch.load(processed_path, map_location="cpu")

    if not isinstance(loaded, tuple) or len(loaded) < 2:
        raise TypeError(
            "Unexpected processed-cache format: "
            f"{type(loaded).__name__}"
        )

    storage, slices = loaded[:2]
    graph_count = len(slices["x"]) - 1

    if args.count < 1 or args.count > graph_count:
        raise ValueError(
            f"--count must be in [1, {graph_count}], "
            f"got {args.count}"
        )

    generator = np.random.default_rng(args.seed)

    selected_indices = sorted(
        int(index)
        for index in generator.choice(
            graph_count,
            size=args.count,
            replace=False,
        )
    )

    number_chain_steps = min(40, int(model.T) - 1)

    print("=" * 100)
    print("Cora teacher-forced Glue")
    print("=" * 100)
    print("Checkpoint        :", checkpoint_path)
    print("Checkpoint SHA256 :", sha256(checkpoint_path))
    print("Processed cache   :", processed_path)
    print("Device            :", device)
    print("Model training    :", model.training)
    print("Dropout training  :", dropout_training)
    print("Model T           :", int(model.T))
    print("Selected pairs    :", len(selected_indices))
    print("Pair indices      :", selected_indices)
    print("=" * 100)

    rows = []

    for order, graph_index in enumerate(selected_indices):
        sample, pos_actual, metadata = extract_real_pair(
            storage,
            slices,
            graph_index,
        )

        if sample.y.shape[-1] != int(model.ydim_output):
            raise RuntimeError(
                f"Pair {graph_index}: y dimension "
                f"{sample.y.shape[-1]} does not match "
                f"checkpoint dimension {model.ydim_output}"
            )

        loader = DataLoader(
            [sample],
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader)).to(device)

        with torch.inference_mode():
            _, networkx_graphs = model.sampling_batch_glue(
                batch_id=order,
                batch_size=1,
                keep_chain=0,
                number_chain_steps=number_chain_steps,
                data=batch,
                figures=False,
            )

        if len(networkx_graphs) != 1:
            raise RuntimeError(
                f"Pair {graph_index}: expected one output graph, "
                f"got {len(networkx_graphs)}"
            )

        generated_graph = nx.Graph(networkx_graphs[0])

        generated_internal, generated_cross = count_generated_edges(
            generated_graph,
            pos_actual,
            metadata["node_count"],
        )

        candidates = metadata["candidate_cross_edges"]

        row = {
            **metadata,
            "generated_internal_edges": generated_internal,
            "generated_cross_edges": generated_cross,
            "true_cross_density": (
                metadata["true_cross_edges"] / candidates
                if candidates
                else 0.0
            ),
            "generated_cross_density": (
                generated_cross / candidates
                if candidates
                else 0.0
            ),
            "internal_edge_delta": (
                generated_internal
                - metadata["input_internal_edges"]
            ),
        }

        rows.append(row)

        print(
            f"[{order + 1:03d}/{len(selected_indices):03d}] "
            f"idx={graph_index:03d} "
            f"sizes=({metadata['community_1_nodes']},"
            f"{metadata['community_2_nodes']}) "
            f"y={metadata['y_class']} "
            f"true_cross={metadata['true_cross_edges']} "
            f"generated_cross={generated_cross} "
            f"density={row['generated_cross_density']:.8f} "
            f"internal_delta={row['internal_edge_delta']}"
        )

        del batch
        del sample
        del generated_graph
        del networkx_graphs

        if torch.cuda.is_available() and (order + 1) % 10 == 0:
            torch.cuda.empty_cache()

    csv_path = output_dir / "teacher_forced_pairs.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    true_edges = [
        row["true_cross_edges"]
        for row in rows
    ]

    generated_edges = [
        row["generated_cross_edges"]
        for row in rows
    ]

    true_densities = [
        row["true_cross_density"]
        for row in rows
    ]

    generated_densities = [
        row["generated_cross_density"]
        for row in rows
    ]

    internal_deltas = [
        row["internal_edge_delta"]
        for row in rows
    ]

    true_edge_mean = float(np.mean(true_edges))
    generated_edge_mean = float(np.mean(generated_edges))
    true_density_mean = float(np.mean(true_densities))
    generated_density_mean = float(np.mean(generated_densities))

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "processed_cache": str(processed_path),
        "seed": args.seed,
        "selected_indices": selected_indices,
        "model_training": bool(model.training),
        "dropout_training": dropout_training,
        "true_cross_edge_summary": describe(true_edges),
        "generated_cross_edge_summary": describe(generated_edges),
        "true_cross_density_summary": describe(true_densities),
        "generated_cross_density_summary": describe(
            generated_densities
        ),
        "internal_edge_delta_summary": describe(internal_deltas),
        "generated_to_true_edge_mean_ratio": (
            generated_edge_mean / true_edge_mean
            if true_edge_mean
            else None
        ),
        "generated_to_true_density_mean_ratio": (
            generated_density_mean / true_density_mean
            if true_density_mean
            else None
        ),
    }

    json_path = output_dir / "teacher_forced_summary.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    print()
    print("=" * 100)
    print("Teacher-forced summary")
    print("=" * 100)
    print("True cross edges:")
    print(result["true_cross_edge_summary"])
    print()
    print("Generated cross edges:")
    print(result["generated_cross_edge_summary"])
    print()
    print("True cross density:")
    print(result["true_cross_density_summary"])
    print()
    print("Generated cross density:")
    print(result["generated_cross_density_summary"])
    print()
    print(
        "Generated / true edge mean ratio:",
        result["generated_to_true_edge_mean_ratio"],
    )
    print(
        "Generated / true density mean ratio:",
        result["generated_to_true_density_mean_ratio"],
    )
    print("Internal-edge delta:")
    print(result["internal_edge_delta_summary"])
    print()
    print("CSV :", csv_path)
    print("JSON:", json_path)
    print("=" * 100)


if __name__ == "__main__":
    main()
