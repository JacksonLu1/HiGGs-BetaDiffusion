"""Read-only full-reverse audit for an Original HiGGs Cora Glue checkpoint.

The tool loads an existing collated PyG test cache and an existing
DiscreteDenoisingDiffusion checkpoint.  It never trains, reprocesses data, or
writes to either input.  Each pair is seeded independently so an interrupted
run can be resumed without changing later pairs' reverse noise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader
from tqdm import tqdm


FIELDNAMES = (
    "pair_index",
    "reverse_seed",
    "y_class",
    "num_nodes",
    "community_1_nodes",
    "community_2_nodes",
    "candidate_count",
    "true_edge_count",
    "generated_edge_count",
    "true_density",
    "generated_density",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "intra_edges_changed",
    "x_changed",
    "asymmetric_entry_count",
    "diagonal_nonzero_count",
    "nonfinite_count",
)

INTEGER_FIELDS = {
    "pair_index",
    "reverse_seed",
    "y_class",
    "num_nodes",
    "community_1_nodes",
    "community_2_nodes",
    "candidate_count",
    "true_edge_count",
    "generated_edge_count",
    "tp",
    "fp",
    "fn",
    "tn",
    "intra_edges_changed",
    "x_changed",
    "asymmetric_entry_count",
    "diagonal_nonzero_count",
    "nonfinite_count",
}

FLOAT_FIELDS = {
    "true_density",
    "generated_density",
    "precision",
    "recall",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Original Cora Glue full reverse on an existing real-pair "
            "processed cache without training or data processing."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sampling-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of pairs to evaluate; 0 means the complete cache.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is non-finite: {result}")
    return result


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def load_collated_cache(path: Path):
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError(
            "Expected the processed cache to contain collated (data, slices), "
            f"got {type(loaded)}"
        )
    data, slices = loaded
    lengths = [
        len(value)
        for value in slices.values()
        if value is not None and hasattr(value, "__len__") and len(value) >= 2
    ]
    if not lengths or len(set(lengths)) != 1:
        raise RuntimeError(f"Inconsistent processed-cache slices: {lengths}")
    return data, slices, lengths[0] - 1


def get_sample(data, slices, index: int):
    return separate(
        cls=data.__class__,
        batch=data,
        idx=index,
        slice_dict=slices,
        decrement=False,
    )


def node_classes(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.long()
    if x.ndim != 2:
        raise RuntimeError(f"Unsupported X shape: {tuple(x.shape)}")
    if x.shape[-1] == 1:
        return x[:, 0].round().long()
    return x.argmax(dim=-1).long()


def pair_masks(sample):
    n = int(sample.num_nodes)
    pos = sample.pos.detach().cpu()
    if pos.ndim != 2 or pos.shape[0] < n or pos.shape[1] < n:
        raise RuntimeError(
            f"Invalid pos shape {tuple(pos.shape)} for {n} nodes"
        )
    same = pos[:n, :n].bool()
    if not torch.equal(same, same.transpose(0, 1)):
        raise RuntimeError("Glue pos mask is not symmetric")
    if n and not bool(torch.diagonal(same).all()):
        raise RuntimeError("Glue pos mask does not include every valid diagonal")
    mutable_upper = torch.triu(~same, diagonal=1)
    intra_upper = torch.triu(same, diagonal=1)
    if n:
        first_size = int(same[0].sum().item())
    else:
        first_size = 0
    second_size = n - first_size
    candidate_count = int(mutable_upper.sum().item())
    if first_size <= 0 or second_size <= 0:
        raise RuntimeError(
            f"Expected two non-empty Glue communities, got {first_size}, "
            f"{second_size}"
        )
    if candidate_count != first_size * second_size:
        raise RuntimeError(
            "Glue mutable mask is not exactly the cross product of two "
            f"communities: {candidate_count} != {first_size * second_size}"
        )
    return same, mutable_upper, intra_upper, first_size, second_size


def true_adjacency(sample) -> torch.Tensor:
    n = int(sample.num_nodes)
    adjacency = torch.zeros((n, n), dtype=torch.bool)
    edge_index = sample.edge_index.detach().cpu().long()
    valid = (
        (edge_index[0] >= 0)
        & (edge_index[1] >= 0)
        & (edge_index[0] < n)
        & (edge_index[1] < n)
    )
    adjacency[edge_index[0, valid], edge_index[1, valid]] = True
    if not torch.equal(adjacency, adjacency.transpose(0, 1)):
        raise RuntimeError("Cached true adjacency is not symmetric")
    return adjacency


def evaluate_pair(model, sample, index: int, reverse_seed: int, device):
    same, mutable, intra, size_1, size_2 = pair_masks(sample)
    true_adj = true_adjacency(sample)
    true_x = node_classes(sample.x.detach().cpu())
    y = sample.y.detach().cpu().reshape(-1)
    y_class = int(y.argmax().item()) if y.numel() > 1 else int(y.item())

    loader = DataLoader([sample], batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(device)
    seed_everything(reverse_seed)
    with torch.inference_mode():
        predicted_graphs, _ = model.sampling_batch_glue(
            batch_id=index,
            batch_size=1,
            keep_chain=0,
            number_chain_steps=40,
            data=batch,
            figures=False,
        )
    if len(predicted_graphs) != 1:
        raise RuntimeError(
            f"Pair {index}: expected one predicted graph, got "
            f"{len(predicted_graphs)}"
        )
    predicted_x, predicted_e = predicted_graphs[0]
    predicted_x = predicted_x.detach().cpu().long()[: true_x.shape[0]]
    predicted_e = predicted_e.detach().cpu()[: true_x.shape[0], : true_x.shape[0]]
    nonfinite = int((~torch.isfinite(predicted_e.float())).sum().item())
    predicted_adj = predicted_e.long() == 1

    asymmetric = int(
        (predicted_adj != predicted_adj.transpose(0, 1)).sum().item()
    )
    diagonal = int(torch.diagonal(predicted_adj).sum().item())
    intra_changed = int(((predicted_adj ^ true_adj) & intra).sum().item())
    x_changed = int((predicted_x != true_x).sum().item())

    tp = int((predicted_adj & true_adj & mutable).sum().item())
    fp = int((predicted_adj & ~true_adj & mutable).sum().item())
    fn = int((~predicted_adj & true_adj & mutable).sum().item())
    candidate_count = int(mutable.sum().item())
    tn = candidate_count - tp - fp - fn
    true_edges = tp + fn
    generated_edges = tp + fp
    precision = tp / generated_edges if generated_edges else 0.0
    recall = tp / true_edges if true_edges else 0.0

    row = {
        "pair_index": index,
        "reverse_seed": reverse_seed,
        "y_class": y_class,
        "num_nodes": int(sample.num_nodes),
        "community_1_nodes": size_1,
        "community_2_nodes": size_2,
        "candidate_count": candidate_count,
        "true_edge_count": true_edges,
        "generated_edge_count": generated_edges,
        "true_density": true_edges / candidate_count,
        "generated_density": generated_edges / candidate_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "intra_edges_changed": intra_changed,
        "x_changed": x_changed,
        "asymmetric_entry_count": asymmetric,
        "diagonal_nonzero_count": diagonal,
        "nonfinite_count": nonfinite,
    }
    for key in FLOAT_FIELDS:
        finite_float(row[key], f"pair_{index}.{key}")
    return row


def typed_row(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in INTEGER_FIELDS:
            result[key] = int(value)
        elif key in FLOAT_FIELDS:
            result[key] = float(value)
        else:
            result[key] = value
    return result


def summarize(records: list[dict[str, Any]], target_count: int) -> dict[str, Any]:
    candidates = sum(row["candidate_count"] for row in records)
    true_edges = sum(row["true_edge_count"] for row in records)
    generated_edges = sum(row["generated_edge_count"] for row in records)
    tp = sum(row["tp"] for row in records)
    fp = sum(row["fp"] for row in records)
    fn = sum(row["fn"] for row in records)
    y_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        y_groups[row["y_class"]].append(row)
    sorted_edges = sorted(
        (row["generated_edge_count"] for row in records), reverse=True
    )
    top_shares = {}
    for count in (1, 5, 10, 20):
        top_shares[f"top_{count}"] = (
            sum(sorted_edges[:count]) / generated_edges
            if generated_edges
            else 0.0
        )
    return {
        "complete": len(records) == target_count,
        "evaluated_pair_count": len(records),
        "target_pair_count": target_count,
        "candidate_count": candidates,
        "true_edge_count": true_edges,
        "generated_edge_count": generated_edges,
        "true_weighted_density": true_edges / candidates if candidates else 0.0,
        "generated_weighted_density": (
            generated_edges / candidates if candidates else 0.0
        ),
        "mean_true_edges_per_pair": true_edges / len(records),
        "mean_generated_edges_per_pair": generated_edges / len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "pairs_with_one_generated_edge": sum(
            row["generated_edge_count"] == 1 for row in records
        ),
        "pairs_with_more_than_23_generated_edges": sum(
            row["generated_edge_count"] > 23 for row in records
        ),
        "pairs_with_more_than_100_generated_edges": sum(
            row["generated_edge_count"] > 100 for row in records
        ),
        "candidate_distribution": distribution(
            row["candidate_count"] for row in records
        ),
        "true_edge_distribution": distribution(
            row["true_edge_count"] for row in records
        ),
        "generated_edge_distribution": distribution(
            row["generated_edge_count"] for row in records
        ),
        "true_density_distribution": distribution(
            row["true_density"] for row in records
        ),
        "generated_density_distribution": distribution(
            row["generated_density"] for row in records
        ),
        "top_generated_edge_shares": top_shares,
        "y_histogram": dict(sorted(Counter(
            row["y_class"] for row in records
        ).items())),
        "by_y": {
            str(y_class): {
                "pair_count": len(group),
                "true_edge_count": sum(row["true_edge_count"] for row in group),
                "generated_edge_count": sum(
                    row["generated_edge_count"] for row in group
                ),
                "mean_true_edges": float(np.mean([
                    row["true_edge_count"] for row in group
                ])),
                "mean_generated_edges": float(np.mean([
                    row["generated_edge_count"] for row in group
                ])),
                "mean_generated_density": float(np.mean([
                    row["generated_density"] for row in group
                ])),
            }
            for y_class, group in sorted(y_groups.items())
        },
        "structural_diagnostics": {
            key: sum(row[key] for row in records)
            for key in (
                "intra_edges_changed",
                "x_changed",
                "asymmetric_entry_count",
                "diagonal_nonzero_count",
                "nonfinite_count",
            )
        },
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    cache_path = args.cache.resolve()
    config_path = args.sampling_config.resolve()
    output_dir = args.output_dir.resolve()
    for path in (root, checkpoint_path, cache_path, config_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / "dgd"))
    sys.path.insert(0, str(root / "higgs"))
    from higgs_utils import get_samplers

    checkpoint_hash = sha256(checkpoint_path)
    cache_hash = sha256(cache_path)
    metadata_path = output_dir / "run_metadata.json"
    metadata = {
        "root": str(root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "cache": str(cache_path),
        "cache_sha256": cache_hash,
        "sampling_config": str(config_path),
        "base_seed": args.seed,
        "pair_seed_policy": "base_seed + pair_index",
        "requested_limit": args.limit,
        "device_argument": args.device,
        "training": False,
        "data_processing": False,
    }
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("checkpoint_sha256", "cache_sha256", "base_seed"):
            if existing_metadata.get(key) != metadata[key]:
                raise RuntimeError(
                    f"Cannot resume: metadata mismatch for {key}: "
                    f"{existing_metadata.get(key)!r} != {metadata[key]!r}"
                )
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    data, slices, cache_count = load_collated_cache(cache_path)
    target_count = cache_count if args.limit == 0 else min(args.limit, cache_count)
    if target_count <= 0:
        raise ValueError("No pairs selected")

    csv_path = output_dir / "pair_results.csv"
    records: list[dict[str, Any]] = []
    completed: set[int] = set()
    if csv_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists; pass --resume or use a new directory: {csv_path}"
            )
        with csv_path.open(newline="", encoding="utf-8") as handle:
            records = [typed_row(row) for row in csv.DictReader(handle)]
        completed = {row["pair_index"] for row in records}
        if any(index >= target_count for index in completed):
            raise RuntimeError(
                "Existing output contains indices outside the selected limit"
            )

    cfg = OmegaConf.load(config_path)
    cfg.dataset.h = 1.5
    cfg.general.glue_model = str(checkpoint_path)
    cfg.general.glue_model_family = "original_glue"
    cfg.general.gpus = 0 if args.device == "cpu" else 1
    cfg, model = get_samplers(cfg)
    if model.__class__.__name__ != "DiscreteDenoisingDiffusion":
        raise RuntimeError(
            "Expected Original Glue DiscreteDenoisingDiffusion, got "
            f"{model.__class__.__name__}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model = model.to(device)
    model.eval()
    dropout_training = sum(
        isinstance(module, torch.nn.Dropout) and module.training
        for module in model.modules()
    )
    if model.training or dropout_training:
        raise RuntimeError(
            f"Glue model is not in eval mode: model.training={model.training}, "
            f"dropout_training={dropout_training}"
        )
    print(
        "Using raw Original Glue weights for full reverse: "
        f"class={model.__class__.__name__} T={model.T} device={device} "
        f"pairs={target_count} resume_completed={len(completed)}",
        flush=True,
    )

    mode = "a" if csv_path.exists() else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()
            handle.flush()
        for index in tqdm(range(target_count), desc="Real Glue full reverse"):
            if index in completed:
                continue
            sample = get_sample(data, slices, index)
            row = evaluate_pair(
                model,
                sample,
                index,
                args.seed + index,
                device,
            )
            writer.writerow(row)
            handle.flush()
            records.append(row)
            if (index + 1) % 25 == 0:
                print(
                    f"pair={index + 1}/{target_count} "
                    f"true={row['true_edge_count']} "
                    f"generated={row['generated_edge_count']} "
                    f"density={row['generated_density']:.8f}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    records.sort(key=lambda row: row["pair_index"])
    summary = summarize(records, target_count)
    payload = {
        "metadata": metadata,
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(device),
            "model_class": model.__class__.__name__,
            "diffusion_steps": int(model.T),
            "model_training": bool(model.training),
            "dropout_training_count": int(dropout_training),
        },
        "summary": summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    text_lines = [
        "Cora Original Glue real-pair full-reverse audit",
        f"checkpoint: {checkpoint_path}",
        f"cache: {cache_path}",
        f"complete: {summary['complete']}",
        f"pairs: {summary['evaluated_pair_count']}/{target_count}",
        f"candidates: {summary['candidate_count']}",
        f"true edges: {summary['true_edge_count']}",
        f"generated edges: {summary['generated_edge_count']}",
        f"true weighted density: {summary['true_weighted_density']}",
        f"generated weighted density: {summary['generated_weighted_density']}",
        f"mean true edges/pair: {summary['mean_true_edges_per_pair']}",
        f"mean generated edges/pair: {summary['mean_generated_edges_per_pair']}",
        f"precision: {summary['precision']}",
        f"recall: {summary['recall']}",
        f"top shares: {summary['top_generated_edge_shares']}",
        f"structural diagnostics: {summary['structural_diagnostics']}",
    ]
    (output_dir / "summary.txt").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8"
    )
    print("\n".join(text_lines), flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
