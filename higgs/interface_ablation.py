"""Pure helpers for HiGGs sampling-interface ablations.

This module intentionally contains no diffusion, loss, threshold, or model
parameter logic.  It only validates and adapts already-discrete stage outputs.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from omegaconf import OmegaConf


_COMPONENT_POLICIES = {"full", "upstream_lcc"}
_CONDITION_MODES = {"direct", "sbm_semantic"}


def _select(cfg, key: str, default: Any) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def interface_options(cfg) -> Dict[str, Any]:
    """Read and type-check all sampling-interface options."""
    options = {
        "h2_component_policy": _select(
            cfg, "general.interface.h2_component_policy", "full"),
        "h1_component_policy": _select(
            cfg, "general.interface.h1_component_policy", "full"),
        "h2_to_h1_condition_mode": _select(
            cfg, "general.interface.h2_to_h1_condition_mode", "direct"),
        "h2_to_glue_condition_mode": _select(
            cfg, "general.interface.h2_to_glue_condition_mode", "direct"),
        "glue_present_class_index": _select(
            cfg, "general.interface.glue_present_class_index", None),
        "glue_training_cache": _select(
            cfg, "general.interface.glue_training_cache", None),
        "force_eval": _select(cfg, "general.interface.force_eval", False),
        "save_intermediates": _select(
            cfg, "general.interface.save_intermediates", False),
        "strict_assertions": _select(
            cfg, "general.interface.strict_assertions", True),
    }
    for key in ("h2_component_policy", "h1_component_policy"):
        if not isinstance(options[key], str) or options[key] not in _COMPONENT_POLICIES:
            raise ValueError(
                f"{key} must be one of {sorted(_COMPONENT_POLICIES)}, "
                f"got {options[key]!r}")
    for key in ("h2_to_h1_condition_mode",
                "h2_to_glue_condition_mode"):
        if not isinstance(options[key], str) or options[key] not in _CONDITION_MODES:
            raise ValueError(
                f"{key} must be one of {sorted(_CONDITION_MODES)}, "
                f"got {options[key]!r}")
    for key in ("force_eval", "save_intermediates", "strict_assertions"):
        if type(options[key]) is not bool:
            raise TypeError(f"{key} must be a bool, got {options[key]!r}")
    class_index = options["glue_present_class_index"]
    if class_index is not None and (
            isinstance(class_index, bool) or not isinstance(class_index, int)):
        raise TypeError(
            "glue_present_class_index must be null or an integer")
    dataset_name = str(_select(cfg, "dataset.name", _select(
        cfg, "general.name", ""))).lower()
    for key in ("h2_to_h1_condition_mode",
                "h2_to_glue_condition_mode"):
        if options[key] == "sbm_semantic" and dataset_name != "sbm":
            raise ValueError(
                f"{key}=sbm_semantic is only valid for dataset.name=sbm, "
                f"got {dataset_name!r}")
    if (options["h2_to_glue_condition_mode"] == "sbm_semantic"
            and class_index is None):
        raise ValueError(
            "interface.glue_present_class_index is required when "
            "h2_to_glue_condition_mode=sbm_semantic")
    seed = _select(cfg, "general.sampling_seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"general.sampling_seed must be an integer, got {seed!r}")
    options["sampling_seed"] = seed
    return options


def seed_sampling(seed: int) -> None:
    """Seed the sampling libraries without forcing deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except (ImportError, TypeError):
        pass
    print(f"[HiGGs interface] sampling_seed={seed}")


def validate_discrete_graph(
        node_types: torch.Tensor,
        adjacency: torch.Tensor,
        *,
        label: str,
        require_long: bool = True) -> None:
    """Validate the discrete graph contract used between HiGGs stages."""
    if not isinstance(node_types, torch.Tensor):
        raise TypeError(f"{label}: node_types must be a torch.Tensor")
    if not isinstance(adjacency, torch.Tensor):
        raise TypeError(f"{label}: adjacency must be a torch.Tensor")
    if require_long and node_types.dtype != torch.long:
        raise TypeError(
            f"{label}: node_types must be torch.long, got {node_types.dtype}")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"{label}: adjacency must be a two-dimensional square matrix")
    if node_types.ndim != 1:
        raise ValueError(f"{label}: node_types must be one-dimensional")
    if node_types.shape[0] != adjacency.shape[0]:
        raise ValueError(
            f"{label}: node/adacency size mismatch "
            f"{node_types.shape[0]} != {adjacency.shape[0]}")
    adjacency_cpu = adjacency.detach().cpu()
    if not torch.equal(adjacency_cpu, adjacency_cpu.transpose(0, 1)):
        raise ValueError(f"{label}: adjacency is not strictly symmetric")
    if adjacency_cpu.numel() and not torch.all(
            (adjacency_cpu == 0) | (adjacency_cpu == 1)):
        raise ValueError(f"{label}: adjacency contains values outside {{0, 1}}")
    if adjacency_cpu.numel() and torch.count_nonzero(
            torch.diagonal(adjacency_cpu)).item() != 0:
        raise ValueError(f"{label}: adjacency diagonal is not zero")


def _component_details(adjacency: torch.Tensor) -> Tuple[int, List[int], np.ndarray]:
    n_nodes = int(adjacency.shape[0])
    if n_nodes == 0:
        return 0, [], np.empty((0,), dtype=np.int64)
    num_components, components = sp.csgraph.connected_components(
        adjacency.detach().cpu().numpy())
    _, counts = np.unique(components, return_counts=True)
    return int(num_components), [int(value) for value in counts], components


def graph_component_audit(
        node_types: torch.Tensor,
        adjacency: torch.Tensor) -> Dict[str, Any]:
    num_components, component_sizes, _ = _component_details(adjacency)
    return {
        "sampled_valid_nodes_before_lcc": int(node_types.shape[0]),
        "component_count_before_lcc": num_components,
        "component_sizes_before_lcc": component_sizes,
        "edges_before_lcc": int(torch.triu(
            adjacency.detach().cpu(), diagonal=1).sum().item()),
    }


def apply_upstream_lcc(
        node_types: torch.Tensor,
        adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact component-selection semantics used by upstream HiGGs.

    In particular, this deliberately uses scipy connected_components followed
    by ``count.argsort()[-1:]`` and ``np.in1d`` rather than NetworkX.
    """
    validate_discrete_graph(
        node_types, adjacency, label="upstream_lcc input")
    if adjacency.shape[0] <= 1:
        out_nodes, out_adjacency = node_types, adjacency
    else:
        _, components = sp.csgraph.connected_components(
            adjacency.detach().cpu().numpy())
        _, count = np.unique(components, return_counts=True)
        subset = np.in1d(components, count.argsort()[-1:])
        subset_tensor = torch.as_tensor(subset, dtype=torch.bool,
                                        device=adjacency.device)
        out_nodes = node_types[subset_tensor]
        out_adjacency = adjacency[subset_tensor][:, subset_tensor]
    validate_discrete_graph(
        out_nodes, out_adjacency, label="upstream_lcc output")
    return out_nodes, out_adjacency


def apply_component_policy(
        node_types: torch.Tensor,
        adjacency: torch.Tensor,
        policy: str,
        *,
        label: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    validate_discrete_graph(node_types, adjacency, label=f"{label} full")
    audit = graph_component_audit(node_types, adjacency)
    if policy == "full":
        used_nodes, used_adjacency = node_types, adjacency
    elif policy == "upstream_lcc":
        used_nodes, used_adjacency = apply_upstream_lcc(
            node_types, adjacency)
    else:
        raise ValueError(f"Unknown component policy: {policy!r}")
    validate_discrete_graph(
        used_nodes, used_adjacency, label=f"{label} used")
    audit.update({
        "component_policy": policy,
        "nodes_after_lcc": int(used_nodes.shape[0]),
        "edges_after_lcc": int(torch.triu(
            used_adjacency.detach().cpu(), diagonal=1).sum().item()),
        "removed_nodes": int(node_types.shape[0] - used_nodes.shape[0]),
    })
    return used_nodes, used_adjacency, audit


def tensor_graph(node_types: torch.Tensor,
                 adjacency: torch.Tensor) -> nx.Graph:
    validate_discrete_graph(
        node_types, adjacency, label="networkx conversion")
    graph = nx.from_numpy_array(adjacency.detach().cpu().numpy())
    values = node_types.detach().cpu().tolist()
    nx.set_node_attributes(
        graph, {index: int(value) for index, value in enumerate(values)},
        "target")
    return graph


def one_hot_histogram(values: torch.Tensor) -> Dict[str, int]:
    rows = values.detach().cpu().reshape(values.shape[0], -1).tolist()
    counter = Counter(",".join(str(int(round(item))) for item in row)
                      for row in rows)
    return dict(sorted(counter.items()))


def class_histogram(values: torch.Tensor) -> Dict[str, int]:
    counter = Counter(int(value) for value in values.detach().cpu().tolist())
    return {str(key): int(counter[key]) for key in sorted(counter)}


def adapt_h2_to_h1_conditions(
        h2_node_types: torch.Tensor,
        num_classes: int,
        mode: str,
        *,
        dataset_name: str) -> torch.Tensor:
    if h2_node_types.dtype != torch.long or h2_node_types.ndim != 1:
        raise TypeError("H2 node types must be a one-dimensional long tensor")
    if mode == "direct":
        if h2_node_types.numel() and (
                h2_node_types.min().item() < 0
                or h2_node_types.max().item() >= num_classes):
            raise ValueError("H2 class index is outside the H1 condition space")
        conditions = torch.nn.functional.one_hot(
            h2_node_types, num_classes=num_classes)
    elif mode == "sbm_semantic":
        if dataset_name.lower() != "sbm":
            raise ValueError("sbm_semantic H2-to-H1 mode requires SBM")
        if num_classes < 2:
            raise ValueError(
                "SBM H1 semantic adapter requires at least two y channels")
        conditions = torch.zeros(
            (h2_node_types.shape[0], num_classes), dtype=torch.long,
            device=h2_node_types.device)
        conditions[:, 0] = 1
    else:
        raise ValueError(f"Unknown H2-to-H1 condition mode: {mode!r}")
    if conditions.shape[0] != h2_node_types.shape[0]:
        raise RuntimeError("H1 condition count does not match H2 node count")
    if conditions.numel() and not torch.all(conditions.sum(dim=-1) == 1):
        raise RuntimeError("H1 conditions are not valid one-hot rows")
    if mode == "sbm_semantic" and conditions.numel():
        expected = torch.zeros_like(conditions)
        expected[:, 0] = 1
        if not torch.equal(conditions, expected):
            raise RuntimeError(
                "SBM semantic H1 adapter produced a class other than [1,0]")
    return conditions


@lru_cache(maxsize=8)
def audit_glue_training_cache(cache_path: str) -> Dict[str, Any]:
    """Audit the collated PyG y values in a read-only Glue train cache."""
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Glue training cache not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    data, slices = torch.load(str(path), map_location="cpu")
    y_slices = slices["y"].reshape(-1).tolist()
    rows: List[Tuple[int, ...]] = []
    for start, end in zip(y_slices[:-1], y_slices[1:]):
        value = data.y[int(start):int(end)].reshape(-1)
        rows.append(tuple(int(round(item)) for item in value.tolist()))
    histogram_counter = Counter(
        ",".join(str(item) for item in row) for row in rows)
    class_counter = Counter()
    for row in rows:
        if sum(row) != 1:
            raise RuntimeError(
                f"Glue training cache contains non-one-hot y: {row}")
        class_counter[int(np.argmax(row))] += 1
    if len(class_counter) != 1:
        raise RuntimeError(
            "Cannot infer one present-edge class from Glue training cache: "
            f"{dict(class_counter)}")
    verified_class = next(iter(class_counter))
    return {
        "absolute_path": str(path),
        "sha256": digest,
        "sample_count": len(rows),
        "one_hot_histogram": dict(sorted(histogram_counter.items())),
        "class_histogram": {
            str(key): int(class_counter[key]) for key in sorted(class_counter)
        },
        "verified_present_class_index": int(verified_class),
    }


def adapt_h2_to_glue_class(
        raw_class_index: int,
        mode: str,
        *,
        configured_present_class_index: Optional[int],
        training_cache_audit: Optional[Mapping[str, Any]],
        dataset_name: str) -> int:
    if mode == "direct":
        return int(raw_class_index)
    if mode != "sbm_semantic":
        raise ValueError(f"Unknown H2-to-Glue condition mode: {mode!r}")
    if dataset_name.lower() != "sbm":
        raise ValueError("sbm_semantic H2-to-Glue mode requires SBM")
    if configured_present_class_index is None:
        raise ValueError(
            "glue_present_class_index cannot be null in sbm_semantic mode")
    if training_cache_audit is None:
        raise ValueError(
            "Glue training-cache audit is required in sbm_semantic mode")
    verified = int(training_cache_audit["verified_present_class_index"])
    if configured_present_class_index != verified:
        raise ValueError(
            "Configured Glue present class does not match the training cache: "
            f"configured={configured_present_class_index}, verified={verified}")
    return int(configured_present_class_index)


def unique_undirected_pairs(
        edge_index: torch.Tensor,
        edge_types: torch.Tensor) -> List[Tuple[int, int, int]]:
    """Keep one ordered record per undirected, non-self-loop H2 edge."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, n_edges]")
    if edge_types.shape[0] != edge_index.shape[1]:
        raise ValueError("edge type count does not match edge_index")
    seen: Dict[Tuple[int, int], int] = {}
    records = []
    for index in range(edge_index.shape[1]):
        left = int(edge_index[0, index].item())
        right = int(edge_index[1, index].item())
        edge_class = int(edge_types[index].item())
        if left == right:
            continue
        key = (min(left, right), max(left, right))
        if key in seen:
            if seen[key] != edge_class:
                raise RuntimeError(
                    "Reverse H2 edge records disagree on edge class: "
                    f"pair={key}, classes=({seen[key]}, {edge_class})")
            continue
        seen[key] = edge_class
        records.append((key[0], key[1], edge_class))
    return records


def model_sampling_state(
        model: torch.nn.Module,
        checkpoint_path: str,
        *,
        stage: str) -> Dict[str, Any]:
    modules = model.modules() if hasattr(model, "modules") else []
    dropouts = [module for module in modules
                if isinstance(module, torch.nn.modules.dropout._DropoutNd)]
    resolved_checkpoint = (
        str(Path(checkpoint_path).resolve()) if checkpoint_path
        else "<not provided>")
    state = {
        "stage": stage,
        "class_name": model.__class__.__name__,
        "checkpoint_path": resolved_checkpoint,
        "model_training": bool(getattr(model, "training", False)),
        "dropout_module_count": len(dropouts),
        "dropout_training_count": sum(
            int(module.training) for module in dropouts),
    }
    print(
        f"[HiGGs interface] {stage}: class={state['class_name']} "
        f"checkpoint={state['checkpoint_path']} "
        f"model.training={state['model_training']} "
        f"dropout_modules={state['dropout_module_count']} "
        f"dropout_training={state['dropout_training_count']}")
    return state


def inference_context(enabled: bool):
    return torch.inference_mode() if enabled else contextlib.nullcontext()


def json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8")
