"""Opt-in H1 features from the current thresholded noisy adjacency."""

from collections.abc import Mapping

import numpy as np
import torch


KCORE_ALGORITHM = "h1-kcore-current-state-component-v1"


def _validate_parameters(k, max_t):
    if type(k) is not int:
        raise TypeError("model.h1_kcore.k must be an integer")
    if k < 1:
        raise ValueError("model.h1_kcore.k must be at least 1")
    if type(max_t) not in (int, float):
        raise TypeError("model.h1_kcore.max_t must be a number")
    if not 0.0 <= max_t <= 1.0:
        raise ValueError("model.h1_kcore.max_t must be in [0, 1]")
    return k, float(max_t)


def h1_kcore_options(raw=None):
    """Validate the optional model config; missing config keeps old H1 intact."""
    result = {"enabled": False, "k": 3, "max_t": 0.5}
    if raw is not None:
        if not isinstance(raw, Mapping):
            raise TypeError("model.h1_kcore must be a mapping")
        unknown = set(raw) - set(result)
        if unknown:
            raise ValueError(f"Unknown model.h1_kcore options: {sorted(unknown, key=repr)}")
        result.update(raw)
    if type(result["enabled"]) is not bool:
        raise TypeError("model.h1_kcore.enabled must be a bool")
    result["k"], result["max_t"] = _validate_parameters(result["k"], result["max_t"])
    return result


@torch.no_grad()
def masked_h1_kcore_features(adjacency, node_mask, t, *, k, max_t):
    """Return X=[B,N,2], E=[B,N,N,1] for each graph with t <= max_t.

    X contains core number divided by n_valid - 1 (zero for a singleton)
    and membership in the k-core. E marks distinct nodes in the same
    connected component of the induced k-core; equal shell index alone
    is not sufficient. Padding, diagonals, and inactive graphs are zero.
    """
    k, max_t = _validate_parameters(k, max_t)
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise ValueError("adjacency must have shape [B,N,N]")
    if adjacency.is_complex():
        raise ValueError("adjacency must be real and binary")
    batch, width = adjacency.shape[:2]
    if node_mask.shape != (batch, width) or node_mask.dtype != torch.bool:
        raise ValueError("node_mask must be bool [B,N]")
    if node_mask.device != adjacency.device:
        raise ValueError("adjacency and node_mask must be on the same device")
    if t.shape not in ((batch,), (batch, 1)) or t.is_complex():
        raise ValueError("t must have shape [B] or [B,1] and be real")
    t_cpu = t.detach().reshape(batch).to("cpu")
    if not torch.isfinite(t_cpu).all() or not ((0 <= t_cpu) & (t_cpu <= 1)).all():
        raise ValueError("t must contain only finite values in [0, 1]")
    active = (t_cpu <= max_t).nonzero(as_tuple=True)[0]
    dtype = adjacency.dtype if adjacency.is_floating_point() else torch.float32
    if not active.numel():
        return (adjacency.new_zeros((batch, width, 2), dtype=dtype),
                adjacency.new_zeros((batch, width, width, 1), dtype=dtype))

    import networkx as nx

    # One batched adjacency transfer, and no dense pair-feature transfer back.
    active_on_device = active.to(adjacency.device)
    adj_cpu = adjacency.detach().index_select(0, active_on_device).to("cpu")
    if adj_cpu.dtype == torch.bfloat16:
        adj_cpu = adj_cpu.float()
    adj_array = adj_cpu.numpy()
    mask_array = node_mask.index_select(0, active_on_device).to("cpu").numpy()
    x = np.zeros((batch, width, 2),
                 dtype=np.float64 if dtype == torch.float64 else np.float32)
    labels = np.full((batch, width), -1, dtype=np.int64)
    for local, idx in enumerate(active.tolist()):
        slots = np.flatnonzero(mask_array[local])
        n = len(slots)
        if n <= 1:
            continue
        compact = adj_array[local][np.ix_(slots, slots)]
        np.fill_diagonal(compact, 0)
        if not np.all((compact == 0) | (compact == 1)):
            raise ValueError("adjacency must be binary on real off-diagonal pairs")
        if not np.array_equal(compact, compact.T):
            raise ValueError("k-core requires an undirected adjacency")
        if not compact.any():
            continue
        rows, cols = np.nonzero(np.triu(compact, k=1))
        graph = nx.Graph()
        graph.add_nodes_from(range(n))  # include isolated real nodes
        graph.add_edges_from(zip(rows.tolist(), cols.tolist()))
        cores = nx.core_number(graph)
        members = [v for v, core in cores.items() if core >= k]
        x[idx, slots, 0] = [cores[v] / (n - 1) for v in range(n)]
        x[idx, slots[members], 1] = 1
        core_graph = graph if len(members) == n else graph.subgraph(members)
        for label, component in enumerate(nx.connected_components(core_graph)):
            labels[idx, slots[list(component)]] = label

    x = torch.from_numpy(x).to(device=adjacency.device, dtype=dtype)
    labels = torch.from_numpy(labels).to(device=adjacency.device)
    row = labels.unsqueeze(-1)
    same_component = (row == labels.unsqueeze(1)) & (row >= 0)
    diagonal = torch.eye(width, dtype=torch.bool, device=adjacency.device)
    e = (same_component & ~diagonal).unsqueeze(-1).to(dtype=dtype)
    return x, e
