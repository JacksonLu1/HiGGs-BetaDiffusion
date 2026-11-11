"""Portable, frozen whole-graph splits and training partitions for Planar/DD."""

import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import networkx as nx
import numpy as np
import torch


SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_torch_cpu(path):
    # Only load the user's trusted local SPECTRE/PyG files, never remote pickles.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_source(name, source):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Place the original {name} file at {source}")
    if name == "planar":
        payload = load_torch_cpu(source)
        if not isinstance(payload, (tuple, list)) or len(payload) != 8:
            raise ValueError("Planar requires the original eight-field SPECTRE file")
        records = []
        for graph_id, adjacency in enumerate(payload[0]):
            adjacency = torch.as_tensor(adjacency).detach().cpu()
            if (tuple(adjacency.shape) != (64, 64)
                    or not torch.isfinite(adjacency).all()
                    or not ((adjacency == 0) | (adjacency == 1)).all()
                    or not torch.equal(adjacency, adjacency.T)
                    or torch.count_nonzero(adjacency.diag())):
                raise ValueError(f"Invalid Planar adjacency at graph {graph_id}")
            graph = nx.from_numpy_array(adjacency.numpy())
            if not nx.is_connected(graph) or not nx.check_planarity(graph)[0]:
                raise ValueError(f"Planar graph {graph_id} is disconnected/nonplanar")
            records.append((graph_id, graph))
        if len(records) != 200:
            raise ValueError(f"Expected 200 Planar graphs, got {len(records)}")
        return records
    if name != "dd":
        raise ValueError(f"Unsupported graph collection: {name}")

    with ZipFile(source) as archive:
        def read_ints(basename):
            matches = [n for n in archive.namelist()
                       if n.replace("\\", "/").split("/")[-1] == basename]
            if len(matches) != 1:
                raise ValueError(f"Missing/ambiguous DD ZIP member: {basename}")
            with archive.open(matches[0]) as handle:
                return np.loadtxt(handle, delimiter=",", dtype=np.int64)

        indicators = read_ints("DD_graph_indicator.txt").reshape(-1)
        labels = read_ints("DD_graph_labels.txt").reshape(-1)
        node_labels = read_ints("DD_node_labels.txt").reshape(-1)
        edges = read_ints("DD_A.txt").reshape(-1, 2)
    if (len(labels) != 1178 or len(indicators) != len(node_labels)
            or not np.array_equal(np.unique(indicators), np.arange(1, 1179))
            or edges.min() < 1 or edges.max() > len(indicators)):
        raise ValueError("DD source dimensions do not match the original TU archive")
    if np.any(indicators[edges[:, 0] - 1] != indicators[edges[:, 1] - 1]):
        raise ValueError("DD contains edges across source graphs")

    # Match SPECTRE's structure-only Protein subset: build from edges, remove
    # isolates, remove self-loops, then select 100..500 nodes; do not take LCC.
    union = nx.from_edgelist(edges.tolist())
    union.remove_nodes_from(list(nx.isolates(union)))
    union.remove_edges_from(list(nx.selfloop_edges(union)))
    records = []
    for graph_id in range(1, len(labels) + 1):
        node_ids = (np.flatnonzero(indicators == graph_id) + 1).tolist()
        graph = union.subgraph(node_ids)
        if 100 <= len(graph) <= 500:
            mapping = {node: i for i, node in enumerate(sorted(graph))}
            records.append((graph_id, nx.relabel_nodes(graph, mapping).copy()))
    if len(records) != 918:
        raise ValueError(f"Expected 918 DD graphs after filtering, got {len(records)}")
    return records


def graph_record(graph_id, graph):
    return {"id": int(graph_id), "n": len(graph),
            "edges": sorted([min(u, v), max(u, v)] for u, v in graph.edges())}


def record_graph(record):
    graph = nx.Graph()
    graph.add_nodes_from(range(record["n"]))
    graph.add_edges_from(record["edges"])
    nx.set_node_attributes(graph, 0, "target")
    return graph


def split_graph_ids(ids, seed):
    n_test = int(round(len(ids) * 0.2))
    n_train = int(round((len(ids) - n_test) * 0.8))
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [ids[i] for i in order]
    return {"train": shuffled[:n_train], "val": shuffled[n_train:-n_test],
            "test": shuffled[-n_test:]}


def validate_manifest(manifest):
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported hierarchy manifest schema")
    if manifest.get("dataset") not in ("planar", "dd"):
        raise ValueError("Unknown dataset in hierarchy manifest")
    records = manifest["graphs"]
    ids = [r["id"] for r in records]
    split_ids = [i for split in SPLITS for i in manifest["splits"][split]]
    if len(ids) != len(set(ids)) or sorted(split_ids) != sorted(ids):
        raise ValueError("Graph splits overlap, omit IDs or contain unknown IDs")
    if any(not manifest["splits"][split] for split in SPLITS):
        raise ValueError("Every whole-graph split must be nonempty")
    if set(manifest["partitions"]) != {str(i) for i in ids}:
        raise ValueError("Partitions must correspond exactly to source graph IDs")
    for record in records:
        n = record["n"]
        edges = record["edges"]
        if (n < 2 or len({tuple(e) for e in edges}) != len(edges)
                or any(len(e) != 2 or not 0 <= e[0] < e[1] < n for e in edges)):
            raise ValueError(f"Invalid topology for source graph {record['id']}")
        parts = manifest["partitions"][str(record["id"])]
        nodes = [i for part in parts for i in part]
        if not parts or any(not p for p in parts) or sorted(nodes) != list(range(n)):
            raise ValueError(f"Invalid partition coverage for graph {record['id']}")


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def ensure_manifest(name, source, path, resolution=2, split_seed=1234, louvain_seed=0):
    path, source = Path(path), Path(source)
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Louvain resolution must be finite and positive")
    settings = {"resolution": float(resolution), "split_seed": int(split_seed),
                "louvain_seed": int(louvain_seed), "weight": None,
                "node_labels": "dummy", "graph_condition": "constant"}
    if path.is_file():
        manifest = read_manifest(path)
        if manifest["dataset"] != name or manifest["settings"] != settings:
            raise ValueError(f"Manifest settings differ; choose a new manifest path: {path}")
        if source.is_file() and file_sha256(source) != manifest["source_sha256"]:
            raise ValueError(f"Source hash differs from frozen manifest: {source}")
        print(f"Reusing frozen {name} hierarchy: {path}")
        return manifest

    digest = file_sha256(source)
    records = read_source(name, source)
    # Split whole graphs before constructing any hierarchy. Original source IDs
    # and canonical local node IDs remain stable across machines and versions.
    splits = split_graph_ids([i for i, _ in records], split_seed)
    partitions = {}
    print(f"Finding communities once per {name} graph: resolution={resolution}, seed={louvain_seed}")
    for graph_id, graph in records:
        parts = nx.community.louvain_communities(
            graph, weight=None, resolution=resolution, seed=louvain_seed)
        partitions[str(graph_id)] = sorted([sorted(p) for p in parts], key=lambda p: p[0])
    manifest = {"schema_version": SCHEMA_VERSION, "dataset": name,
                "source_sha256": digest, "source_name": source.name,
                "networkx_version": nx.__version__, "settings": settings,
                "graphs": [graph_record(i, g) for i, g in records],
                "splits": splits, "partitions": partitions}
    validate_manifest(manifest)
    if file_sha256(source) != digest:
        raise ValueError("Source changed during preprocessing")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
    return manifest
