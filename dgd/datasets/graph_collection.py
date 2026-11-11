"""SBM-style DiGress hierarchy datasets for structure-only Planar and DD."""

from pathlib import Path

import networkx as nx
import torch
from torch.nn import functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from datasets.collection_sources import (
    SPLITS, ensure_manifest, file_sha256, load_torch_cpu, project_path,
    read_manifest, record_graph,
)


def hierarchy_samples(manifest, split, h, max_size):
    """Keep each source graph entirely inside its assigned split."""
    if h not in (1, 1.5, 2) or max_size < 2:
        raise ValueError("Expected h=1/1.5/2 and max_size>=2")
    records = {r["id"]: r for r in manifest["graphs"]}
    samples = []
    skipped = 0
    for graph_id in manifest["splits"][split]:
        graph = record_graph(records[graph_id])
        parts = manifest["partitions"][str(graph_id)]
        membership = {n: i for i, p in enumerate(parts) for n in p}
        meta = nx.Graph()
        meta.add_nodes_from(range(len(parts)))
        meta.add_edges_from((membership[u], membership[v]) for u, v in graph.edges()
                            if membership[u] != membership[v])
        if h == 2:
            if not 3 <= len(meta) <= max_size or meta.number_of_edges() == 0:
                skipped += 1
                continue
            specs = [(meta, list(meta), None, 0, 2)]
        elif h == 1:
            specs = []
            for part_id, part in enumerate(parts):
                subgraph = graph.subgraph(part)
                if not 2 <= len(part) <= max_size or not subgraph.number_of_edges():
                    skipped += 1
                    continue
                specs.append((subgraph, part, part_id, 0, 1))
        else:
            specs = []
            for a, b in sorted((min(u, v), max(u, v)) for u, v in meta.edges()):
                if any(not 2 <= len(parts[i]) <= max_size for i in (a, b)):
                    skipped += 1
                    continue
                nodes = parts[a] + parts[b]
                specs.append((graph.subgraph(nodes), nodes, len(parts[a]), 1, 1))

        for subgraph, nodes, part_info, condition, x_dim in specs:
            order = {n: i for i, n in enumerate(nodes)}
            edges = sorted((order[u], order[v]) for u, v in subgraph.edges())
            directed = sorted(edges + [(v, u) for u, v in edges])
            edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
            x = F.one_hot(torch.zeros(len(nodes), dtype=torch.long), x_dim).float()
            edge_attr = F.one_hot(torch.ones(len(directed), dtype=torch.long), 2).float()
            y = F.one_hot(torch.tensor([condition], dtype=torch.long), 2).float()
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        idx=len(samples), source_graph_id=torch.tensor([graph_id]))
            if h == 1:
                data.community_id = torch.tensor([part_info])
            if h == 1.5:
                # Membership comes from the frozen partition, not connected
                # components: a Louvain part may itself be disconnected.
                side = torch.arange(len(nodes)) < part_info
                intra = side[:, None] == side[None, :]
                data.pos = intra
            samples.append(data)
    if not samples:
        raise ValueError(f"No {split} h={h} samples; inspect frozen partitions/max_size")
    if h == 1.5:
        # PyG concatenates pos along rows, so all masks in a split need the
        # same width. Use the actual largest pair rather than 2 * max_size.
        width = max(data.num_nodes for data in samples)
        for data in samples:
            padding = width - data.num_nodes
            data.pos = F.pad(data.pos, (0, padding, 0, padding), value=False)
    print(f"{manifest['dataset']} {split} h={h}: {len(samples)} samples; skipped={skipped}")
    return samples


class GraphCollectionDataset(InMemoryDataset):
    def __init__(self, root, manifest_path, stage, h=1, max_size=500):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = read_manifest(self.manifest_path)
        self.manifest_sha256 = file_sha256(self.manifest_path)
        self.h, self.max_size = h, max_size
        self.stage = stage
        if stage not in SPLITS:
            raise ValueError(f"Unknown split {stage}")
        super().__init__(str(root), None, None, None)
        self.data, self.slices = load_torch_cpu(self.processed_paths[SPLITS.index(stage)])

    @property
    def raw_file_names(self):
        return [str(self.manifest_path)]

    @property
    def processed_file_names(self):
        tag = f"v1_h{self.h}_max{self.max_size}_{self.manifest_sha256}"
        return [f"proc_{split}_{tag}.pt" for split in SPLITS]

    def process(self):
        # Validate every split before writing any hierarchy cache.
        datasets = [hierarchy_samples(self.manifest, s, self.h, self.max_size) for s in SPLITS]
        for samples, path in zip(datasets, self.processed_paths):
            torch.save(self.collate(samples), path)


class GraphCollectionDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.root = project_path(cfg.dataset.datadir)
        self.manifest_path = project_path(cfg.dataset.manifest)

    def prepare_data(self):
        if self.dataloaders is not None:
            return
        cfg = self.cfg
        if cfg.dataset.regime != "majority":
            raise ValueError("Planar/DD structure-only baseline supports regime=majority only")
        ensure_manifest(cfg.dataset.name, project_path(cfg.dataset.source), self.manifest_path,
                        cfg.dataset.resolution, cfg.dataset.split_seed, cfg.dataset.louvain_seed)
        datasets = {s: GraphCollectionDataset(self.root, self.manifest_path, s,
                                             cfg.dataset.h, cfg.dataset.max_size) for s in SPLITS}
        self.dataloaders = {s: DataLoader(d, batch_size=cfg.train.batch_size,
                                         num_workers=cfg.train.num_workers,
                                         shuffle=(s == "train")) for s, d in datasets.items()}


class GraphCollectionInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.name = dataset_config.name
        self.manifest_sha256 = file_sha256(datamodule.manifest_path)
        self.training_graph_ids = read_manifest(datamodule.manifest_path)["splits"]["train"]
        # Estimate priors from train only; val/test stay reserved for evaluation.
        datasets = datamodule.dataloaders["train"].dataset
        node_dim = datasets[0].x.shape[-1]
        self.node_types = torch.zeros(node_dim)
        self.edge_types = torch.zeros(2)
        max_n = max(d.num_nodes for d in datasets)
        self.n_nodes = torch.zeros(max_n + 1)
        for data in datasets:
            n, e = data.num_nodes, data.edge_index.shape[1]
            self.n_nodes[n] += 1
            self.node_types += data.x.sum(0)
            self.edge_types += torch.tensor([n * (n - 1) - e, e])
        self.n_nodes /= self.n_nodes.sum()
        self.node_types /= self.node_types.sum()
        self.edge_types /= self.edge_types.sum()
        super().complete_infos(self.n_nodes, self.node_types)
