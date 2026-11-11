"""Additive, cache-only dataset adapter. No shared Planar/DD files are patched."""
import json
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from diffusion.distributions import DistributionNodes


def load_cpu(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def file_sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FrozenHierarchyDataset(InMemoryDataset):
    def __init__(self, root, manifest_path, stage, h, max_size=125):
        assert stage in ('train', 'val', 'test') and max_size == 125
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_sha256 = file_sha(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        assert self.manifest['dataset'] in ('community_small', 'enzymes')
        self.stage, self.h, self.max_size = stage, h, max_size
        super().__init__(str(root), None, None, None)
        self.data, self.slices = load_cpu(self.processed_paths[('train', 'val', 'test').index(stage)])

    @property
    def raw_file_names(self):
        return [str(self.manifest_path)]

    @property
    def processed_file_names(self):
        h = str(float(self.h)) if float(self.h) == 1.5 else str(int(self.h))
        return [f'proc_{s}_v1_h{h}_max125_{self.manifest_sha256}.pt'
                for s in ('train', 'val', 'test')]

    def _download(self):
        pass  # The installer has already copied the pinned manifest and raw file.

    def _process(self):
        if not all(Path(p).is_file() for p in self.processed_paths):
            raise RuntimeError('Missing frozen cache: rebuilding/repartitioning is forbidden')

    def process(self):
        raise RuntimeError('Frozen caches cannot be reprocessed')


class FrozenCollectionDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.manifest_path = Path(cfg.dataset.manifest).resolve()
        assert cfg.dataset.name in ('community_small', 'enzymes')
        assert cfg.dataset.resolution == 1 and cfg.dataset.split_seed == 1234
        assert cfg.dataset.louvain_seed == 0

    def prepare_data(self):
        if self.dataloaders is not None:
            return
        cfg = self.cfg
        datasets = {s: FrozenHierarchyDataset(cfg.dataset.datadir, self.manifest_path,
                    s, cfg.dataset.h, cfg.dataset.max_size) for s in ('train', 'val', 'test')}
        self.dataloaders = {s: DataLoader(ds, batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers, shuffle=(s == 'train')) for s, ds in datasets.items()}


class SmallCollectionNodePrior(DistributionNodes):
    def log_prob(self, batch_n_nodes):
        if batch_n_nodes.ndim != 1:
            raise ValueError('Expected one-dimensional node counts')
        p = self.prob.to(device=batch_n_nodes.device)
        valid = (batch_n_nodes >= 0) & (batch_n_nodes < p.numel())
        result = torch.zeros_like(batch_n_nodes, dtype=p.dtype)
        result[valid] = p[batch_n_nodes[valid]]
        return torch.log(result + 1e-30)


class SmallCollectionBetaInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.name = dataset_config.name
        self.manifest_sha256 = file_sha(datamodule.manifest_path)
        self.training_graph_ids = json.loads(datamodule.manifest_path.read_text())['splits']['train']
        train = datamodule.dataloaders['train'].dataset
        self.n_nodes = torch.zeros(max(d.num_nodes for d in train) + 1)
        self.node_types = torch.zeros(train[0].x.shape[-1])
        self.edge_types = torch.zeros(2)
        for data in train:
            n, e = data.num_nodes, data.edge_index.shape[1]
            self.n_nodes[n] += 1
            self.node_types += data.x.sum(0)
            self.edge_types += torch.tensor([n * (n - 1) - e, e])
        self.n_nodes /= self.n_nodes.sum()
        self.node_types /= self.node_types.sum()
        self.edge_types /= self.edge_types.sum()
        super().complete_infos(self.n_nodes, self.node_types)
        self.nodes_dist = SmallCollectionNodePrior(self.n_nodes)
