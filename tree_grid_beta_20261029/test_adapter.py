"""Synthetic cache layout tests; not original-server cache validation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1] / 'HiGGs-main'
FROZEN = Path('C:/path/to/HiGGs-main1/HiGGs-main/deliverables/tree_grid_partition_20261028/files')
sys.path[:0] = [str(CHECKOUT / 'dgd'), str(HERE / 'payload/tools')]
import tg_beta_common as c
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, InMemoryDataset
spec = importlib.util.spec_from_file_location('tg_synthetic_adapter', HERE / 'payload/dgd/datasets/tree_grid_beta.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class AdapterTests(unittest.TestCase):
    def test_cache_filenames_train_only_prior_bool_glue_and_no_rebuild(self):
        for name, info in c.SPECS.items():
            manifest = FROZEN / 'data_sources/splits' / (
                name + '_higgs_r{}_s1234_l0.json'.format(info['resolution']))
            frozen = json.loads(manifest.read_text())
            for stage, (h, _, _) in c.STAGES.items():
                with tempfile.TemporaryDirectory(dir=HERE) as directory:
                    root = Path(directory)
                    (root / 'processed').mkdir()
                    for split in ('train', 'val', 'test'):
                        n = 3 if split == 'train' else 8
                        width = 2 if stage == 'h2' else 1
                        x = torch.nn.functional.one_hot(torch.zeros(n, dtype=torch.long), width).float()
                        edge = torch.tensor([[0, 1], [1, 0]])
                        d = Data(x=x, edge_index=edge, edge_attr=torch.tensor([[0., 1.], [0., 1.]]),
                            y=torch.tensor([[0., 1.]]) if stage == 'glue' else torch.tensor([[1., 0.]]),
                            source_graph_id=torch.tensor([frozen['splits'][split][0]]))
                        if stage == 'glue':
                            side = torch.arange(n) < 2
                            d.pos = side[:, None] == side[None, :]
                        data, slices = InMemoryDataset.collate([d, d.clone()])
                        path = root / 'processed' / (
                            'proc_{}_v1_h{}_max{}_{}.pt'.format(split, h, info['max_size'], info['manifest_sha256']))
                        torch.save((data, slices), path)
                    cfg = OmegaConf.create({'dataset': {'name': name, 'h': float(h),
                        'max_size': info['max_size'], 'resolution': info['resolution'],
                        'split_seed': 1234, 'louvain_seed': 0, 'manifest': str(manifest), 'datadir': str(root)},
                        'train': {'batch_size': 32, 'num_workers': 0}, 'general': {'name': 'tg_fixture'}})
                    dm = adapter.FrozenCollectionDataModule(cfg)
                    dm.prepare_data()
                    infos = adapter.TreeGridBetaInfos(dm, cfg.dataset)
                    self.assertEqual(len(infos.n_nodes), 4)  # held-out 8-node fixtures must not enter prior
                    self.assertEqual(float(infos.n_nodes[3]), 1.)
                    for loader in dm.dataloaders.values():
                        if stage == 'glue':
                            self.assertEqual(next(iter(loader)).pos.dtype, torch.bool)
                    self.assertTrue(torch.isfinite(infos.nodes_dist.log_prob(torch.tensor([0, 8]))).all())
                    self.assertTrue(infos.nodes_dist.log_prob(torch.tensor([0, 8])).is_floating_point())
                with tempfile.TemporaryDirectory(dir=HERE) as directory:
                    with self.assertRaisesRegex(RuntimeError, 'rebuilding/repartitioning'):
                        adapter.FrozenHierarchyDataset(directory, manifest, 'train', float(h), info['max_size'])
                    self.assertFalse(list(Path(directory).rglob('*.pt')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
