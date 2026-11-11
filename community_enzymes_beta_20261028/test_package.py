"""Local verification: frozen real data + CPU adapter, not GPU training success."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ORIGINAL_LOCAL = Path('C:/path/to/HiGGs-main1/HiGGs-main')
sys.path.insert(0, str(HERE / 'payload/tools'))
import sc_beta_common as c


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


installer = module('sc_installer_test', HERE / 'install.py')
launcher = module('sc_launcher_test', HERE / 'launch.py')


class ProtocolTests(unittest.TestCase):
    def test_all_python_compiles(self):
        for path in HERE.rglob('*.py'):
            if '__pycache__' not in path.parts:
                compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    def test_exact_original_manifest_bytes(self):
        for name, spec in c.SPECS.items():
            path = ORIGINAL_LOCAL / f'data_sources/splits/{name}_higgs_r1_s1234_l0.json'
            self.assertEqual(c.sha(path), spec['manifest_sha256'])
            obj = json.loads(path.read_text())
            c.validate_manifest(name, obj)
            self.assertNotEqual(obj['splits']['test'], obj['author_split_reference']['test'])

    def test_bad_split_rejected(self):
        p = ORIGINAL_LOCAL / 'data_sources/splits/community_small_higgs_r1_s1234_l0.json'
        obj = json.loads(p.read_text())
        obj['splits']['test'][0] = obj['splits']['train'][0]
        with self.assertRaises(AssertionError):
            c.validate_manifest('community_small', obj)

    def test_install_copy_plan_is_additive(self):
        items, files = installer.plans(c.ROOT, ORIGINAL_LOCAL)
        self.assertEqual(len(files), 24)
        self.assertFalse(set(files) & set(c.PROTECTED))
        self.assertTrue(all('/processed/proc_' in p or p.startswith(('configs/dataset/', 'data_sources/')) for p in files))
        for name, item in items.items():
            self.assertEqual(len(item['cache_paths']), 3)
            for group in item['cache_paths'].values():
                for value in group.values():
                    self.assertIn(name + '_higgs_r1_s1234', value)

    def test_counts_and_edgeless_h2_retained(self):
        expected = {'community_small': ((64, 128, 64), (16, 32, 16), (20, 40, 20)),
                    'enzymes': ((376, 1766, 1608), (94, 423, 382), (117, 546, 492))}
        for name, counts in expected.items():
            obj = json.loads((ORIGINAL_LOCAL / f'data_sources/splits/{name}_higgs_r1_s1234_l0.json').read_text())
            for split, row in zip(('train', 'val', 'test'), counts):
                self.assertEqual(tuple(installer.cache_counts(obj, h, split) for h in ('2', '1', '1.5')), row)

    def test_memory_only_no_utilization_or_compute_gate(self):
        with patch.object(launcher.subprocess, 'check_output', return_value='GPU-uuid, 10000\n') as call:
            self.assertEqual(launcher.memory_check(2), 'GPU-uuid')
            self.assertEqual(call.call_count, 1)
            self.assertNotIn('utilization', str(call.call_args))
        with patch.object(launcher.subprocess, 'check_output', return_value='GPU-uuid, 8191\n'):
            with self.assertRaises(RuntimeError):
                launcher.memory_check(2)


class RealCacheAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ORIGINAL_LOCAL / 'dgd'))
        cls.adapter = module('datasets.small_collection_beta', HERE / 'payload/dgd/datasets/small_collection_beta.py')

    def test_all_eighteen_original_caches(self):
        from omegaconf import OmegaConf
        import torch
        for name, spec in c.SPECS.items():
            cfg = OmegaConf.create({'dataset': {'name': name, 'h': 1, 'max_size': 125,
                'resolution': 1, 'split_seed': 1234, 'louvain_seed': 0,
                'manifest': str(ORIGINAL_LOCAL / f'data_sources/splits/{name}_higgs_r1_s1234_l0.json'),
                'datadir': str(ORIGINAL_LOCAL / f'data/{name}_higgs_r1_s1234')},
                'train': {'batch_size': 32, 'num_workers': 0}, 'general': {'name': 'adapter_unit'}})
            for stage, (h, _, _) in c.STAGES.items():
                cfg.dataset.h = 1.5 if h == '1.5' else int(h)
                dm = self.adapter.FrozenCollectionDataModule(cfg)
                dm.prepare_data()
                info = self.adapter.SmallCollectionBetaInfos(dm, cfg.dataset)
                self.assertEqual(len(dm.dataloaders['train'].dataset), spec['train_samples'][h])
                self.assertEqual(info.manifest_sha256, spec['manifest_sha256'])
                self.assertAlmostEqual(float(info.n_nodes.sum()), 1.0, places=5)
                for split, loader in dm.dataloaders.items():
                    ds = loader.dataset
                    self.assertTrue(len(ds) > 0)
                    self.assertEqual(ds[0].x.shape[1], 2 if stage == 'h2' else 1)
                    self.assertTrue(torch.equal(ds[0].y, torch.tensor([[0., 1.]]) if stage == 'glue' else torch.tensor([[1., 0.]])))
                    self.assertTrue(set(ds.data.source_graph_id.tolist()) <= set(ds.manifest['splits'][split]))
                values = info.nodes_dist.log_prob(torch.tensor([0, len(info.n_nodes) + 10]))
                self.assertTrue(values.is_floating_point() and torch.isfinite(values).all())

    def test_missing_cache_refuses_rebuild(self):
        import tempfile
        manifest = ORIGINAL_LOCAL / 'data_sources/splits/community_small_higgs_r1_s1234_l0.json'
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            with self.assertRaisesRegex(RuntimeError, 'rebuilding/repartitioning'):
                self.adapter.FrozenHierarchyDataset(temporary, manifest, 'train', 2)
            self.assertFalse(list(Path(temporary).rglob('*.pt')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
