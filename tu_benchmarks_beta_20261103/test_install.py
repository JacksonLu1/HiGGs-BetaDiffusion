"""Read-only frozen-protocol and additive installer regression tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'payload/tools'))
import tu_beta_common as common


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


installer = load_module('qb_installer_test', HERE / 'install.py')
FROZEN = Path('C:/path/to/HiGGs-main1/HiGGs-main/deliverables/tu_benchmarks_20261102/files')
RAW_FIXTURE = Path('C:/path/to/HiGGs-main1/HiGGs-main/tmp/tu_sources_profile_20261102')


class ProtocolTests(unittest.TestCase):
    def test_bundle_tampering_is_refused(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            package = Path(directory)
            payload = package / 'payload/tools/a.py'
            payload.parent.mkdir(parents=True)
            payload.write_text('VALUE = 1\n')
            bundle = {'schema_version': 1, 'sha256': {'payload/tools/a.py': common.sha(payload)}}
            (package / 'BUNDLE.json').write_text(json.dumps(bundle))
            with patch.object(installer, 'PACKAGE', package):
                installer.verify_bundle()
                payload.write_text('VALUE = 2\n')
                with self.assertRaisesRegex(AssertionError, 'differs'):
                    installer.verify_bundle()

    @unittest.skipUnless(FROZEN.is_dir(), 'Local source bundle is unavailable; production installer verifies server copies')
    def test_exact_source_manifest_config_hashes_and_counts(self):
        for name, spec in common.SPECS.items():
            manifest_path = FROZEN / f'data_sources/splits/{name}_r1_s1234_l0.json'
            self.assertEqual(common.sha(manifest_path), spec['manifest_sha256'])
            self.assertEqual(common.sha(RAW_FIXTURE / f'{name}.json'), spec['raw_sha256'])
            self.assertEqual(common.sha(FROZEN / f'configs/dataset/{name}.yaml'), spec['config_sha256'])
            manifest = json.loads(manifest_path.read_text())
            common.validate_manifest(name, manifest)
            for h, wanted in spec['cache_counts'].items():
                self.assertEqual([installer.cache_counts(manifest, h, split)
                                  for split in ('train', 'val', 'test')], wanted)

    @unittest.skipUnless(FROZEN.is_dir(), 'Local source bundle is unavailable')
    def test_singleton_hierarchy_samples_are_not_filtered(self):
        for name, h, wanted in (('imdb_binary', '2', 98), ('nci1', '1', 1)):
            manifest = json.loads((FROZEN / f'data_sources/splits/{name}_r1_s1234_l0.json').read_text())
            samples = list(installer.expected_samples(manifest, h, 'train'))
            if h == '2':
                self.assertEqual(sum(record['nodes'] == 1 for record in samples), wanted)
            else:
                self.assertGreaterEqual(sum(record['nodes'] == 1 for record in samples), wanted)

    @unittest.skipUnless(FROZEN.is_dir(), 'Local source bundle is unavailable')
    def test_resolution_split_and_dropped_graph_changes_refused(self):
        manifest = json.loads((FROZEN / 'data_sources/splits/imdb_binary_r1_s1234_l0.json').read_text())
        for field, value in (('resolution', 2.0), ('split_seed', 0), ('louvain_seed', 7)):
            changed = copy.deepcopy(manifest)
            changed['settings'][field] = value
            with self.assertRaises(AssertionError):
                common.validate_manifest('imdb_binary', changed)
        changed = copy.deepcopy(manifest)
        changed['splits']['test'][0] = changed['splits']['train'][0]
        with self.assertRaises(AssertionError):
            common.validate_manifest('imdb_binary', changed)
        changed = copy.deepcopy(manifest)
        changed['source_graphs_dropped'] = [0]
        with self.assertRaises(AssertionError):
            common.validate_manifest('imdb_binary', changed)

    def test_target_plan_different_existing_file_refused_unchanged(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            source, target = root / 'source.py', root / 'target.py'
            source.write_text('source = True')
            target.write_text('user_content = True')
            before = target.read_bytes()
            with self.assertRaisesRegex(AssertionError, 'refusing overwrite'):
                installer.target_plan(root, {'target.py': source})
            self.assertEqual(target.read_bytes(), before)
            self.assertEqual(list(root.iterdir()), [source, target])

    def test_target_plan_existing_identical_file_reused(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            source = root / 'source.py'
            source.write_text('same = True')
            self.assertEqual(installer.target_plan(root, {'source.py': source}), {'source.py': common.sha(source)})

    def test_target_plan_escape_refused(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            source = root / 'source.py'
            source.write_text('test')
            with self.assertRaises(AssertionError):
                installer.target_plan(root, {'../escape.py': source})

    def test_training_budget_and_scope(self):
        self.assertEqual(common.SPECS['imdb_binary']['epochs'], 1000)
        for name in ('collab', 'nci1'):
            self.assertEqual(common.SPECS[name]['epochs'], 500)
        self.assertEqual(set(s['train_seed'] for s in common.SPECS.values()), {0})
        self.assertEqual(common.STAGES['h1'], ('1', 'beta_h1_twohop_mean_hog_rw', 'val/h1_total'))

    def test_all_source_syntax(self):
        for path in HERE.rglob('*.py'):
            if '__pycache__' not in path.parts:
                compile(path.read_text(encoding='utf-8'), str(path), 'exec')


class CacheAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise unittest.SkipTest('Requires the graph runtime for tensor checks')
        cls.torch, cls.Data = torch, Data

    def sample(self, h='1.5'):
        torch = self.torch
        data = self.Data(x=torch.ones((3, 1)), edge_index=torch.tensor([[0, 1], [1, 0]]),
                         edge_attr=torch.tensor([[0., 1.], [0., 1.]]), y=torch.tensor([[0., 1.]]),
                         idx=0, source_graph_id=torch.tensor([7]))
        data.pos = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
        expected = {'nodes': 3, 'edges': [(0, 1)], 'graph_id': 7, 'side': 2, 'community_id': None}
        return data, expected

    def test_exact_glue_sample(self):
        data, expected = self.sample()
        installer.verify_sample(data, expected, '1.5', 0, 4)

    def test_float_mask_or_changed_topology_refused(self):
        data, expected = self.sample()
        data.pos = data.pos.float()
        with self.assertRaises(AssertionError):
            installer.verify_sample(data, expected, '1.5', 0, 4)
        data, expected = self.sample()
        data.edge_index[0, 0] = 2
        with self.assertRaises(AssertionError):
            installer.verify_sample(data, expected, '1.5', 0, 4)

    def test_singleton_h2_is_valid_sample(self):
        torch = self.torch
        data = self.Data(x=torch.tensor([[1., 0.]]), edge_index=torch.empty((2, 0), dtype=torch.long),
                         edge_attr=torch.empty((0, 2)), y=torch.tensor([[1., 0.]]),
                         idx=0, source_graph_id=torch.tensor([8]))
        installer.verify_sample(data, {'nodes': 1, 'edges': [], 'graph_id': 8,
                                      'community_id': None, 'side': None}, '2', 0, None)


class AdapterRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime_root = Path('C:/path/to/HiGGs-GBD/HiGGs-main')
        if not runtime_root.is_dir():
            raise unittest.SkipTest('Local modified runtime is not available')
        sys.path[:0] = [str(runtime_root / 'dgd'), str(runtime_root)]
        try:
            cls.adapter = load_module('qb_adapter_runtime_test', HERE / 'payload/dgd/datasets/tu_benchmark_beta.py')
            import torch
            from torch_geometric.data import Data, InMemoryDataset
        except ImportError as error:
            raise unittest.SkipTest('Graph runtime dependency unavailable: ' + str(error))
        cls.torch, cls.Data, cls.InMemoryDataset = torch, Data, InMemoryDataset

    def singleton(self):
        torch = self.torch
        return self.Data(x=torch.ones((1, 1)), edge_index=torch.empty((2, 0), dtype=torch.long),
                         edge_attr=torch.empty((0, 2)), y=torch.tensor([[1., 0.]]),
                         idx=0, source_graph_id=torch.tensor([0]))

    def test_singleton_edge_prior_and_out_of_support_node_log_prob_finite(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            manifest = Path(directory) / 'manifest.json'
            manifest.write_text(json.dumps({'splits': {'train': [0]}}))
            datamodule = SimpleNamespace(manifest_path=manifest,
                dataloaders={'train': SimpleNamespace(dataset=[self.singleton()])})
            infos = self.adapter.TUBenchmarkBetaInfos(datamodule, SimpleNamespace(name='collab'))
            self.assertTrue(self.torch.equal(infos.edge_types, self.torch.tensor([1., 0.])))
            probs = infos.nodes_dist.log_prob(self.torch.tensor([0, 1, 2, -1]))
            self.assertTrue(self.torch.isfinite(probs).all())
            self.assertEqual(infos.training_graph_ids, [0])

    def test_cache_only_loading_and_missing_cache_refusal(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'dataset': 'collab', 'splits': {'train': [0]}}))
            digest = common.sha(manifest)
            processed = root / 'data/processed'
            processed.mkdir(parents=True)
            for split in ('train', 'val', 'test'):
                path = processed / f'proc_{split}_v1_h1_max492_{digest}.pt'
                self.torch.save(self.InMemoryDataset.collate([self.singleton(), self.singleton()]), path)
            before = {str(p): common.sha(p) for p in root.rglob('*') if p.is_file()}
            dataset = self.adapter.FrozenHierarchyDataset(root / 'data', manifest, 'train', 1, 492)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0].num_nodes, 1)
            self.assertEqual(before, {str(p): common.sha(p) for p in root.rglob('*') if p.is_file()})
            (processed / f'proc_test_v1_h1_max492_{digest}.pt').unlink()
            with self.assertRaisesRegex(RuntimeError, 'rebuilding/repartitioning is forbidden'):
                self.adapter.FrozenHierarchyDataset(root / 'data', manifest, 'train', 1, 492)


if __name__ == '__main__':
    unittest.main(verbosity=2)

