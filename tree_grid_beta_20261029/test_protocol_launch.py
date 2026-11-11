"""Read-only protocol, GPU ranking and baseline-comparison regression tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'payload/tools'))
import tg_beta_common as common


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load('tg_test_launch', HERE / 'launch.py')
evaluate = load('tg_test_evaluate', HERE / 'evaluate.py')
install = load('tg_test_install', HERE / 'install.py')
FROZEN = Path('C:/path/to/HiGGs-main1/HiGGs-main/deliverables/tree_grid_partition_20261028/files')


class ProtocolTests(unittest.TestCase):
    def test_exact_pinned_manifests_and_counts(self):
        for name, spec in common.SPECS.items():
            path = FROZEN / 'data_sources/splits' / (name + '_higgs_r{}_s1234_l0.json'.format(spec['resolution']))
            self.assertEqual(common.sha(path), spec['manifest_sha256'])
            manifest = json.loads(path.read_text())
            common.validate_manifest(name, manifest)
            for h, expected in spec['cache_counts'].items():
                self.assertEqual([install.cache_counts(manifest, h, s) for s in ('train', 'val', 'test')], expected)

    def test_different_split_resolution_and_dropped_graph_refused(self):
        for name, spec in common.SPECS.items():
            manifest = json.loads((FROZEN / 'data_sources/splits' / (
                name + '_higgs_r{}_s1234_l0.json'.format(spec['resolution']))).read_text())
            changed = copy.deepcopy(manifest)
            changed['settings']['resolution'] = 3
            with self.assertRaises(AssertionError):
                common.validate_manifest(name, changed)
            changed = copy.deepcopy(manifest)
            changed['splits']['test'][0] = changed['splits']['train'][0]
            with self.assertRaises(AssertionError):
                common.validate_manifest(name, changed)

    def test_choose_distinct_largest_memory_without_utilization_check(self):
        text = '0, GPU-a, 15000\n1, GPU-b, 43000\n2, GPU-c, 40000\n3, GPU-d, 44000\n'
        with patch.object(launch.subprocess, 'check_output', return_value=text) as call:
            rows = launch.gpu_inventory()
        self.assertEqual([r['index'] for r in rows], [3, 1, 2, 0])
        self.assertEqual(call.call_args.args[0][1], '--query-gpu=index,uuid,memory.free')
        plan = launch.select_gpus(('tree', 'grid'), rows)
        self.assertEqual((plan['tree']['index'], plan['grid']['index']), (3, 1))
        self.assertNotIn('utilization', ' '.join(call.call_args.args[0]))
        with self.assertRaises(RuntimeError):
            launch.select_gpus(('tree', 'grid'), [{'index': 0, 'uuid': 'x', 'free_mib': 8000}])

    def test_baseline_protocol_requires_same_manifest_scope_and_evaluator(self):
        good = {'dataset': 'tree', 'manifest_sha256': 'abc', 'generated_count': 40,
                'test_graph_count': 40, 'component_policy': 'lcc', 'table1_mode': 'tv',
                'table3_evaluation_louvain': {'resolution': 1, 'seed': 0, 'weight': None}}
        self.assertTrue(evaluate.comparable(good, 'tree', 'abc', 40))
        for key, value in [('component_policy', 'full'), ('table1_mode', 'emd'),
                           ('manifest_sha256', 'other'), ('generated_count', 20)]:
            bad = dict(good, **{key: value})
            self.assertFalse(evaluate.comparable(bad, 'tree', 'abc', 40))

    def test_all_package_python_syntax(self):
        for path in HERE.rglob('*.py'):
            if 'runtime_tmp' not in path.parts:
                compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    def test_three_formal_stages_start_before_wait_and_failure_is_not_hidden(self):
        for failure in (False, True):
            events = []
            class FakeProcess:
                def __init__(self, command, **kwargs):
                    self.stage = command[command.index('--stage') + 1]
                    self.pid = 123
                    events.append(('start', self.stage))
                def poll(self):
                    events.append(('poll', self.stage))
                    return 1 if failure and self.stage == 'h1' else 0
            with tempfile.TemporaryDirectory(dir=HERE) as directory:
                root = Path(directory)
                args = SimpleNamespace(root=str(root), dataset='tree', lock_fd=3)
                with patch.object(launch.subprocess, 'Popen', side_effect=FakeProcess), \
                     patch.object(launch, 'read_reports', return_value={}), \
                     patch.object(launch, 'common', return_value=SimpleNamespace(check_hashes=lambda _: None)):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, 'Training failed'):
                            launch.train_group(args, root, 'formal', {})
                        self.assertFalse((root / 'train/input_hash_check.json').exists())
                    else:
                        launch.train_group(args, root, 'formal', {})
                        self.assertTrue((root / 'train/input_hash_check.json').is_file())
                self.assertEqual(events[:3], [('start', 'h2'), ('start', 'h1'), ('start', 'glue')])
                self.assertEqual(set(events[3:]), {('poll', s) for s in ('h2', 'h1', 'glue')})


if __name__ == '__main__':
    unittest.main(verbosity=2)
