"""CPU/fixture recovery tests; no real training, GPU, server or job signals."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ego_recovery_runner_test', HERE / 'run_recovery.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_prefix(sample, count=31):
    import networkx as nx
    directory = sample / 'sampling'
    directory.mkdir(parents=True)
    rows = []
    for index in range(count):
        graph = nx.path_graph(3)
        nx.set_node_attributes(graph, 0, 'target')
        nx.set_edge_attributes(graph, 1, 'weight')
        path = directory / f'sampled_graph_{index}.pkl'
        with path.open('wb') as handle:
            pickle.dump(graph, handle)
        rows.append({'index': index, 'sha256': driver.sha(path), 'nodes': 3, 'edges': 2})
    dump(sample / 'sampling_progress.json', rows)
    return rows


class PrefixTests(unittest.TestCase):
    def test_exact_trusted_prefix_and_no_mutation(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            sample = Path(directory)
            make_prefix(sample)
            before = {str(p): driver.sha(p) for p in sample.rglob('*') if p.is_file()}
            rows = driver.checked_prefix(sample)
            self.assertEqual(len(rows), 31)
            self.assertEqual([r['index'] for r in rows], list(range(31)))
            driver.check_hashes(before)

    def test_missing_extra_changed_or_out_of_order_prefix_refused(self):
        for mode in ('missing', 'extra', 'changed', 'order'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=HERE) as directory:
                sample = Path(directory)
                rows = make_prefix(sample)
                if mode == 'missing':
                    (sample / 'sampling/sampled_graph_3.pkl').unlink()
                elif mode == 'extra':
                    (sample / 'sampling/sampled_graph_31.pkl').write_bytes(b'extra')
                elif mode == 'changed':
                    (sample / 'sampling/sampled_graph_3.pkl').write_bytes(b'changed')
                else:
                    rows[0], rows[1] = rows[1], rows[0]
                    dump(sample / 'sampling_progress.json', rows)
                with self.assertRaises(RuntimeError):
                    driver.checked_prefix(sample)

    def test_replay_requires_semantics_and_identical_sha(self):
        import networkx as nx
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            sample = Path(directory) / 'old'
            make_prefix(sample)
            prefix = driver.checked_prefix(sample)
            old = Path(prefix[0]['path'])
            generated = Path(directory) / 'generated.pkl'
            generated.write_bytes(old.read_bytes())
            with old.open('rb') as handle:
                graph = pickle.load(handle)
            result = driver.verify_replayed(0, generated, graph, prefix)
            self.assertTrue(result['semantic_equal'] and result['sha256_equal'])
            altered = graph.copy()
            altered.nodes[0]['target'] = 7
            with self.assertRaisesRegex(RuntimeError, 'Replay differs'):
                driver.verify_replayed(0, generated, altered, prefix)
            generated.write_bytes(old.read_bytes() + b' ')
            with self.assertRaisesRegex(RuntimeError, 'Replay differs'):
                driver.verify_replayed(0, generated, graph, prefix)


class PreflightTests(unittest.TestCase):
    def fixture(self, root):
        failed = root / 'outputs' / driver.FAILED_NAME
        training = failed / 'train'
        sample = root / 'higgs/outputs' / driver.FAILED_NAME / 'sample'
        item = {'manifest_sha256': driver.MANIFEST, 'splits': {'test': 151}}
        source = root / 'frozen_input.txt'
        source.write_text('preserved')
        before = {str(source): driver.sha(source)}
        reports = {}
        for stage in ('h2', 'h1', 'glue'):
            folder = training / stage
            folder.mkdir(parents=True)
            best, last = folder / 'best.ckpt', folder / 'last.ckpt'
            best.write_bytes(b'best')
            last.write_bytes(b'last')
            reports[stage] = {'best': str(best), 'last': str(last),
                              'best_sha256': driver.sha(best), 'last_sha256': driver.sha(last)}
            dump(folder / 'formal_result.json', reports[stage])
        dump(training / 'input_hash_check.json', {'unchanged': True, 'sha256': before})
        dump(failed / 'pipeline_status.json', {'dataset': 'ego', 'status': 'failed', 'phase': 'sampling',
             'training_output': str(training), 'sampling_output': str(sample),
             'gpu': {'index': 2, 'uuid': 'GPU-test'}})
        (failed / 'sample.log').write_text('torch.linalg.eigh failed to converge')
        make_prefix(sample)
        dump(sample / 'input_manifest.json', {'dataset': 'ego', 'manifest_sha256': driver.MANIFEST,
             'requested_samples': 151, 'real_test_graphs': 151, 'exploratory': False,
             'training_run': str(training), 'checkpoints': reports, 'source_sha256': before,
             'protocol': 'seed1234 once; strict saved-config loading; full H2/H1 retained'})
        for path in (root / 'tools/qb_beta_sample.py',
                     root / 'four_benchmarks_beta_20261101/evaluate.py',
                     root / 'new_package/stable_eigen.py', root / 'new_package/run_recovery.py'):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('pass\n')
        dump(root / 'new_package/BUNDLE.json', {'schema_version': 1, 'sha256': {
            path.name: driver.sha(path) for path in (root / 'new_package').glob('*.py')}})
        common = SimpleNamespace(baseline=Mock(return_value=before),
            get_item=Mock(return_value=item), check_hashes=driver.check_hashes)
        sampler = SimpleNamespace(read_reports=Mock(return_value=reports))
        return failed, sample, common, sampler

    def test_success_keeps_old_inputs_and_selects_exact_reports(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory).resolve()
            failed, sample, common, sampler = self.fixture(root)
            old_hashes = {str(p): driver.sha(p) for p in root.rglob('*') if p.is_file()}
            gpu = {'index': 2, 'uuid': 'GPU-test', 'free_mib': 8192}
            with patch.object(driver, 'ROOT', root), patch.object(driver, 'PACKAGE', root / 'new_package'), \
                 patch.object(driver.sys, 'platform', 'linux'), \
                 patch.object(driver, 'load_sampler', return_value=(common, sampler)), \
                 patch.object(driver, 'gpu_info', return_value=gpu), \
                 patch.object(driver, 'runtime_environment', return_value={'pytorch': 'test'}):
                request = driver.preflight(root, failed, 2)
            self.assertEqual(len(request['prefix']), 31)
            self.assertEqual(request['gpu'], gpu)
            self.assertFalse(request['policy']['rng_snapshots'].startswith('resume'))
            sampler.read_reports.assert_called_once_with(failed / 'train', 'ego', driver.MANIFEST, 'formal')
            driver.check_hashes(old_hashes)

    def test_active_status_or_wrong_gpu_refused(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory).resolve()
            failed, sample, common, sampler = self.fixture(root)
            with patch.object(driver, 'ROOT', root), patch.object(driver, 'PACKAGE', root / 'new_package'), \
                 patch.object(driver.sys, 'platform', 'linux'), \
                 patch.object(driver, 'load_sampler', return_value=(common, sampler)):
                with self.assertRaisesRegex(RuntimeError, 'physical GPU2'):
                    driver.preflight(root, failed, 3)
                state = driver.read_json(failed / 'pipeline_status.json')
                state['status'] = 'running'
                dump(failed / 'pipeline_status.json', state)
                with self.assertRaisesRegex(RuntimeError, 'confirmed failed'):
                    driver.preflight(root, failed, 2)

    def test_different_failed_checkpoint_selection_refused(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory).resolve()
            failed, sample, common, sampler = self.fixture(root)
            record = driver.read_json(sample / 'input_manifest.json')
            record['checkpoints']['h1']['best_sha256'] = '0' * 64
            dump(sample / 'input_manifest.json', record)
            with patch.object(driver, 'ROOT', root), patch.object(driver, 'PACKAGE', root / 'new_package'), \
                 patch.object(driver.sys, 'platform', 'linux'), \
                 patch.object(driver, 'load_sampler', return_value=(common, sampler)):
                with self.assertRaisesRegex(RuntimeError, 'checkpoint selection'):
                    driver.preflight(root, failed, 2)


class RngAndLockTests(unittest.TestCase):
    def test_snapshot_preserves_python_numpy_torch_rng(self):
        import numpy as np
        import torch
        random.seed(9)
        np.random.seed(8)
        torch.manual_seed(7)
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory(dir=HERE) as directory, \
             patch.object(torch.cuda, 'get_rng_state_all', return_value=[]):
            path = Path(directory) / 'rng/after_0000.pt'
            driver.save_rng_snapshot(torch, path, 0)
            self.assertTrue(path.is_file())
        self.assertEqual(py_state, random.getstate())
        self.assertEqual(np_state[0], np.random.get_state()[0])
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertEqual(np_state[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_probe_releases_sampler_lock_without_signals(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            (root / 'logs/sampling').mkdir(parents=True)
            (root / 'logs/.four_benchmarks_beta_ego.lock').touch()
            (root / 'logs/sampling/ego_mean_hog_fullchain.lock').touch()
            calls = []
            fake = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda fd, mode: calls.append(fd))
            with patch.dict(sys.modules, {'fcntl': fake}), patch.object(os_module(), 'kill') as kill:
                lock = driver.acquire_lock(root)
                self.assertFalse(lock.closed)
                self.assertEqual(len(calls), 2)
                with self.assertRaises(OSError):
                    driver.os.fstat(calls[1])  # sampler probe descriptor is closed, not inherited.
                kill.assert_not_called()
                lock.close()

    def test_busy_original_lock_is_refused(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            (root / 'logs').mkdir()
            (root / 'logs/.four_benchmarks_beta_ego.lock').touch()
            fake = SimpleNamespace(LOCK_EX=1, LOCK_NB=2,
                                   flock=Mock(side_effect=BlockingIOError()))
            with patch.dict(sys.modules, {'fcntl': fake}):
                with self.assertRaisesRegex(RuntimeError, 'wait, do not kill'):
                    driver.acquire_lock(root)


class RuntimeHookTests(unittest.TestCase):
    def exercise(self, mismatch=False):
        import inspect
        import networkx as nx
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory).resolve()
            old_sample = root / 'old/sample'
            make_prefix(old_sample)
            prefix = driver.checked_prefix(old_sample)
            output = root / 'new'
            output.mkdir()
            request = {'root': str(root), 'training_output': str(root / 'training'),
                       'prefix': prefix, 'before_sha256': {}}
            fake_sample = ModuleType('sample_main')
            fake_models = ModuleType('four_benchmark_models')
            fake_common = ModuleType('qb_beta_common')
            fake_stable = ModuleType('stable_eigen')
            calls = []
            sentinel = object()
            class BaseGlue:
                def __init__(self, cfg, dataset_infos, train_metrics=None, sampling_metrics=None,
                             visualization_tools=None, extra_features=None, domain_features=None):
                    self.cfg = cfg
                    self.extra_features = SimpleNamespace(eigenfeatures=sentinel)
            classes = {'h1': object(), 'h2': object(), 'glue': BaseGlue}
            def select(stage):
                return classes[stage]
            fake_models.FourBenchmarkGlue, fake_models.model_class = BaseGlue, select
            fake_common.baseline = lambda root_arg, dataset: {}
            prior_baseline = fake_common.baseline
            def make_guarded(existing, callback, diagnostic_dir=None):
                self.assertIs(existing, sentinel)
                self.assertEqual(diagnostic_dir, output / 'diagnostics')
                calls.append('guarded_glue')
                return 'scoped_guard'
            fake_stable.make_guarded_eigenfeatures = make_guarded
            def sample_graph(*args, **kwargs):
                index = kwargs['graph_number']
                path = Path.cwd() / f'sampled_graph_{index}.pkl'
                if index < 31:
                    path.write_bytes(Path(prefix[index]['path']).read_bytes())
                    with path.open('rb') as handle:
                        graph = pickle.load(handle)
                else:
                    graph = nx.path_graph(3)
                    with path.open('wb') as handle:
                        pickle.dump(graph, handle)
                if mismatch and index == 0:
                    graph.nodes[0]['target'] = 99
                return graph, graph.copy()
            fake_sample.sample_large_graph = sample_graph
            def fake_run_path(filename, run_name):
                self.assertEqual(run_name, '__main__')
                self.assertIs(fake_models.model_class('h1'), classes['h1'])
                self.assertIs(fake_models.model_class('h2'), classes['h2'])
                ctor = fake_models.model_class('glue')
                signature = inspect.signature(ctor.__init__)
                self.assertIn('cfg', signature.parameters)
                self.assertIn('dataset_infos', signature.parameters)
                loaded = ctor(SimpleNamespace(dataset=SimpleNamespace(name='ego')), object())
                self.assertEqual(loaded.extra_features.eigenfeatures, 'scoped_guard')
                fake_common.baseline(root, 'ego')
                sample_dir = output / 'sample/sampling'
                sample_dir.mkdir(parents=True)
                os_module().chdir(sample_dir)
                for index in range(151):
                    fake_sample.sample_large_graph(graph_number=index)
            previous_cwd, previous_argv = Path.cwd(), sys.argv[:]
            try:
                with patch.dict(sys.modules, {'sample_main': fake_sample,
                    'four_benchmark_models': fake_models, 'qb_beta_common': fake_common,
                    'stable_eigen': fake_stable}), \
                    patch.object(driver.runpy, 'run_path', side_effect=fake_run_path), \
                    patch.object(driver, 'save_rng_snapshot') as snapshot:
                    if mismatch:
                        with self.assertRaisesRegex(RuntimeError, 'Replay differs'):
                            driver.run_sampling(request, output)
                        self.assertEqual(snapshot.call_count, 0)
                        self.assertFalse((output / 'recovery_sampling_complete.json').exists())
                    else:
                        driver.run_sampling(request, output)
                        self.assertEqual(snapshot.call_count, 151)
                        report = driver.read_json(output / 'recovery_sampling_complete.json')
                        self.assertEqual(report['count'], 151)
                        self.assertEqual(report['prefix_verified_count'], 31)
                        self.assertFalse(report['rng_resume_implemented'])
                        self.assertEqual(len(driver.read_json(output / 'prefix_replay_audit.json')), 31)
                self.assertEqual(calls, ['guarded_glue'])
                self.assertIs(fake_models.model_class, select)
                self.assertIs(fake_sample.sample_large_graph, sample_graph)
                self.assertIs(fake_common.baseline, prior_baseline)
                self.assertEqual(sys.argv, previous_argv)
                driver.check_hashes({r['path']: r['sha256'] for r in prefix})
            finally:
                os_module().chdir(previous_cwd)

    def test_full151_hooked_replay_preserves31_and_restores_runtime(self):
        self.exercise()

    def test_prefix_mismatch_stops_before_continuation_and_restores_runtime(self):
        self.exercise(mismatch=True)


def os_module():
    return driver.os


class ComparisonTests(unittest.TestCase):
    def fixture(self, base):
        original = base / 'original'
        old = original / 'outputs/small_benchmarks_tables_20261102_000000/ego'
        output = base / 'recovery'
        new = output / 'evaluation'
        expected = {'Nodes': 'Gaussian-TV', 'Degree': 'Gaussian-EMD', 'Clustering': 'Gaussian-EMD',
                    'Eccentricity': 'Gaussian', 'Spectre': 'Gaussian-EMD'}
        p = {'dataset': 'ego', 'protocol': 'higgs_frozen_mixed_v1', 'manifest_sha256': driver.MANIFEST,
             'test_graph_ids': list(range(151)), 'component_policy': 'lcc',
             'evaluation_louvain': {'resolution': 1, 'seed': 0, 'weight': None},
             'edgeless_community_policy': 'one_community_per_node',
             'diameter_policy': 'largest_component_per_graph',
             'training_partitions_modified': False, 'learning_scope': 'topology_only',
             'samples_sha256': {str(i): str(i) for i in range(151)}}
        for folder, value in ((old, .2), (new, .1)):
            rows1 = [{'Metric': name, 'Kernel': kernel, 'Value': value} for name, kernel in expected.items()]
            protocol = dict(p, table1=rows1)
            if folder == new:
                protocol['count'] = 151
            dump(folder / 'protocol.json', protocol)
            write_csv(folder / 'table1_mixed.csv', rows1)
            write_csv(folder / 'table3_summary.csv', [
                {'Metric': name, 'RealMean': 2., 'RealStd': 0., 'GeneratedMean': 2. + value, 'GeneratedStd': 0.}
                for name in ('Nodes', 'Edges', 'Diameter', 'NC', 'MedianCommunitySize', 'Clustering', 'Density', 'Transitivity')])
            (folder / 'table3_per_graph.csv').write_text('present\n')
        return original, old, output, new

    def test_original_legacy_schema_without_count_matches(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            original, old, output, new = self.fixture(Path(directory))
            before = {str(p): driver.sha(p) for p in old.rglob('*') if p.is_file()}
            with patch.object(driver, 'ORIGINAL', original):
                driver.compare_existing(output)
            status = driver.read_json(output / 'comparison_status.json')
            self.assertEqual(status['status'], 'complete')
            self.assertTrue((output / 'table1_original_vs_recovery.csv').is_file())
            self.assertTrue((output / 'table3_original_vs_recovery_means.csv').is_file())
            driver.check_hashes(before)

    def test_missing_baseline_is_not_recomputed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            original, old, output, new = self.fixture(Path(directory))
            changed = driver.read_json(old / 'protocol.json')
            changed['manifest_sha256'] = 'wrong'
            dump(old / 'protocol.json', changed)
            with patch.object(driver, 'ORIGINAL', original):
                driver.compare_existing(output)
            self.assertEqual(driver.read_json(output / 'comparison_status.json')['status'],
                             'original_baseline_not_found')
            self.assertFalse((output / 'table1_original_vs_recovery.csv').exists())

    def test_kernel_or_real_reference_difference_refused(self):
        for mode in ('kernel', 'real'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=HERE) as directory:
                original, old, output, new = self.fixture(Path(directory))
                if mode == 'kernel':
                    values = list(driver.table_rows(new / 'table1_mixed.csv').values())
                    values[1]['Kernel'] = 'Gaussian-TV'
                    write_csv(new / 'table1_mixed.csv', values)
                else:
                    values = list(driver.table_rows(new / 'table3_summary.csv').values())
                    values[1]['RealMean'] = 5.
                    write_csv(new / 'table3_summary.csv', values)
                with patch.object(driver, 'ORIGINAL', original), self.assertRaises(RuntimeError):
                    driver.compare_existing(output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
