"""Model-free safety, provenance, and reuse checks for the improved-only launcher."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('test_improved_launcher_module', Path(__file__).with_name('launch_improved.py'))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
mix = module.mix


class ImprovedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='launcher_fixture_', dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        self.original, self.improved = self.root / 'original', self.root / 'improved'
        self.original.mkdir()
        self.improved.mkdir()
        self.source, self.manifest = self.original / 'source.py', self.original / 'manifest.json'
        self.sample = self.improved / 'sampled_graph_0.pkl'
        for path in (self.source, self.manifest, self.sample):
            path.write_text(path.name)
        self.job = {'dataset': 'tree', 'variant': 'improved', 'count': 1, 'test_graph_ids': [160],
                    'manifest': str(self.manifest), 'manifest_sha256': mix.sha(self.manifest),
                    'samples_sha256': {str(self.sample): mix.sha(self.sample)}}
        self.plan = {'original_root': str(self.original), 'improved_root': str(self.improved),
                     'results_root': str(self.improved / 'higgs/outputs'),
                     'sources_sha256': {str(self.source): mix.sha(self.source)}, 'tool_sha256': {},
                     'evaluation_louvain': {'resolution': 1, 'seed': 0, 'weight': None},
                     'jobs': [self.job], 'pending': []}
        self.output = self.original / 'outputs/mixed_collection_tables_fixture/tree_improved'
        self.output.mkdir(parents=True)
        self.path = self.output / 'mixed_evaluation.json'
        self.report = {'protocol': mix.PROTOCOL, 'dataset': 'tree', 'variant': 'improved',
            'count': 1, 'test_graph_ids': [160], 'component_policy': 'lcc',
            'evaluation_louvain': self.plan['evaluation_louvain'], 'inputs_unchanged': True,
            'kernel_mapping': mix.KERNELS, 'mmd': {name: .1 for name in mix.METRICS},
            'input_sha256': {**self.plan['sources_sha256'], **self.job['samples_sha256'],
                             str(self.manifest): mix.sha(self.manifest)},
            'table3': [{'Metric': name, 'RealMean': 1., 'GeneratedMean': 2., 'RealStd': 0., 'GeneratedStd': .5}
                       for name in ('Nodes', 'Edges', 'Diameter', 'NC', 'MedianCommunitySize', 'Clustering', 'Density', 'Transitivity')]}
        mix.write_json(self.path, self.report)
        mix.write_csv(self.output / 'table1_mixed.csv', [{'Metric': name, 'Kernel': mix.KERNELS[name], 'Value': .1} for name in mix.METRICS])
        mix.write_csv(self.output / 'table3_summary.csv', self.report['table3'])

    def tearDown(self):
        self.tmp.cleanup()

    def save_report(self):
        self.path.write_text(json.dumps(self.report))

    def test_accept_exact_completed_and_find_under_original(self):
        found = module.completed_match(self.path, self.job, self.plan)
        self.assertEqual(found['output'], str(self.output))
        self.assertEqual(module.find_completed(self.plan, self.job), found)

    def test_input_graph_changed_refuses_reuse(self):
        self.sample.write_text('changed graph')
        self.assertIsNone(module.completed_match(self.path, self.job, self.plan))

    def test_source_changed_refuses_reuse(self):
        self.source.write_text('changed evaluator')
        self.assertIsNone(module.completed_match(self.path, self.job, self.plan))

    def test_wrong_kernel_scope_count_variant_or_ids_refuses(self):
        for key, value in (('kernel_mapping', {}), ('component_policy', 'full'), ('count', 2),
                           ('variant', 'original'), ('test_graph_ids', [161]), ('inputs_unchanged', False)):
            with self.subTest(key=key):
                before = copy.deepcopy(self.report)
                self.report[key] = value
                self.save_report()
                self.assertIsNone(module.completed_match(self.path, self.job, self.plan))
                self.report = before
        self.save_report()

    def test_missing_csv_refuses_partial(self):
        (self.output / 'table3_summary.csv').unlink()
        self.assertIsNone(module.completed_match(self.path, self.job, self.plan))

    def test_csv_mmd_mismatch_refuses(self):
        path = self.output / 'table1_mixed.csv'
        path.write_text(path.read_text().replace('0.1', '0.2'))
        self.assertIsNone(module.completed_match(self.path, self.job, self.plan))

    def test_reject_extra_graph_not_in_requested_job(self):
        extra = self.improved / 'sampled_graph_1.pkl'
        extra.write_text('not part of this arm')
        self.report['input_sha256'][str(extra)] = mix.sha(extra)
        self.save_report()
        self.assertIsNone(module.completed_match(self.path, self.job, self.plan))

    def test_reject_nonfinite_mmd(self):
        self.report['mmd']['Nodes'] = float('nan')
        self.save_report()
        self.assertIsNone(module.completed_match(self.path, self.job, self.plan))

    def test_reject_incomplete_table3_even_when_csv_agrees(self):
        self.report['table3'] = self.report['table3'][:1]
        self.save_report()
        with (self.output / 'table3_summary.csv').open('w', newline='', encoding='utf-8') as handle:
            import csv
            writer = csv.DictWriter(handle, fieldnames=list(self.report['table3'][0]))
            writer.writeheader()
            writer.writerows(self.report['table3'])
        self.assertIsNone(module.completed_match(self.path, self.job, self.plan))

    def test_summary_contains_only_requested_improved_arm_and_means(self):
        output = self.improved / 'summary'
        output.mkdir()
        records = [{'returncode': 0, 'output': str(self.output), 'reused': True}]
        module.summarize_latest(self.plan, output, records)
        import csv
        with (output / 'latest_hog_table1_mixed.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        self.assertNotIn('Original', rows[0])
        with (output / 'latest_hog_table3_means.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(rows[0]), {'Dataset', 'Metric', 'RealMean', 'GeneratedMean'})

    def test_no_launch_for_all_reused(self):
        self.job['reuse'] = module.completed_match(self.path, self.job, self.plan)
        with mock.patch.object(module.sys, 'platform', 'linux'), mock.patch.dict('sys.modules', {'fcntl': mock.Mock()}), \
                mock.patch.object(module.subprocess, 'Popen') as popen:
            module.launch(self.plan)
        popen.assert_not_called()

    def test_no_launch_for_only_pending(self):
        self.plan['jobs'] = []
        with mock.patch.object(module.sys, 'platform', 'linux'), mock.patch.dict('sys.modules', {'fcntl': mock.Mock()}), \
                mock.patch.object(module.subprocess, 'Popen') as popen:
            module.launch(self.plan)
        popen.assert_not_called()

    def test_prepare_only_improved_and_sets_modified_output(self):
        with mock.patch.object(mix, 'make_plan', return_value=self.plan) as prepare, \
                mock.patch.object(module, 'find_completed', return_value=None), \
                mock.patch.object(module, 'ROOT', self.improved), mock.patch.object(module, 'ORIGINAL', self.original):
            plan = module.prepare(['tree'])
        prepare.assert_called_once_with(self.original, self.improved, ['tree'], ['improved'])
        self.assertEqual(plan['results_root'], str(self.improved / 'higgs/outputs'))

    def test_wrong_variant_rejected_by_worker_plan(self):
        self.plan['jobs'][0]['variant'] = 'original'
        with mock.patch.object(module, 'ROOT', self.improved), mock.patch.object(module, 'ORIGINAL', self.original):
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                module.verify_plan(self.plan)

    def test_busy_shared_lock_launches_nothing_and_shows_status(self):
        self.job['reuse'] = None
        fake = mock.Mock()
        fake.LOCK_EX, fake.LOCK_NB = 2, 4
        fake.flock.side_effect = BlockingIOError()
        with mock.patch.object(module.sys, 'platform', 'linux'), mock.patch.dict('sys.modules', {'fcntl': fake}), \
                mock.patch.object(module, 'ROOT', self.improved), mock.patch.object(module, 'ORIGINAL', self.original), \
                mock.patch.object(module, 'status') as status, mock.patch.object(module.subprocess, 'Popen') as popen:
            module.launch(self.plan)
        status.assert_called_once()
        popen.assert_not_called()
        self.assertFalse((self.improved / 'higgs/outputs').exists())

    def test_worker_rejects_changed_frozen_plan_before_any_process(self):
        path = self.root / 'plan.json'
        path.write_text('{}')
        with mock.patch.object(module.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'Frozen launch plan changed'):
                module.run(path, '0' * 64, 9)
        popen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
