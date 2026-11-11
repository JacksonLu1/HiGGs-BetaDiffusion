"""Synthetic JSON checkpoint fixtures for copy/metadata control, not real weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('freeze_metadata_helper', HERE / 'run_early.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class MetadataTests(unittest.TestCase):
    def test_copy_best_and_last_without_forging_h1_formal500(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory).resolve()
            run = root / 'outputs' / helper.RUN
            snapshot = root / 'outputs/dd_earlystop_snapshot_fixture'
            snapshot.mkdir(parents=True)
            originals = {}
            selected = {'h2': 439, 'h1': 109, 'glue': 399}
            saved = {'h2': 499, 'h1': 329, 'glue': 499}
            for stage in selected:
                folder = run / stage / 'checkpoints' / ('dd_' + stage + '_mean_hog_formal')
                folder.mkdir(parents=True)
                best = folder / ('epoch={}.ckpt'.format(selected[stage]))
                last = folder / 'last.ckpt'
                best.write_text(json.dumps({'epoch': selected[stage], 'global_step': selected[stage] * 10}))
                last.write_text(json.dumps({'epoch': saved[stage], 'global_step': saved[stage] * 10,
                    'callbacks': {'monitor': {'monitor': helper.MONITORS[stage],
                                             'best_model_score': .8, 'best_model_path': str(best)}}}))
                if stage != 'h1':
                    report = {'stage': stage, 'epoch': 499, 'manifest_sha256': helper.DIGEST,
                        'strict_reload': True, 'strict_best_reload': True, 'monitor': helper.MONITORS[stage],
                        'best': str(best), 'last': str(last), 'best_sha256': helper.sha(best),
                        'last_sha256': helper.sha(last)}
                    (run / stage / 'formal_result.json').write_text(json.dumps(report))
                originals.update({str(p): helper.sha(p) for p in (best, last)})
            def fixture_load(path):
                return json.loads(Path(path).read_text())
            def fixture_reload(path, stage, manifest):
                state = fixture_load(path)
                return {'epoch': state['epoch'], 'global_step': state['global_step'], 'strict_reload': True}
            with patch.object(helper, 'load_cpu', side_effect=fixture_load), \
                 patch.object(helper, 'strict_best', side_effect=fixture_reload) as reload:
                selection = helper.freeze_checkpoints(root, run, snapshot, {})
            self.assertEqual(reload.call_count, 6)
            self.assertTrue(selection['h1_early_stopped'])
            self.assertEqual(selection['checkpoints']['h1']['training_last_epoch'], 329)
            self.assertEqual(selection['checkpoints']['h1']['epoch'], 109)
            self.assertFalse((run / 'h1/formal_result.json').exists())
            helper.check_hashes(originals)
            helper.check_hashes(selection['input_sha256'])
            for stage in selected:
                record = selection['checkpoints'][stage]
                self.assertEqual(helper.sha(record['best']), originals[record['source_best']])
                self.assertEqual(helper.sha(record['saved_last']), originals[record['source_last']])

    def test_gpu_free_memory_only_largest_wins(self):
        with patch.object(helper.subprocess, 'check_output', return_value='0, GPU-a, 9000\n1, GPU-b, 44000\n2, GPU-c, 22000\n') as query:
            self.assertEqual(helper.choose_gpu()['index'], 1)
            self.assertNotIn('utilization', ' '.join(query.call_args.args[0]))
        with patch.object(helper.subprocess, 'check_output', return_value='0, GPU-a, 8191\n'):
            with self.assertRaises(AssertionError):
                helper.choose_gpu()

    def test_package_python_syntax(self):
        for path in HERE.glob('*.py'):
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')


if __name__ == '__main__':
    unittest.main(verbosity=2)
