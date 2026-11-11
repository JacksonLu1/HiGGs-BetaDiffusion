"""Pipeline integration and free-memory scheduling; no GPU processes launched."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('qb_launch_test', HERE / 'launch.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class SchedulingTests(unittest.TestCase):
    def test_free_memory_ranking_no_utilization_or_process_gate(self):
        rows = [{'index': 0, 'free_mib': 12000, 'utilization': 0},
                {'index': 1, 'free_mib': 45000, 'utilization': 99},
                {'index': 2, 'free_mib': 42000, 'utilization': 0},
                {'index': 3, 'free_mib': 5000, 'utilization': 0}]
        self.assertEqual([r['index'] for r in driver.candidates(rows)], [1, 2, 0])
        self.assertEqual([r['index'] for r in driver.candidates(rows, reserved={1}, allowed=[0, 1])], [0])
        self.assertEqual(driver.candidates(rows, allowed=[3]), [])

    def test_four_dataset_names_not_old_collections(self):
        c = SimpleNamespace(SPECS=dict.fromkeys(('ego_small', 'proteins', 'msrc21', 'ego')))
        with patch.object(driver, 'common', return_value=c):
            self.assertEqual(driver.names_for('all'), tuple(c.SPECS))
            self.assertEqual(driver.names_for('proteins'), ('proteins',))


class PipelineTests(unittest.TestCase):
    def exercise(self, fail=False):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            lock_path = root / 'logs/.four_benchmarks_beta_ego_small.lock'
            lock_path.parent.mkdir()
            lock_path.touch()
            out = root / 'outputs/run'
            request = root / 'request.json'
            request.write_text(json.dumps({'launcher_sha256': 'hash', 'gpu': {'index': 1}}))
            c = SimpleNamespace(baseline=Mock(return_value={}), sha=Mock(return_value='hash'),
                get_item=Mock(return_value={'manifest_sha256': 'frozen', 'splits': {'test': 40}}),
                check_hashes=Mock())
            events = []
            def group(args, folder, mode, before):
                events.append(mode)
                if fail and mode == 'formal':
                    raise RuntimeError('synthetic formal failure')
                return folder / ('smoke' if mode == 'smoke' else 'train'), {}
            def child(command, log, pid_path, cwd, lock_fd, env=None):
                program = Path(command[3]).name
                events.append(program + ':' + log.name)
                if log.name == 'smoke_sample.log':
                    (out / 'smoke_sample').mkdir()
                    (out / 'smoke_sample/input_hash_check.json').write_text('{"unchanged": true}')
                if log.name == 'evaluation.log':
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            with lock_path.open('rb') as handle:
                args = SimpleNamespace(root=str(root), output=str(out), request=str(request),
                                       dataset='ego_small', lock_fd=handle.fileno())
                with patch.object(driver, 'common', return_value=c), \
                     patch.object(driver, 'train_group', side_effect=group), \
                     patch.object(driver, 'run_child', side_effect=child):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'synthetic formal'):
                            driver.worker(args)
                    else:
                        driver.worker(args)
            state = json.loads((out / 'pipeline_status.json').read_text())
            self.assertEqual(state['status'], 'failed' if fail else 'complete')
            self.assertEqual(events[:3], ['smoke', 'qb_beta_sample.py:smoke_sample.log', 'formal'])
            if fail:
                self.assertEqual(len(events), 3)
            else:
                self.assertEqual(events[3:], ['qb_beta_sample.py:sample.log', 'evaluate.py:evaluation.log'])
                c.check_hashes.assert_called()

    def test_smoke_train_sample_then_mixed_evaluate(self):
        self.exercise()

    def test_failed_training_does_not_sample_or_evaluate(self):
        self.exercise(fail=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
