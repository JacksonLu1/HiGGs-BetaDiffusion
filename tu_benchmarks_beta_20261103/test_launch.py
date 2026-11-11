"""Pipeline integration and free-memory scheduling; no GPU processes launched."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('qb_launch_test', HERE / 'launch.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class SchedulingTests(unittest.TestCase):
    def test_cli_defaults_three_concurrent_and_memory_only(self):
        with patch.object(sys, 'argv', ['launch.py', 'pipeline']), \
             patch.object(driver, 'launch') as submit:
            driver.main()
        args = submit.call_args.args[0]
        self.assertEqual(args.max_parallel, 3)
        self.assertEqual(args.min_free_gib, 32)
        self.assertIsNone(args.gpus)

    def test_active_dataset_lock_refuses_before_gpu_or_process_launch(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            root = Path(folder)
            c = SimpleNamespace(baseline=Mock(), SPECS={'collab': {}})
            fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2,
                flock=Mock(side_effect=BlockingIOError('already running')))
            args = SimpleNamespace(root=str(root), dataset='collab', max_parallel=3, min_free_gib=32)
            with patch.object(driver, 'ROOT', root), \
                 patch.object(driver.sys, 'platform', 'linux'), \
                 patch.dict(sys.modules, {'fcntl': fake_fcntl}), \
                 patch.object(driver, 'common', return_value=c), \
                 patch.object(driver, 'gpu_inventory') as inventory, \
                 patch.object(driver.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(RuntimeError, 'already queued/running'):
                    driver.launch(args)
            inventory.assert_not_called()
            spawn.assert_not_called()

    def test_free_memory_ranking_no_utilization_or_process_gate(self):
        rows = [{'index': 0, 'free_mib': 12000, 'utilization': 0},
                {'index': 1, 'free_mib': 45000, 'utilization': 99},
                {'index': 2, 'free_mib': 42000, 'utilization': 0},
                {'index': 3, 'free_mib': 5000, 'utilization': 0}]
        self.assertEqual([r['index'] for r in driver.candidates(rows)], [1, 2, 0])
        self.assertEqual([r['index'] for r in driver.candidates(rows, reserved={1}, allowed=[0, 1])], [0])
        self.assertEqual(driver.candidates(rows, allowed=[3]), [])

    def test_three_dataset_names_not_old_collections(self):
        c = SimpleNamespace(SPECS=dict.fromkeys(('imdb_binary', 'collab', 'nci1')))
        with patch.object(driver, 'common', return_value=c):
            self.assertEqual(driver.names_for('all'), tuple(c.SPECS))
            self.assertEqual(driver.names_for('collab'), ('collab',))


class PipelineTests(unittest.TestCase):
    def test_collab_reports_require_matched_per_stage_batches(self):
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            output = Path(folder)
            stages = {'h2': ('2', 'beta_h2', 'val/beta_h2_loss'),
                      'h1': ('1', 'beta_h1', 'val/h1_total'),
                      'glue': ('1.5', 'original_glue', 'val/epoch_NLL')}
            batches = {'h2': 32, 'h1': 16, 'glue': 8}
            c = SimpleNamespace(get_item=Mock(return_value={'manifest_sha256': 'frozen'}),
                SPECS={'collab': {'epochs': 500, 'batch_size': batches}}, STAGES=stages,
                sha=Mock(return_value='unchanged'))
            for stage in stages:
                where = output / stage
                where.mkdir()
                report = {'dataset': 'collab', 'stage': stage, 'epoch': 499,
                    'manifest_sha256': 'frozen', 'strict_reload': True, 'strict_best_reload': True,
                    'inputs_unchanged': True, 'train_seed': 0, 'batch_size': batches[stage],
                    'monitor': stages[stage][2], 'diffusion_steps': 400 if stage == 'glue' else 100,
                    'best': str(where / 'best.ckpt'), 'last': str(where / 'last.ckpt'),
                    'best_sha256': 'unchanged', 'last_sha256': 'unchanged'}
                if stage == 'h1':
                    report.update(two_hop={'normalization': 'mean', 'warm_start': None, 'channels': 16},
                        hog={'rw_depth': 4, 'two_hop': True}, input_widths={'X': 12, 'E': 6, 'y': 14})
                (where / 'formal_result.json').write_text(json.dumps(report))
            with patch.object(driver, 'common', return_value=c):
                self.assertEqual(set(driver.read_reports(output, 'collab', output, 'formal')), set(stages))
                changed = output / 'glue/formal_result.json'
                report = json.loads(changed.read_text())
                report['batch_size'] = 32
                changed.write_text(json.dumps(report))
                with self.assertRaises(AssertionError):
                    driver.read_reports(output, 'collab', output, 'formal')

    def exercise(self, fail=False):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            lock_path = root / 'logs/.tu_benchmarks_beta_imdb_binary.lock'
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
                                       dataset='imdb_binary', lock_fd=handle.fileno())
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
            self.assertEqual(events[:3], ['smoke', 'tu_beta_sample.py:smoke_sample.log', 'formal'])
            if fail:
                self.assertEqual(len(events), 3)
            else:
                self.assertEqual(events[3:], ['tu_beta_sample.py:sample.log', 'evaluate.py:evaluation.log'])
                c.check_hashes.assert_called()

    def test_smoke_train_sample_then_mixed_evaluate(self):
        self.exercise()

    def test_failed_training_does_not_sample_or_evaluate(self):
        self.exercise(fail=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)

