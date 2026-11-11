"""Synthetic metadata and mocked signals only; never manage real processes/GPUs."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, call, patch


HERE = Path(__file__).resolve().parent
FAKE_FCNTL = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=Mock())
SPEC = importlib.util.spec_from_file_location('dd_parallel_test_subject', HERE / 'dd_glue_parallel.py')
APP = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {'fcntl': FAKE_FCNTL}):
    SPEC.loader.exec_module(APP)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.test_handoff_', dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'outputs' / APP.RUN
        self.out.mkdir(parents=True)
        (self.root / 'logs/training').mkdir(parents=True)
        self.lock_path = self.root / 'logs/training/dd_beta_formal.lock'
        self.lock_path.write_text('', encoding='utf-8')
        self.driver = self.root / 'tools/diagnostics/dd_beta_formal_20261027.py'
        self.driver.parent.mkdir(parents=True)
        self.driver.write_text('# synthetic driver; never executed\n', encoding='utf-8')
        self.parent = self.identity(1001, 'all', parent=1)
        self.children = {'h1': self.identity(1002, 'h1'), 'h2': self.identity(1003, 'h2')}
        self.request = {'root': str(self.root), 'out': str(self.out), 'gpu': 2,
                        'parent_pid': 1001, 'h1_pid': 1002, 'h2_pid': 1003,
                        'log': str(self.root / 'logs/training/parallel.log')}
        self.fake_fcntl = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=Mock())
        self.fake_signal = types.SimpleNamespace(SIGSTOP=19, SIGTERM=15, SIGCONT=18)

    def identity(self, pid, stage, parent=1001):
        target = self.out if stage == 'all' else self.out / stage
        return {'pid': pid, 'state': 'S', 'ppid': parent, 'uid': 500,
                'startticks': str(100000 + pid),
                'argv': ['python', '-u', '-B', str(self.driver), str(self.root), str(target), stage]}

    def reports(self):
        for stage in APP.MONITORS:
            folder = self.out / stage / 'checkpoints' / f'dd_{stage}_mean_hog_formal'
            folder.mkdir(parents=True, exist_ok=True)
            best, last = folder / 'epoch=9.ckpt', folder / 'last.ckpt'
            best.write_bytes(b'synthetic best: not a real checkpoint')
            last.write_bytes(b'synthetic last: not a real checkpoint')
            report = {'stage': stage, 'epoch': 499, 'global_step': 5000,
                      'monitor': APP.MONITORS[stage], 'strict_reload': True,
                      'strict_best_reload': True, 'manifest_sha256': APP.MANIFEST_SHA,
                      'best': str(best), 'last': str(last), 'best_sha256': APP.sha(best),
                      'last_sha256': APP.sha(last), 'best_epoch': 9, 'best_score': .7}
            if stage == 'h1':
                report.update(two_hop={'normalization': 'mean', 'warm_start': None},
                              hog={'rw_depth': 4, 'two_hop': True},
                              input_widths={'X': 12, 'E': 6, 'y': 14})
            (self.out / stage / 'formal_result.json').write_text(json.dumps(report), encoding='utf-8')

    def test_exact_parent_and_child_identity(self):
        values = {1001: self.parent, 1002: self.children['h1']}
        with patch.object(APP, 'proc', side_effect=lambda pid: values.get(pid)), \
             patch.object(APP.os, 'getuid', return_value=500, create=True):
            self.assertEqual(APP.expected_process(self.root, self.out, 1001, 'all'), self.parent)
            self.assertEqual(APP.expected_process(self.root, self.out, 1002, 'h1', 1001), self.children['h1'])

    def test_wrong_run_stage_parent_user_or_zombie_are_refused(self):
        mutations = (
            lambda v: v['argv'].__setitem__(-2, str(self.out / 'other')),
            lambda v: v['argv'].__setitem__(-1, 'glue'),
            lambda v: v.update(ppid=4000),
            lambda v: v.update(uid=501),
            lambda v: v.update(state='Z'),
        )
        for mutate in mutations:
            item = copy.deepcopy(self.children['h1'])
            mutate(item)
            with self.subTest(mutation=mutate), patch.object(APP, 'proc', return_value=item), \
                 patch.object(APP.os, 'getuid', return_value=500, create=True), self.assertRaises(AssertionError):
                APP.expected_process(self.root, self.out, 1002, 'h1', 1001)

    def test_same_process_rejects_reused_pid_or_changed_command(self):
        for key, value in (('startticks', 'different'), ('argv', ['other']), ('state', 'Z')):
            item = dict(self.parent, **{key: value})
            with patch.object(APP, 'proc', return_value=item):
                self.assertFalse(APP.same_process(self.parent))

    def test_existing_glue_artifacts_refused_before_proc_scan(self):
        for name in ('glue', 'glue.log', 'glue.pid'):
            path = self.out / name
            path.touch()
            try:
                with self.subTest(name=name), self.assertRaises(AssertionError):
                    APP.no_glue(self.root, self.out)
            finally:
                path.unlink()

    def test_preflight_exact_scope_and_all_required_guards(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(APP.sys, 'platform', 'linux'))
            stack.enter_context(patch.object(APP, 'ROOT', self.root))
            stack.enter_context(patch.object(APP.os, 'pidfd_open', Mock(), create=True))
            stack.enter_context(patch.object(APP.signal, 'pidfd_send_signal', Mock(), create=True))
            identity = stack.enter_context(patch.object(APP, 'expected_process', side_effect=[self.parent, self.children['h1'], self.children['h2']]))
            owner = stack.enter_context(patch.object(APP, 'flock_owner', return_value=True))
            no_glue = stack.enter_context(patch.object(APP, 'no_glue'))
            memory = stack.enter_context(patch.object(APP, 'check_memory'))
            baseline = stack.enter_context(patch.object(APP, 'baseline', return_value={'synthetic': 'hash'}))
            actual = APP.preflight(self.request)
        self.assertEqual(actual, (self.root, self.out, self.parent, self.children, {'synthetic': 'hash'}))
        self.assertEqual(identity.call_count, 3)
        owner.assert_called_once_with(self.lock_path, 1001)
        no_glue.assert_called_once_with(self.root, self.out)
        memory.assert_called_once_with(2)
        baseline.assert_called_once_with(self.root)

    def test_wrong_root_refused_before_process_or_signal_calls(self):
        request = dict(self.request, root=str(self.root / 'wrong'))
        with patch.object(APP.sys, 'platform', 'linux'), patch.object(APP, 'ROOT', self.root), \
             patch.object(APP, 'expected_process') as identity, patch.object(APP, 'signal_parent') as sender, \
             self.assertRaises(AssertionError):
            APP.preflight(request)
        identity.assert_not_called()
        sender.assert_not_called()

    def transfer_context(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(APP.os, 'pidfd_open', return_value=91, create=True))
        stack.enter_context(patch.object(APP.os, 'close'))
        stack.enter_context(patch.object(APP, 'same_process', return_value=True))
        stack.enter_context(patch.object(APP, 'wait_stopped'))
        stack.enter_context(patch.object(APP, 'no_glue'))
        stack.enter_context(patch.object(APP, 'flock_owner', return_value=True))
        stack.enter_context(patch.object(APP.select, 'select', return_value=([91], [], [])))
        stack.enter_context(patch.object(APP, 'fcntl', self.fake_fcntl))
        stack.enter_context(patch.object(APP, 'signal', self.fake_signal))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_successful_transfer_signals_only_pinned_parent(self):
        with self.transfer_context(), patch.object(APP, 'signal_parent') as sender:
            lock = APP.transfer(self.root, self.out, self.parent, self.children)
            self.addCleanup(lock.close)
        self.assertEqual(sender.call_args_list, [call(91, 19), call(91, 15), call(91, 18)])
        self.fake_fcntl.flock.assert_called_once()

    def test_glue_race_after_freeze_resumes_parent_without_terminating(self):
        with self.transfer_context(), patch.object(APP, 'signal_parent') as sender, \
             patch.object(APP, 'no_glue', side_effect=AssertionError('Glue appeared')), \
             self.assertRaises(AssertionError):
            APP.transfer(self.root, self.out, self.parent, self.children)
        self.assertEqual(sender.call_args_list, [call(91, 19), call(91, 18)])
        self.fake_fcntl.flock.assert_not_called()

    def test_child_exit_after_freeze_resumes_parent(self):
        with self.transfer_context(), patch.object(APP, 'signal_parent') as sender, \
             patch.object(APP, 'same_process', side_effect=[True, False]), self.assertRaises(FileNotFoundError):
            APP.transfer(self.root, self.out, self.parent, self.children)
        self.assertEqual(sender.call_args_list, [call(91, 19), call(91, 18)])

    def test_just_completed_child_is_accepted_if_report_is_valid(self):
        self.reports()
        with self.transfer_context(), patch.object(APP, 'signal_parent') as sender, \
             patch.object(APP, 'same_process', side_effect=[True, False, True]):
            lock = APP.transfer(self.root, self.out, self.parent, self.children)
            self.addCleanup(lock.close)
        self.assertEqual(sender.call_args_list, [call(91, 19), call(91, 15), call(91, 18)])

    def test_parent_identity_change_sends_no_signals(self):
        with self.transfer_context(), patch.object(APP, 'signal_parent') as sender, \
             patch.object(APP, 'same_process', return_value=False), self.assertRaises(AssertionError):
            APP.transfer(self.root, self.out, self.parent, self.children)
        sender.assert_not_called()

    def test_signal_adapter_uses_pidfd_not_numeric_kill(self):
        with patch.object(APP.signal, 'pidfd_send_signal', Mock(), create=True) as pidfd, \
             patch.object(APP.os, 'kill') as numeric:
            APP.signal_parent(91, signal.SIGTERM)
        pidfd.assert_called_once_with(91, signal.SIGTERM, None, 0)
        numeric.assert_not_called()

    def test_memory_guard_is_memory_only(self):
        with patch.object(APP.subprocess, 'check_output', return_value='9000\n') as query, \
             contextlib.redirect_stdout(io.StringIO()):
            APP.check_memory(2)
        self.assertIn('--query-gpu=memory.free', query.call_args.args[0])
        self.assertNotIn('utilization', str(query.call_args))
        with patch.object(APP.subprocess, 'check_output', return_value='8191\n'), self.assertRaises(AssertionError):
            APP.check_memory(2)

    def test_valid_reports_cover_all_three_stages(self):
        self.reports()
        for stage in APP.MONITORS:
            self.assertEqual(APP.validate_report(self.out, stage)['epoch'], 499)

    def test_report_rejects_bad_manifest_epoch_or_strict_reload(self):
        self.reports()
        path = self.out / 'h1/formal_result.json'
        good = json.loads(path.read_text())
        for key, value in (('epoch', 1), ('manifest_sha256', 'wrong'), ('strict_reload', False), ('strict_best_reload', False)):
            path.write_text(json.dumps(dict(good, **{key: value})), encoding='utf-8')
            with self.subTest(key=key), self.assertRaises(AssertionError):
                APP.validate_report(self.out, 'h1')

    def test_report_rejects_checkpoint_hash_changes(self):
        self.reports()
        path = self.out / 'h2/formal_result.json'
        report = json.loads(path.read_text())
        Path(report['best']).write_bytes(b'changed')
        with self.assertRaises(AssertionError):
            APP.validate_report(self.out, 'h2')

    def test_baseline_requires_original_smoke_and_driver_and_manifest_hashes(self):
        source = self.root / 'synthetic-source.txt'
        source.write_text('unchanged', encoding='utf-8')
        smoke = self.root / 'outputs' / APP.SMOKE
        smoke.mkdir()
        (smoke / 'input_hash_check.json').write_text(json.dumps({'unchanged': True, 'sha256': {str(source): APP.sha(source)}}), encoding='utf-8')
        manifest = self.root / 'data_sources/splits/dd_hierarchy_r2_s1234_l0.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{}', encoding='utf-8')
        with patch.object(APP, 'DRIVER_SHA', APP.sha(self.driver)), patch.object(APP, 'MANIFEST_SHA', APP.sha(manifest)):
            actual = APP.baseline(self.root)
            self.assertIn(str(self.driver), actual)
            source.write_text('changed', encoding='utf-8')
            with self.assertRaises(AssertionError):
                APP.baseline(self.root)

    def run_fake_worker(self, *, missing=None, changed_source=False,
                        fail_intent=False, fail_spawn=False,
                        fail_metadata=False, fail_pid_write=False):
        self.reports()
        if missing:
            (self.out / missing / 'formal_result.json').unlink()
        source = self.root / 'synthetic-source.txt'
        source.write_text('unchanged', encoding='utf-8')
        before = {str(source): APP.sha(source)}
        if changed_source:
            source.write_text('changed', encoding='utf-8')
        request_path = self.out / 'request.json'
        request_path.write_text(json.dumps(self.request), encoding='utf-8')
        (self.root / 'logs/dd_latest_train.json').write_text(json.dumps({'pid': 1001}), encoding='utf-8')
        glue = Mock(pid=2001, returncode=0)
        glue.poll.return_value = 0
        scheduler_lock = Mock()
        self.worker_events = []
        self.worker_transfer = Mock(return_value=scheduler_lock)
        self.worker_drain = Mock()
        self.worker_lock = scheduler_lock
        scheduler_lock.close.side_effect = lambda: self.worker_events.append('lock_closed')
        real_atomic = APP.atomic_json
        real_write_text = Path.write_text

        def atomic(path, value):
            path = Path(path)
            if path == request_path:
                self.worker_events.append('intent')
                if fail_intent:
                    raise OSError('synthetic intent failure')
            if fail_metadata and path.name in ('glue_parallel_handoff.json', 'dd_latest_train.json') and not value.get('completed'):
                self.worker_events.append('nonfatal_metadata_failure')
                raise OSError('synthetic nonfatal metadata failure')
            return real_atomic(path, value)

        def write_text(path, *args, **kwargs):
            if fail_pid_write and Path(path) == self.out / 'glue.pid':
                self.worker_events.append('pid_write_failed')
                raise OSError('synthetic PID write failure')
            return real_write_text(path, *args, **kwargs)

        def drain(children, existing_glue):
            self.assertEqual(scheduler_lock.close.call_count, 0, 'Lock must remain held while draining')
            self.worker_events.append('drain_while_locked')
            self.worker_drain(children, existing_glue)

        def spawn(*args, **kwargs):
            self.worker_events.append('spawn')
            if fail_spawn:
                raise OSError('synthetic spawn failure')
            return glue

        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(APP, 'fcntl', self.fake_fcntl))
        stack.enter_context(patch.object(APP, 'preflight', return_value=(self.root, self.out, self.parent, self.children, before)))
        stack.enter_context(patch.object(APP, 'transfer', self.worker_transfer))
        stack.enter_context(patch.object(APP, 'same_process', side_effect=[True, True, False, False]))
        stack.enter_context(patch.object(APP, 'no_glue'))
        stack.enter_context(patch.object(APP, 'check_memory'))
        stack.enter_context(patch.object(APP, 'atomic_json', side_effect=atomic))
        stack.enter_context(patch.object(Path, 'write_text', new=write_text))
        stack.enter_context(patch.object(APP, 'drain_existing', side_effect=drain))
        stack.enter_context(patch.object(APP.time, 'sleep'))
        popen = stack.enter_context(patch.object(APP.subprocess, 'Popen', side_effect=spawn))
        sender = stack.enter_context(patch.object(APP, 'signal_parent'))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        with stack:
            APP.worker(request_path)
        sender.assert_not_called()
        self.assertEqual(popen.call_args.args[0][-1], 'glue')
        self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
        scheduler_lock.close.assert_called_once()

    def test_worker_writes_completion_only_after_three_valid_reports(self):
        self.run_fake_worker()
        audit = json.loads((self.out / 'input_hash_check.json').read_text())
        self.assertTrue(audit['unchanged'])
        self.assertTrue(json.loads((self.out / 'glue_parallel_handoff.json').read_text())['completed'])

    def test_missing_report_never_marks_complete(self):
        with self.assertRaises(RuntimeError):
            self.run_fake_worker(missing='h2')
        self.assertFalse((self.out / 'input_hash_check.json').exists())
        self.assertNotIn('completed', json.loads((self.out / 'glue_parallel_handoff.json').read_text()))

    def test_changed_source_never_marks_complete(self):
        with self.assertRaises(AssertionError):
            self.run_fake_worker(changed_source=True)
        self.assertFalse((self.out / 'input_hash_check.json').exists())

    def test_intent_write_failure_never_transfers_or_spawns(self):
        with self.assertRaises(OSError):
            self.run_fake_worker(fail_intent=True)
        self.worker_transfer.assert_not_called()
        self.worker_drain.assert_not_called()
        self.assertNotIn('spawn', self.worker_events)
        self.assertNotIn('lock_closed', self.worker_events)

    def test_nonfatal_metadata_failure_after_spawn_keeps_monitoring(self):
        self.run_fake_worker(fail_metadata=True)
        self.assertEqual(self.worker_events.count('spawn'), 1)
        self.assertGreaterEqual(self.worker_events.count('nonfatal_metadata_failure'), 3)
        self.worker_drain.assert_not_called()
        self.assertTrue(json.loads((self.out / 'input_hash_check.json').read_text())['unchanged'])
        self.assertTrue(json.loads((self.out / 'glue_parallel_handoff.json').read_text())['completed'])

    def test_pid_write_failure_after_spawn_does_not_abort(self):
        self.run_fake_worker(fail_pid_write=True)
        self.assertIn('pid_write_failed', self.worker_events)
        self.worker_drain.assert_not_called()
        self.assertTrue((self.out / 'input_hash_check.json').exists())

    def test_spawn_failure_drains_existing_training_before_lock_close(self):
        with self.assertRaises(OSError):
            self.run_fake_worker(fail_spawn=True)
        self.assertLess(self.worker_events.index('drain_while_locked'), self.worker_events.index('lock_closed'))
        self.worker_drain.assert_called_once_with(self.children, None)
        self.assertFalse((self.out / 'input_hash_check.json').exists())

    def test_failed_report_drains_already_spawned_glue_before_unlock(self):
        with self.assertRaises(RuntimeError):
            self.run_fake_worker(missing='h2')
        self.assertLess(self.worker_events.index('drain_while_locked'), self.worker_events.index('lock_closed'))
        self.assertEqual(self.worker_drain.call_count, 1)
        self.assertEqual(self.worker_drain.call_args.args[1].pid, 2001)

    def test_drain_waits_for_known_children_and_glue_without_signals(self):
        glue = Mock(pid=2001)
        glue.poll.side_effect = [None, 0]
        with patch.object(APP, 'same_process', side_effect=[True, False, False, False, False]), \
             patch.object(APP.time, 'sleep') as sleep, patch.object(APP, 'signal_parent') as sender, \
             contextlib.redirect_stdout(io.StringIO()):
            APP.drain_existing(self.children, glue)
        self.assertEqual(sleep.call_args_list, [call(5), call(5)])
        sender.assert_not_called()

    def test_record_or_warn_converts_only_metadata_errors_to_warning(self):
        with patch.object(APP, 'atomic_json', side_effect=OSError('synthetic disk error')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(APP.record_or_warn(self.out / 'meta.json', {'phase': 'test'}))
        self.assertIn('continues holding its lock', output.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
