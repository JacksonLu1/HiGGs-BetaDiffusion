"""Mock-only process safety checks. NO real PID, signal, GPU or server access."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('early_stop_helper_test', HERE / 'run_early.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class StopSafety(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=HERE)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / 'outputs' / helper.RUN
        self.run.mkdir(parents=True)
        (self.run / 'h1.pid').write_text('12345')
        self.snapshot = self.root / 'outputs/dd_earlystop_snapshot_test'
        self.snapshot.mkdir()
        driver = str(self.root / 'tools/diagnostics/dd_beta_formal_20261027.py')
        self.identity = {'pid': 12345, 'uid': 77, 'state': 'R', 'startticks': '12345678',
                         'argv': ['python', '-u', '-B', driver, str(self.root), str(self.run / 'h1'), 'h1']}
        self.events = []
        self.selection = {'checkpoints': {'h1': {'training_last_epoch': 329}}}
        self.report_path = self.run / 'h1/formal_result.json'
        self.report_path.parent.mkdir()
        self.report_path.write_text('{}')
        self.patchers = [
            patch.object(helper, 'import_project'), patch.object(helper, 'baseline', return_value={}),
            patch.object(helper, 'complete_report', return_value=(self.report_path, {})),
            patch.object(helper, 'choose_gpu'), patch.object(helper, 'wait_stopped'),
            patch.object(helper, 'validate_sampler_ready'),
            patch.object(helper, 'proc', side_effect=lambda _: dict(self.identity)),
            patch.object(helper.os, 'getuid', return_value=77, create=True),
            patch.object(helper.os, 'close'), patch.object(helper.signal, 'signal', return_value=None),
            patch.object(helper.signal, 'SIGSTOP', 19, create=True),
            patch.object(helper.signal, 'SIGCONT', 18, create=True),
            patch.object(helper.select, 'select', return_value=([100], [], [])),
            patch.object(helper, 'pidfd_api', return_value=(lambda *args: 100, self.send)),
            patch.object(helper, 'freeze_checkpoints', side_effect=self.freeze),
        ]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)

    def send(self, fd, value, info=None, flags=0):
        self.assertEqual(fd, 100)
        self.events.append(value)

    def freeze(self, *args):
        self.events.append('freeze_validated')
        helper.write_json(self.snapshot / 'selection.json', self.selection)
        return self.selection

    def audit(self):
        return json.loads((self.snapshot / 'stop_audit.json').read_text())

    def test_valid_snapshot_precedes_term_and_exact_pidfd_only(self):
        helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.events, [19, 'freeze_validated', 15, 18])
        self.assertEqual(self.audit()['status'], 'h1_stopped')
        self.assertTrue(self.audit()['snapshot_verified'])
        self.assertTrue(self.audit()['h1_exit_confirmed'])
        self.assertFalse(self.audit()['other_processes_signaled'])

    def test_bad_frozen_checkpoint_resumes_without_termination(self):
        helper.freeze_checkpoints.side_effect = RuntimeError('invalid copied checkpoint')
        with self.assertRaisesRegex(RuntimeError, 'invalid copied'):
            helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.events, [19, 18])
        self.assertTrue(self.audit()['h1_resumed_before_termination'])
        self.assertEqual(self.audit()['status'], 'failed_before_termination')

    def test_interruption_immediately_after_stop_still_resumes(self):
        def interrupted(fd, value, *args):
            self.events.append(value)
            if value == 19:
                raise KeyboardInterrupt()
        helper.pidfd_api.return_value = (lambda *args: 100, interrupted)
        with self.assertRaises(KeyboardInterrupt):
            helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.events, [19, 18])
        self.assertTrue(self.audit()['h1_resumed_before_termination'])

    def test_preexisting_pause_never_resumed_or_terminated(self):
        self.identity['state'] = 'T'
        with self.assertRaisesRegex(AssertionError, 'already paused'):
            helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.events, [])

    def test_wrong_stage_never_signaled(self):
        self.identity['argv'][-1] = 'h2'
        with self.assertRaisesRegex(AssertionError, 'Wrong run or stage'):
            helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.events, [])

    def test_reused_pid_refused_before_stop(self):
        identities = [dict(self.identity), dict(self.identity, startticks='different')]
        helper.proc.side_effect = lambda _: identities.pop(0)
        with self.assertRaisesRegex(AssertionError, 'PID changed'):
            helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.events, [])

    def test_exit_timeout_is_uncertain_not_success(self):
        helper.select.select.return_value = ([], [], [])
        with self.assertRaisesRegex(RuntimeError, 'exit not confirmed'):
            helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.audit()['status'], 'stop_uncertain')
        self.assertNotIn('h1_exit_confirmed', self.audit())
        self.assertNotIn('h1_resumed_before_termination', self.audit())

    def test_finalepoch_race_resumes_instead_of_mislabeling(self):
        self.selection['checkpoints']['h1']['training_last_epoch'] = 499
        with self.assertRaisesRegex(AssertionError, 'reached final epoch'):
            helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.events, [19, 'freeze_validated', 18])

    def test_no_gpu_memory_failure_precedes_any_signal(self):
        helper.choose_gpu.side_effect = RuntimeError('No free VRAM')
        with self.assertRaisesRegex(RuntimeError, 'No free VRAM'):
            helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.events, [])

    def test_sampler_source_preflight_failure_resumes_without_termination(self):
        helper.validate_sampler_ready.side_effect = AssertionError('sampling source changed')
        with self.assertRaisesRegex(AssertionError, 'sampling source changed'):
            helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.events, [19, 'freeze_validated', 18])
        self.assertEqual(self.audit()['status'], 'failed_before_termination')

    def test_already_completed_has_proof_and_no_signal(self):
        helper.proc.side_effect = lambda _: None
        helper.freeze_and_stop(self.root, self.run, self.snapshot)
        self.assertEqual(self.events, ['freeze_validated'])
        self.assertEqual(self.audit()['status'], 'already_completed')
        self.assertTrue(self.audit()['snapshot_verified'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
