"""CPU numerical guard tests; no Ego/server/formal sampling claim."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stable_eigen import EigenGuardError, GuardedEigh, make_guarded_eigenfeatures


NATIVE = torch.linalg.eigh


def converge_failure(matrix, UPLO='L'):
    kind = getattr(torch._C, '_LinAlgError', RuntimeError)
    raise kind('linalg.eigh: The algorithm failed to converge because the input matrix is ill-conditioned')


class TinyEigenFeatures:
    mode = 'all'
    def __call__(self, noisy_data):
        eigvals, eigvectors = torch.linalg.eigh(noisy_data['L'])
        # Both this unchanged body and other torch APIs must still be used.
        return eigvals / torch.ones_like(eigvals), eigvectors * 1


class GuardTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.matrix = torch.tensor([[[2., -1., -1., 0.], [-1., 2., 0., -1.],
                                     [-1., 0., 2., -1.], [0., -1., -1., 2.]]])

    def test_normal_native_return_identity_bits_rng_and_no_global_patch(self):
        result = NATIVE(self.matrix)
        primary = Mock(return_value=result)
        callback = Mock()
        guard = GuardedEigh(callback, primary_solver=primary)
        before = torch.get_rng_state().clone()
        self.assertIs(guard(self.matrix), result)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        primary.assert_called_once()
        callback.assert_not_called()
        self.assertIs(torch.linalg.eigh, NATIVE)
        self.assertEqual(guard.fallbacks, 0)

    def test_forced_linalg_failure_same_matrix_cpu_float64_and_audit(self):
        before = self.matrix.clone()
        before_rng = torch.get_rng_state().clone()
        received, records = [], []
        def cpu(matrix, UPLO='L'):
            self.assertEqual(matrix.device.type, 'cpu')
            self.assertEqual(matrix.dtype, torch.float64)
            received.append(matrix.clone())
            return NATIVE(matrix, UPLO=UPLO)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            guard = GuardedEigh(records.append, directory,
                                primary_solver=converge_failure, cpu_solver=cpu)
            values, vectors = guard(self.matrix)
            self.assertEqual(values.dtype, self.matrix.dtype)
            self.assertEqual(vectors.dtype, self.matrix.dtype)
            self.assertEqual(values.device, self.matrix.device)
            self.assertTrue(torch.equal(received[0], self.matrix.double()))
            self.assertTrue(torch.equal(self.matrix, before))
            self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record['event'], 'eigh_cpu_float64_fallback')
            self.assertFalse(record['matrix_changed'])
            self.assertFalse(record['jitter_added'])
            self.assertFalse(record['eigenbasis_bit_identity_claimed'])
            self.assertLess(record['cpu_float64_quality']['residual_relative_frobenius'], 1e-10)
            self.assertLess(record['cpu_float64_quality']['orthogonality_normalized_frobenius'], 1e-10)
            self.assertTrue(Path(record['matrix_file']).is_file())
            self.assertEqual(len(record['matrix_file_sha256']), 64)
            saved = torch.load(record['matrix_file'], map_location='cpu', weights_only=False)
            self.assertTrue(torch.equal(saved['L'], self.matrix))
            json.dumps(record, allow_nan=False)

    def test_double_precision_and_unbatched_inputs_supported(self):
        for matrix in (self.matrix.double(), self.matrix[0], self.matrix[0].double()):
            values, vectors = GuardedEigh(primary_solver=converge_failure)(matrix)
            self.assertEqual(values.shape, matrix.shape[:-1])
            self.assertEqual(vectors.shape, matrix.shape)
            self.assertEqual(values.dtype, matrix.dtype)

    def test_nonfinite_inputs_rejected_dumped_not_sanitized(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            matrix = self.matrix.clone()
            matrix[0, 0, 0] = bad
            primary, cpu, records = Mock(), Mock(), []
            with tempfile.TemporaryDirectory(dir=HERE) as directory:
                guard = GuardedEigh(records.append, directory, primary, cpu)
                with self.assertRaisesRegex(EigenGuardError, 'Nonfinite'):
                    guard(matrix)
                primary.assert_not_called()
                cpu.assert_not_called()
                self.assertFalse(records[0]['finite'])
                self.assertTrue(Path(records[0]['matrix_file']).is_file())
                json.dumps(records[0], allow_nan=False)
            self.assertTrue(torch.isnan(matrix[0, 0, 0]) if bad != bad else matrix[0, 0, 0] == bad)

    def test_even_small_asymmetry_refused_without_modification(self):
        matrix = self.matrix.clone()
        matrix[0, 0, 1] = -0.999999
        before = matrix.clone()
        primary, cpu, records = Mock(), Mock(), []
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            with self.assertRaisesRegex(EigenGuardError, 'Asymmetric'):
                GuardedEigh(records.append, directory, primary, cpu)(matrix)
            self.assertTrue(torch.equal(before, matrix))
            self.assertGreater(records[0]['symmetry_max_abs'], 0)
            primary.assert_not_called()
            cpu.assert_not_called()

    def test_generic_runtime_error_propagates_without_retry(self):
        error = RuntimeError('CUDA out of memory')
        primary, cpu, callback = Mock(side_effect=error), Mock(), Mock()
        with self.assertRaisesRegex(RuntimeError, 'out of memory') as raised:
            GuardedEigh(callback, primary_solver=primary, cpu_solver=cpu)(self.matrix)
        self.assertIs(raised.exception, error)
        cpu.assert_not_called()
        callback.assert_not_called()

    def test_nonfinite_native_outputs_refused_without_cpu_retry(self):
        cpu, records = Mock(), []
        native = Mock(return_value=(torch.full((1, 4), float('nan')), torch.eye(4).unsqueeze(0)))
        with self.assertRaisesRegex(EigenGuardError, 'Native eigensolver returned nonfinite'):
            GuardedEigh(records.append, primary_solver=native, cpu_solver=cpu)(self.matrix)
        cpu.assert_not_called()
        self.assertEqual(records[-1]['event'], 'eigh_guard_rejected')

    def test_overflowed_quality_values_cannot_pass_nan_comparisons(self):
        matrix = torch.full((1, 2, 2), 1e308, dtype=torch.float64)
        values = torch.full((1, 2), 1e308, dtype=torch.float64)
        vectors = torch.ones((1, 2, 2), dtype=torch.float64)
        with self.assertRaisesRegex(EigenGuardError, 'Nonfinite eigensolver quality audit'):
            GuardedEigh._quality(matrix, values, vectors)

    def test_unrelated_runtime_with_convergence_word_not_retried(self):
        cpu = Mock()
        with self.assertRaisesRegex(RuntimeError, 'custom trainer'):
            GuardedEigh(primary_solver=Mock(side_effect=RuntimeError('custom trainer failed to converge')),
                        cpu_solver=cpu)(self.matrix)
        cpu.assert_not_called()

    def test_inaccurate_or_nonfinite_cpu_eigenpairs_stop_with_evidence(self):
        n = self.matrix.shape[-1]
        for result in ((torch.zeros(1, n, dtype=torch.float64), torch.eye(n, dtype=torch.float64).unsqueeze(0)),
                       (torch.full((1, n), float('nan'), dtype=torch.float64),
                        torch.eye(n, dtype=torch.float64).unsqueeze(0)),
                       (torch.zeros(1, n, dtype=torch.float64), torch.zeros(1, n, n, dtype=torch.float64))):
            records = []
            with tempfile.TemporaryDirectory(dir=HERE) as directory:
                guard = GuardedEigh(records.append, directory, converge_failure, Mock(return_value=result))
                with self.assertRaises(EigenGuardError):
                    guard(self.matrix)
                self.assertEqual(records[-1]['event'], 'eigh_guard_rejected')
                self.assertTrue(Path(records[-1]['matrix_file']).is_file())

    def test_cpu_solver_failure_propagates_as_guard_refusal(self):
        records = []
        with self.assertRaisesRegex(EigenGuardError, 'CPU float64'):
            GuardedEigh(records.append, primary_solver=converge_failure,
                        cpu_solver=Mock(side_effect=RuntimeError('LAPACK error')))(self.matrix)
        self.assertEqual(records[-1]['event'], 'eigh_guard_rejected')

    def test_inference_only_fallback(self):
        cpu = Mock()
        matrix = self.matrix.clone().requires_grad_(True)
        with self.assertRaisesRegex(EigenGuardError, 'inference-only'):
            GuardedEigh(primary_solver=converge_failure, cpu_solver=cpu)(matrix)
        cpu.assert_not_called()

    def test_repeated_spectrum_and_disconnected_zero_matrix(self):
        for matrix in (torch.zeros(1, 10, 10), torch.eye(8).unsqueeze(0)):
            values, vectors = GuardedEigh(primary_solver=converge_failure)(matrix)
            self.assertTrue(torch.isfinite(values).all())
            self.assertTrue(torch.isfinite(vectors).all())


class FeatureWrapperTests(unittest.TestCase):
    def test_private_function_binding_preserves_original_and_result_bits(self):
        existing = TinyEigenFeatures()
        matrix = torch.tensor([[[1., -1.], [-1., 1.]]])
        original = existing({'L': matrix})
        wrapper = make_guarded_eigenfeatures(existing, Mock())
        actual = wrapper({'L': matrix})
        self.assertEqual(wrapper.mode, 'all')
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original, actual)))
        self.assertIs(torch.linalg.eigh, NATIVE)
        self.assertIs(TinyEigenFeatures.__call__.__globals__['torch'], torch)
        self.assertIs(wrapper.original, existing)

    def test_private_solver_failure_does_not_change_other_feature_instances(self):
        matrix = torch.tensor([[[1., -1.], [-1., 1.]]])
        existing, other = TinyEigenFeatures(), TinyEigenFeatures()
        records = []
        wrapper = make_guarded_eigenfeatures(existing, records.append)
        wrapper.solver.primary_solver = converge_failure
        before = torch.get_rng_state().clone()
        actual = wrapper({'L': matrix})
        native = other({'L': matrix})
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(all(torch.isfinite(v).all() for v in actual))
        self.assertTrue(all(torch.isfinite(v).all() for v in native))
        self.assertEqual(wrapper.solver.fallbacks, 1)
        self.assertIs(torch.linalg.eigh, NATIVE)

    def test_actual_existing_eigenfeatures_function_unchanged(self):
        checkout = HERE.parents[1] / 'HiGGs-main'
        sys.path.insert(0, str(checkout / 'dgd'))
        from diffusion.extra_features import EigenFeatures
        existing = EigenFeatures('all')
        mask = torch.tensor([[True, True, True, True, False]])
        adjacency = torch.zeros(1, 5, 5, dtype=torch.long)
        adjacency[0, 0, 1] = adjacency[0, 1, 0] = 1
        adjacency[0, 1, 2] = adjacency[0, 2, 1] = 1
        data = {'E_t': torch.nn.functional.one_hot(adjacency, 2).float(), 'node_mask': mask}
        before = torch.get_rng_state().clone()
        expected = existing(data)
        records = []
        wrapper = make_guarded_eigenfeatures(existing, records.append)
        actual = wrapper(data)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))
        self.assertEqual(records, [])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertIs(torch.linalg.eigh, NATIVE)
        wrapper.solver.primary_solver = converge_failure
        fallback = wrapper(data)
        self.assertEqual([v.shape for v in fallback], [v.shape for v in expected])
        self.assertTrue(all(torch.isfinite(v).all() for v in fallback))
        self.assertEqual(len(records), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
