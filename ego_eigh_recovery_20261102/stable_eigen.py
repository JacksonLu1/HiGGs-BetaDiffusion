"""Process-local EigenFeatures solver guard for the Ego Glue recovery.

The existing feature function and all feature postprocessing are preserved.
Only its direct ``torch.linalg.eigh`` lookup is rebound in a PRIVATE function
globals dictionary. No torch/module/class/source/checkpoint is monkeypatched.

Successful native solves are returned unchanged. A recognized nonconvergence
on a finite, exactly symmetric matrix can use CPU float64 on that SAME matrix.
No jitter, graph repair, feature removal, eigenvalue clipping, or random draw
is introduced. A fallback eigenbasis need not be bit-identical to a CUDA basis,
especially for repeated eigenvalues; the runtime audit records every fallback.
"""
import ast
import hashlib
import inspect
import math
import os
from pathlib import Path
import textwrap
import time
import types

import torch


POLICY = 'ego_glue_eigh_native_then_same_matrix_cpu_float64_v1'


class EigenGuardError(RuntimeError):
    """An invalid input or unacceptable fallback must stop this sample."""


def _nonconvergence(error):
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    failed = any(s in message for s in ('failed to converge', 'did not converge',
                                      'failure to converge'))
    kinds = tuple(t for t in (getattr(torch._C, '_LinAlgError', None),
                              getattr(torch.linalg, 'LinAlgError', None))
                  if isinstance(t, type))
    identified = (bool(kinds) and isinstance(error, kinds)) or 'linalg.eigh' in message
    return bool(failed and identified)


class GuardedEigh:
    """An eigh-callable with explicit refusal and numerical evidence.

    Optional solver arguments are injectable for deterministic software tests.
    Production callers should retain the native defaults.
    """
    def __init__(self, audit_callback=None, diagnostic_dir=None,
                 primary_solver=None, cpu_solver=None):
        self.primary_solver = primary_solver or torch.linalg.eigh
        self.cpu_solver = cpu_solver or torch.linalg.eigh
        self.audit_callback = audit_callback
        self.diagnostic_dir = Path(diagnostic_dir).resolve() if diagnostic_dir is not None else None
        self.calls = 0
        self.fallbacks = 0

    def _record(self, matrix):
        return {'policy': POLICY, 'call': self.calls, 'shape': list(matrix.shape),
                'dtype': str(matrix.dtype), 'device': str(matrix.device),
                'matrix_changed': False, 'jitter_added': False,
                'finite': None, 'symmetry_max_abs': None}

    def _capture(self, matrix, record):
        if self.diagnostic_dir is None:
            record['matrix_file'] = None
            return
        # This directory must be a NEW recovery output, never the failed run.
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        filename = 'eigh_L_{}_{}_{}.pt'.format(os.getpid(), self.calls, time.time_ns())
        path = self.diagnostic_dir / filename
        with path.open('xb') as handle:
            torch.save({'L': matrix.detach().to('cpu').clone(), 'audit': dict(record)}, handle)
        value = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                value.update(block)
        record['matrix_file'] = str(path)
        record['matrix_file_sha256'] = value.hexdigest()

    def _emit(self, record):
        if self.audit_callback is not None:
            self.audit_callback(dict(record))

    def _reject(self, matrix, record, reason, error=None):
        record.update(event='eigh_guard_rejected', reason=reason)
        if error is not None:
            record['error'] = '{}: {}'.format(type(error).__name__, error)
        self._capture(matrix, record)
        self._emit(record)
        raise EigenGuardError(reason) from error

    @staticmethod
    def _quality(matrix, values, vectors):
        n = matrix.shape[-1]
        if values.shape != matrix.shape[:-1] or vectors.shape != matrix.shape:
            raise EigenGuardError('eigh returned unexpected shapes')
        if not torch.isfinite(values).all().item() or not torch.isfinite(vectors).all().item():
            raise EigenGuardError('eigh returned nonfinite eigenvalues or eigenvectors')
        error = matrix @ vectors - vectors * values.unsqueeze(-2)
        denominator = torch.linalg.matrix_norm(matrix).clamp_min(1.0)
        residual = (torch.linalg.matrix_norm(error) / denominator).max().item()
        identity = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
        orthogonal = (torch.linalg.matrix_norm(vectors.transpose(-1, -2) @ vectors - identity)
                      / max(1.0, float(n) ** 0.5)).max().item()
        if not math.isfinite(residual) or not math.isfinite(orthogonal):
            raise EigenGuardError('Nonfinite eigensolver quality audit; overflow cannot pass the guard')
        return {'residual_relative_frobenius': float(residual),
                'orthogonality_normalized_frobenius': float(orthogonal)}

    def __call__(self, matrix, UPLO='L'):
        self.calls += 1
        if not torch.is_tensor(matrix):
            raise EigenGuardError('Expected a torch tensor')
        record = self._record(matrix)
        if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2] or matrix.numel() == 0:
            self._reject(matrix, record, 'Expected a nonempty square matrix or matrix batch')
        if matrix.dtype not in (torch.float32, torch.float64) or matrix.is_complex():
            self._reject(matrix, record, 'This sampling guard accepts real float32/float64 matrices only')
        if UPLO not in ('L', 'U'):
            self._reject(matrix, record, 'UPLO must be L or U')
        finite = bool(torch.isfinite(matrix).all().item())
        record['finite'] = finite
        if not finite:
            record['nan_count'] = int(torch.isnan(matrix).sum().item())
            record['inf_count'] = int(torch.isinf(matrix).sum().item())
            self._reject(matrix, record, 'Nonfinite L; no sanitization or solver retry allowed')
        symmetry = float((matrix - matrix.transpose(-1, -2)).abs().max().item())
        record['symmetry_max_abs'] = symmetry
        if symmetry != 0.0:
            self._reject(matrix, record, 'Asymmetric L; no symmetrization or solver retry allowed')
        try:
            # Return the native result itself: normal-path bits and return type
            # are unchanged. Validation creates no random draws.
            result = self.primary_solver(matrix, UPLO=UPLO)
            if (not torch.isfinite(result[0]).all().item()
                    or not torch.isfinite(result[1]).all().item()):
                self._reject(matrix, record, 'Native eigensolver returned nonfinite outputs; no silent retry')
            return result
        except RuntimeError as error:
            if not _nonconvergence(error):
                raise
            record['primary_error'] = '{}: {}'.format(type(error).__name__, error)
            if matrix.requires_grad:
                self._reject(matrix, record, 'CPU fallback is inference-only; L requires gradients', error)
            # Moving to float64 is exact for finite float32 entries. Do not
            # average/symmetrize/perturb this matrix after its validation.
            cpu = matrix.detach().to(device='cpu', dtype=torch.float64).clone()
            try:
                values, vectors = self.cpu_solver(cpu, UPLO=UPLO)
                quality = self._quality(cpu, values, vectors)
            except (RuntimeError, ValueError) as cpu_error:
                self._reject(matrix, record, 'CPU float64 eigensolver failed or returned invalid outputs', cpu_error)
            record['cpu_float64_quality'] = quality
            tolerance = 1e-10
            if any(v > tolerance for v in quality.values()):
                self._reject(matrix, record, 'CPU float64 eigensolver residual/orthogonality audit failed')
            cast_values = values.to(device=matrix.device, dtype=matrix.dtype)
            cast_vectors = vectors.to(device=matrix.device, dtype=matrix.dtype)
            cast_quality = self._quality(cpu, cast_values.detach().to('cpu', torch.float64),
                                         cast_vectors.detach().to('cpu', torch.float64))
            cast_tolerance = max(tolerance, float(torch.finfo(matrix.dtype).eps)
                                 * max(64, 8 * matrix.shape[-1]))
            record['cast_back_quality'] = cast_quality
            record['cpu_float64_tolerance'] = tolerance
            record['cast_back_tolerance'] = cast_tolerance
            if any(v > cast_tolerance for v in cast_quality.values()):
                self._reject(matrix, record, 'Cast-back eigenpairs failed their numerical audit')
            self.fallbacks += 1
            record.update(event='eigh_cpu_float64_fallback', fallback_index=self.fallbacks,
                          solver='torch.linalg.eigh on unchanged CPU float64 L',
                          eigenbasis_bit_identity_claimed=False)
            self._capture(matrix, record)
            self._emit(record)
            return cast_values, cast_vectors


class _LinalgProxy:
    def __init__(self, original, solver):
        self._original = original
        self.eigh = solver

    def __getattr__(self, name):
        return getattr(self._original, name)


class _TorchProxy:
    def __init__(self, original, solver):
        self._original = original
        self.linalg = _LinalgProxy(original.linalg, solver)

    def __getattr__(self, name):
        return getattr(self._original, name)


class GuardedEigenFeatures:
    """Delegate an existing feature object through its unchanged Python body."""
    def __init__(self, existing, audit_callback, diagnostic_dir=None):
        if getattr(existing, 'mode', None) not in ('all', 'not-clustering'):
            raise EigenGuardError('Recovery supports only the existing full eigenvector feature modes')
        function = type(existing).__call__
        if not isinstance(function, types.FunctionType) or function.__closure__:
            raise EigenGuardError('Expected an ordinary EigenFeatures Python call implementation')
        if function.__globals__.get('torch') is not torch:
            raise EigenGuardError('Unexpected torch binding in EigenFeatures implementation')
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'eigh'
                 and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'linalg'
                 and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == 'torch']
        if len(calls) != 1:
            raise EigenGuardError('Expected exactly one direct torch.linalg.eigh call in EigenFeatures')
        self.original = existing
        self.solver = GuardedEigh(audit_callback, diagnostic_dir)
        private_globals = dict(function.__globals__)
        private_globals['torch'] = _TorchProxy(torch, self.solver)
        self._call = types.FunctionType(function.__code__, private_globals,
                                        function.__name__, function.__defaults__, None)
        self._call.__kwdefaults__ = function.__kwdefaults__

    def __call__(self, noisy_data):
        return self._call(self.original, noisy_data)

    def __getattr__(self, name):
        return getattr(self.original, name)


def make_guarded_eigenfeatures(existing, audit_callback, diagnostic_dir=None):
    return GuardedEigenFeatures(existing, audit_callback, diagnostic_dir)
