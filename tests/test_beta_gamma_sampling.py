"""Regression tests for the dtype-tiny Gamma underflow guard."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dgd"))

from diffusion.beta_diffusion_utils import (  # noqa: E402
    forward_beta_sample,
    positive_gamma_sample,
    reverse_beta_step,
)


def test_gamma_shapes_are_positive_finite_and_reproducible():
    for shape in (1.0e-6, 1.0e-5, 1.0e-3, 0.1, 1.0, 100.0):
        parameter = torch.full((4096,), shape)
        torch.manual_seed(12345)
        first = positive_gamma_sample(parameter)
        torch.manual_seed(12345)
        second = positive_gamma_sample(parameter)
        assert torch.equal(first, second)
        assert torch.isfinite(first).all()
        assert (first > 0).all()
        assert torch.isfinite(first.log()).all()
        assert float((first == 1.0e-6).float().mean()) < 0.01
        assert first.min() >= torch.finfo(first.dtype).tiny


def test_forward_and_reverse_remain_finite():
    clean = torch.tensor([[[0.09], [0.99]]])
    eta = torch.full_like(clean, 100.0)
    alpha_t = torch.tensor([[[2.3e-6]]])
    torch.manual_seed(9)
    state, diagnostics = forward_beta_sample(
        clean, alpha_t, eta, 1.0e-6, "logit")
    assert torch.isfinite(state).all()
    assert diagnostics.nan_count == 0 and diagnostics.inf_count == 0
    torch.manual_seed(10)
    updated, reverse_diagnostics = reverse_beta_step(
        state, clean, alpha_t,
        torch.tensor([[[3.0e-6]]]), eta, 1.0e-6, "logit")
    assert torch.isfinite(updated).all()
    assert reverse_diagnostics.nan_count == 0
    assert reverse_diagnostics.inf_count == 0


if __name__ == "__main__":
    test_gamma_shapes_are_positive_finite_and_reproducible()
    test_forward_and_reverse_remain_finite()
    print("beta Gamma sampling tests passed")
