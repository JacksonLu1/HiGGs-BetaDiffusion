"""Float64 accuracy and strict-failure tests for beta_h2 KLUB terms."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dgd"))

from diffusion.beta_diffusion_utils import beta_kl  # noqa: E402


def reference_beta_kl(a_p, b_p, a_q, b_q):
    ap, bp, aq, bq = (value.to(torch.float64)
                      for value in (a_p, b_p, a_q, b_q))
    return (
        (torch.lgamma(aq) + torch.lgamma(bq) - torch.lgamma(aq + bq))
        - (torch.lgamma(ap) + torch.lgamma(bp) - torch.lgamma(ap + bp))
        + (ap - aq) * (torch.digamma(ap) - torch.digamma(ap + bp))
        + (bp - bq) * (torch.digamma(bp) - torch.digamma(ap + bp))
    )


def test_identical_and_parameter_ranges():
    for values in (
            torch.tensor([1.0e-6, 1.0e-5, 1.0e-3, 0.1, 1.0, 100.0]),
            torch.tensor([5.3e-6, 2.0e-5, 2.24e-4, 10.0, 100.0, 1000.0])):
        result, diagnostics = beta_kl(
            values, values + 1.0, values, values + 1.0, 1.0e-6,
            return_diagnostics=True)
        assert result.dtype == torch.float64
        assert result.abs().max() <= 1.0e-12
        assert diagnostics["materially_negative_count"] == 0
        assert diagnostics["nan_kl_count"] == 0
        assert diagnostics["inf_kl_count"] == 0


def test_against_float64_reference_and_no_false_zeros():
    torch.manual_seed(7)
    alpha = torch.sigmoid(torch.linspace(-13.0, 10.0, 101))
    all_result, all_reference = [], []
    for step in (0, 1, 10, 50, 97, 99):
        delta = alpha[step + 1] - alpha[step]
        alpha_s = alpha[step + 1]
        for difference in (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4):
            target = torch.tensor([0.09, 0.99] * 128)
            direction = torch.where(target < 0.5, 1.0, -1.0)
            predicted = target + direction * difference
            a_p = 100.0 * delta * predicted
            b_p = 100.0 * (1.0 - alpha_s * predicted)
            a_q = 100.0 * delta * target
            b_q = 100.0 * (1.0 - alpha_s * target)
            result, diagnostics = beta_kl(
                a_p, b_p, a_q, b_q, 1.0e-6,
                tensor_name=f"step_{step}_difference_{difference}",
                return_diagnostics=True)
            reference = reference_beta_kl(a_p, b_p, a_q, b_q)
            all_result.append(result)
            all_reference.append(reference)
            assert torch.isfinite(result).all()
            assert diagnostics["materially_negative_count"] == 0
    result = torch.cat(all_result)
    reference = torch.cat(all_reference)
    positive = reference > 1.0e-12
    false_zero_rate = ((result == 0) & positive).sum() / positive.sum()
    relative_error = ((result - reference).abs()
                      / reference.abs().clamp_min(1.0e-30))[positive].mean()
    assert false_zero_rate <= 1.0e-6
    assert relative_error < 1.0e-5


def test_mask_backward_and_strict_invalid_parameters():
    predicted = torch.tensor([0.2, 0.7], requires_grad=True)
    target = torch.tensor([0.3, 0.8])
    result = beta_kl(
        10.0 * predicted, 10.0 * (1.0 - predicted),
        10.0 * target, 10.0 * (1.0 - target), 1.0e-6)
    result.mean().backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.count_nonzero(predicted.grad) > 0

    masked_result, diagnostics = beta_kl(
        torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]),
        torch.tensor([1.5, 0.0]), torch.tensor([2.5, 0.0]), 1.0e-6,
        mask=torch.tensor([True, False]), return_diagnostics=True)
    assert masked_result[1] == 0
    assert diagnostics["valid_element_count"] == 1

    try:
        beta_kl(torch.tensor([0.0]), torch.tensor([1.0]),
                torch.tensor([1.0]), torch.tensor([1.0]), 1.0e-6,
                tensor_name="invalid_test")
    except RuntimeError as error:
        assert "invalid_test" in str(error) and "bad_count=1" in str(error)
    else:
        raise AssertionError("invalid effective Beta parameters did not fail")


if __name__ == "__main__":
    test_identical_and_parameter_ranges()
    test_against_float64_reference_and_no_false_zeros()
    test_mask_backward_and_strict_invalid_parameters()
    print("beta_kl numerical tests passed")
