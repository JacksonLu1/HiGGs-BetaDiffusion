"""Numerically safe beta-diffusion primitives shared by HiGGs Beta H2/H1.

The forward and reverse transitions follow Graph Beta Diffusion (GBD):

    G_t ~ Beta(eta * alpha_t * G_0, eta * (1 - alpha_t * G_0))
    G_s = G_t + P * (1 - G_t)

where ``P`` is beta distributed and ``s < t`` in diffusion time.  This module
contains no HiGGs stage-selection logic so h1 and glue remain independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


def fixed_h2_condition(num_classes: int = 2) -> torch.Tensor:
    """Return the non-semantic graph-level condition used by every h2 path.

    HiGGs h2 currently has no meaningful graph-level condition.  A fixed
    one-hot vector keeps the existing model dimensions while making training
    and inference deterministic and semantically consistent.
    """
    if num_classes < 1:
        raise ValueError("h2 condition must have at least one class")
    condition = torch.zeros((1, num_classes), dtype=torch.float32)
    condition[0, 0] = 1.0
    return condition


@dataclass
class BetaDiagnostics:
    min_parameter: torch.Tensor
    max_parameter: torch.Tensor
    nan_count: torch.Tensor
    inf_count: torch.Tensor
    gamma_clamp_count: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "min_beta_parameter": self.min_parameter,
            "max_beta_parameter": self.max_parameter,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "gamma_clamp_count": self.gamma_clamp_count,
        }


def sigmoid_alpha(t: torch.Tensor, start: float, end: float,
                  power: float) -> torch.Tensor:
    """Return the monotonically decreasing GBD sigmoid noise schedule."""
    t = t.clamp(0.0, 1.0)
    logit_alpha = start + (end - start) * t.pow(power)
    return torch.sigmoid(logit_alpha)


def scale_shift(value: torch.Tensor, scale: float, shift: float) -> torch.Tensor:
    return value * scale + shift


def inverse_scale_shift(value: torch.Tensor, scale: float,
                        shift: float) -> torch.Tensor:
    if scale <= 0:
        raise ValueError("scale_shift scale must be positive")
    return (value - shift) / scale


def logits_to_unit_probability(raw_logit: torch.Tensor) -> torch.Tensor:
    """Convert a network raw logit to [0, 1] exactly once."""
    return torch.sigmoid(raw_logit)


def unit_probability_to_shifted_state(unit_probability: torch.Tensor,
                                      scale: float,
                                      shift: float) -> torch.Tensor:
    """Map a clean unit probability into the Beta shifted state domain."""
    return scale_shift(unit_probability, scale, shift)


def shifted_clean_to_unit_probability(shifted_clean: torch.Tensor,
                                      scale: float,
                                      shift: float) -> torch.Tensor:
    """Invert a shifted clean prediction without applying sigmoid again."""
    return inverse_scale_shift(shifted_clean, scale, shift).clamp(0.0, 1.0)


def safe_beta_parameters(a: torch.Tensor, b: torch.Tensor,
                         eps: float) -> Tuple[torch.Tensor, torch.Tensor, BetaDiagnostics]:
    """Clamp beta parameters and report diagnostics before sampling."""
    raw = torch.cat((a.reshape(-1), b.reshape(-1)))
    nan_count = torch.isnan(raw).sum()
    inf_count = torch.isinf(raw).sum()
    a = torch.nan_to_num(a, nan=eps, posinf=1.0 / eps, neginf=eps).clamp_min(eps)
    b = torch.nan_to_num(b, nan=eps, posinf=1.0 / eps, neginf=eps).clamp_min(eps)
    combined = torch.cat((a.reshape(-1), b.reshape(-1)))
    diagnostics = BetaDiagnostics(
        min_parameter=combined.min().detach(),
        max_parameter=combined.max().detach(),
        nan_count=nan_count.detach(),
        inf_count=inf_count.detach(),
        gamma_clamp_count=torch.zeros((), dtype=torch.long, device=a.device),
    )
    return a, b, diagnostics


def positive_gamma_sample(shape: torch.Tensor,
                          return_clamp_count: bool = False):
    """Sample Gamma(shape, 1) with only a dtype-level underflow guard."""
    sample = torch._standard_gamma(shape.float())
    tiny = torch.finfo(sample.dtype).tiny
    clamp_count = (sample < tiny).sum().detach()
    sample = sample.clamp_min(tiny)
    return (sample, clamp_count) if return_clamp_count else sample


def gamma_beta_sample(a: torch.Tensor, b: torch.Tensor, eps: float,
                      return_logit: bool) -> Tuple[torch.Tensor, BetaDiagnostics]:
    """Sample Beta(a, b) through two Gamma variables, as in GBD."""
    a, b, diagnostics = safe_beta_parameters(a, b, eps)
    gamma_u, clamp_u = positive_gamma_sample(a, return_clamp_count=True)
    gamma_v, clamp_v = positive_gamma_sample(b, return_clamp_count=True)
    diagnostics.gamma_clamp_count = clamp_u + clamp_v
    # This is an underflow guard, not a distribution-level epsilon.  Using
    # beta_eps here biases nearly every draw when the Gamma shape is tiny.
    log_u = torch.log(gamma_u)
    log_v = torch.log(gamma_v)
    logit = log_u - log_v
    sample = logit if return_logit else torch.sigmoid(logit)
    return sample.to(a.dtype), diagnostics


def forward_beta_sample(clean: torch.Tensor, alpha_t: torch.Tensor,
                        eta: torch.Tensor, eps: float,
                        input_space: str) -> Tuple[torch.Tensor, BetaDiagnostics]:
    """Sample a marginal forward state in original or logit space."""
    if input_space not in {"logit", "original"}:
        raise ValueError(f"Unsupported beta input_space: {input_space}")
    a = eta * alpha_t * clean
    b = eta * (1.0 - alpha_t * clean)
    return gamma_beta_sample(a, b, eps, return_logit=input_space == "logit")


def reverse_beta_step(current: torch.Tensor, predicted_clean: torch.Tensor,
                      alpha_t: torch.Tensor, alpha_s: torch.Tensor,
                      eta: torch.Tensor, eps: float,
                      input_space: str) -> Tuple[torch.Tensor, BetaDiagnostics]:
    """Perform one GBD reverse thickening transition from time t to s."""
    delta_alpha = (alpha_s - alpha_t).clamp_min(eps)
    a = eta * delta_alpha * predicted_clean
    b = eta * (1.0 - alpha_s * predicted_clean)
    logit_p, diagnostics = gamma_beta_sample(a, b, eps, return_logit=True)

    if input_space == "logit":
        # logit(x + p(1-x)) = logsumexp(logit(x), logit(p), logit(x)+logit(p)).
        stacked = torch.stack((current, logit_p, current + logit_p), dim=-1)
        updated = torch.logsumexp(stacked, dim=-1)
    elif input_space == "original":
        p = torch.sigmoid(logit_p)
        updated = current + p * (1.0 - current)
        updated = updated.clamp(eps, 1.0 - eps)
    else:
        raise ValueError(f"Unsupported beta input_space: {input_space}")
    return updated, diagnostics


def beta_kl(a_p: torch.Tensor, b_p: torch.Tensor,
            a_q: torch.Tensor, b_q: torch.Tensor,
            eps: float, mask: torch.Tensor | None = None,
            tensor_name: str = "beta_kl",
            negative_tolerance: float = 1.0e-10,
            context=None,
            return_diagnostics: bool = False):
    """Elementwise ``KL(Beta(a_p,b_p) || Beta(a_q,b_q))`` in float64.

    Only masked-in elements are validated and reduced by callers.  Invalid
    parameters, non-finite KL values, and materially negative KL values are
    errors rather than silently converted to zero.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if negative_tolerance < 0:
        raise ValueError("negative_tolerance must be non-negative")

    original_dtype = a_p.dtype
    device_type = a_p.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        parameters = torch.broadcast_tensors(
            a_p.to(torch.float64), b_p.to(torch.float64),
            a_q.to(torch.float64), b_q.to(torch.float64))
        a_p64, b_p64, a_q64, b_q64 = parameters
        if mask is None:
            valid_mask = torch.ones_like(a_p64, dtype=torch.bool)
        else:
            valid_mask = mask.to(device=a_p64.device, dtype=torch.bool)
            while valid_mask.ndim < a_p64.ndim:
                valid_mask = valid_mask.unsqueeze(-1)
            valid_mask = torch.broadcast_to(valid_mask, a_p64.shape)

        stacked = torch.stack(parameters, dim=-1)
        parameter_ok = torch.isfinite(stacked) & (stacked > 0)
        invalid_parameter_count = (
            ((~parameter_ok) & valid_mask.unsqueeze(-1)).sum().detach())

        def error_context():
            value = context() if callable(context) else context
            return f", context={value}" if value else ""

        def parameter_range():
            finite_valid_parameters = stacked[
                valid_mask.unsqueeze(-1).expand_as(stacked)
                & torch.isfinite(stacked)]
            if not finite_valid_parameters.numel():
                return float("nan"), float("nan")
            return (float(finite_valid_parameters.min().detach().cpu()),
                    float(finite_valid_parameters.max().detach().cpu()))

        if bool(invalid_parameter_count.cpu()):
            parameter_min, parameter_max = parameter_range()
            raise RuntimeError(
                f"{tensor_name}: invalid Beta parameters; dtype={original_dtype}, "
                f"min={parameter_min}, max={parameter_max}, "
                f"bad_count={int(invalid_parameter_count.cpu())}{error_context()}")

        # Masked positions may legitimately contain zeros from graph padding.
        # Replace only those positions before evaluating special functions.
        one = torch.ones((), dtype=torch.float64, device=a_p64.device)
        a_p64 = torch.where(valid_mask, a_p64, one)
        b_p64 = torch.where(valid_mask, b_p64, one)
        a_q64 = torch.where(valid_mask, a_q64, one)
        b_q64 = torch.where(valid_mask, b_q64, one)

        log_beta_p = (torch.lgamma(a_p64) + torch.lgamma(b_p64)
                      - torch.lgamma(a_p64 + b_p64))
        log_beta_q = (torch.lgamma(a_q64) + torch.lgamma(b_q64)
                      - torch.lgamma(a_q64 + b_q64))
        kl = (
            log_beta_q - log_beta_p
            + (a_p64 - a_q64)
            * (torch.digamma(a_p64) - torch.digamma(a_p64 + b_p64))
            + (b_p64 - b_q64)
            * (torch.digamma(b_p64) - torch.digamma(a_p64 + b_p64))
        )

        valid_kl = kl[valid_mask]
        nan_kl_count = torch.isnan(valid_kl).sum().detach()
        inf_kl_count = torch.isinf(valid_kl).sum().detach()
        if bool((nan_kl_count + inf_kl_count).cpu()):
            parameter_min, parameter_max = parameter_range()
            raise RuntimeError(
                f"{tensor_name}: non-finite Beta KL; dtype={kl.dtype}, "
                f"parameter_min={parameter_min}, parameter_max={parameter_max}, "
                f"nan_count={int(nan_kl_count.cpu())}, "
                f"inf_count={int(inf_kl_count.cpu())}{error_context()}")

        tiny_negative = valid_mask & (kl < 0) & (kl >= -negative_tolerance)
        materially_negative = valid_mask & (kl < -negative_tolerance)
        tiny_negative_count = tiny_negative.sum().detach()
        materially_negative_count = materially_negative.sum().detach()
        if bool(materially_negative_count.cpu()):
            negative_values = kl[materially_negative]
            parameter_min, parameter_max = parameter_range()
            raise RuntimeError(
                f"{tensor_name}: materially negative Beta KL; dtype={kl.dtype}, "
                f"kl_min={float(negative_values.min().detach().cpu())}, "
                f"parameter_min={parameter_min}, parameter_max={parameter_max}, "
                f"bad_count={int(materially_negative_count.cpu())}, "
                f"tolerance={negative_tolerance}{error_context()}")

        kl = torch.where(tiny_negative, torch.zeros_like(kl), kl)
        kl = torch.where(valid_mask, kl, torch.zeros_like(kl))
        corrected_valid_kl = kl[valid_mask]
        valid_element_count = valid_mask.sum().detach()
        if corrected_valid_kl.numel():
            min_valid_kl = corrected_valid_kl.min().detach()
            max_valid_kl = corrected_valid_kl.max().detach()
            mean_valid_kl = corrected_valid_kl.mean().detach()
            zero_kl_count = (corrected_valid_kl == 0).sum().detach()
            positive_kl_count = (corrected_valid_kl > 0).sum().detach()
        else:
            zero64 = torch.zeros((), dtype=torch.float64, device=kl.device)
            min_valid_kl = max_valid_kl = mean_valid_kl = zero64
            zero_kl_count = positive_kl_count = torch.zeros(
                (), dtype=torch.long, device=kl.device)

        diagnostics = {
            "valid_element_count": valid_element_count,
            "invalid_parameter_count": invalid_parameter_count,
            "nan_kl_count": nan_kl_count,
            "inf_kl_count": inf_kl_count,
            "tiny_negative_count": tiny_negative_count,
            "materially_negative_count": materially_negative_count,
            "zero_kl_count": zero_kl_count,
            "positive_kl_count": positive_kl_count,
            "min_valid_kl": min_valid_kl,
            "max_valid_kl": max_valid_kl,
            "mean_valid_kl": mean_valid_kl,
        }
    return (kl, diagnostics) if return_diagnostics else kl


def categorical_precondition_stats(alpha_t: torch.Tensor, eta: torch.Tensor,
                                   probability: torch.Tensor, low: float,
                                   high: float, input_space: str,
                                   eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean/std for a Bernoulli mixture over the two scale-shift bounds."""
    probability = probability.to(device=alpha_t.device, dtype=alpha_t.dtype)
    while probability.ndim < alpha_t.ndim:
        probability = probability.unsqueeze(0)
    z = torch.stack((torch.full_like(probability, low),
                     torch.full_like(probability, high)), dim=0)
    probs = torch.stack((1.0 - probability, probability), dim=0)
    alpha = alpha_t.unsqueeze(0)
    eta_expanded = eta.unsqueeze(0)

    if input_space == "logit":
        a = (eta_expanded * alpha * z).clamp_min(eps)
        b = (eta_expanded * (1.0 - alpha * z)).clamp_min(eps)
        component_mean = torch.digamma(a) - torch.digamma(b)
        component_var = torch.polygamma(1, a) + torch.polygamma(1, b)
        mean = (probs * component_mean).sum(dim=0)
        second = (probs * (component_var + component_mean.square())).sum(dim=0)
        variance = (second - mean.square()).clamp_min(eps)
    elif input_space == "original":
        mean0 = (probs * z).sum(dim=0)
        second0 = (probs * z.square()).sum(dim=0)
        var0 = (second0 - mean0.square()).clamp_min(0.0)
        mean = alpha_t * mean0
        variance = ((alpha_t * mean0 - alpha_t.square() * second0) /
                    (eta + 1.0) + alpha_t.square() * var0).clamp_min(eps)
    else:
        raise ValueError(f"Unsupported beta input_space: {input_space}")
    return mean, variance.sqrt().clamp_min(eps)


def apply_graph_masks(X: torch.Tensor, E: torch.Tensor,
                      node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask padding, remove diagonal edges, and enforce undirected symmetry."""
    x_mask = node_mask.unsqueeze(-1)
    edge_mask = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).unsqueeze(-1)
    X = X * x_mask
    E = E * edge_mask
    n = E.size(1)
    diagonal = torch.eye(n, device=E.device, dtype=torch.bool).unsqueeze(0)
    E = E.masked_fill(diagonal.unsqueeze(-1), 0.0)
    upper_mask = torch.triu(
        torch.ones((n, n), device=E.device, dtype=E.dtype), diagonal=1
    ).view(1, n, n, 1)
    upper = E * upper_mask
    E = upper + upper.transpose(1, 2)
    return X, E


def valid_upper_edge_mask(node_mask: torch.Tensor) -> torch.Tensor:
    """Select each valid undirected, non-diagonal edge exactly once."""
    valid_pair = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    upper = torch.triu(torch.ones_like(valid_pair), diagonal=1).bool()
    return valid_pair & upper


def discretize_graph(node_values: torch.Tensor, edge_values: torch.Tensor,
                     node_mask: torch.Tensor, edge_threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert continuous class channels to HiGGs-compatible discrete tensors."""
    node_types = node_values.argmax(dim=-1)
    max_edge, edge_channel = edge_values.max(dim=-1)
    edge_types = torch.where(max_edge > edge_threshold, edge_channel + 1,
                             torch.zeros_like(edge_channel))
    edge_types = torch.triu(edge_types, diagonal=1)
    edge_types = edge_types + edge_types.transpose(1, 2)
    valid_edges = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    edge_types = edge_types.masked_fill(~valid_edges, 0)
    node_types = node_types.masked_fill(~node_mask, -1)
    return node_types, edge_types
