"""GBD KLUB training objective for the conditional HiGGs h2 model."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from diffusion.beta_diffusion_utils import beta_kl


class BetaH2Loss(nn.Module):
    """Sampling/correction Beta KL loss with per-element concentration."""

    def __init__(self, lambda_edge: float, sampling_weight: float,
                 correction_weight: float, eps: float):
        super().__init__()
        if sampling_weight < 0 or correction_weight < 0:
            raise ValueError("Beta loss weights must be non-negative")
        self.lambda_edge = float(lambda_edge)
        self.sampling_weight = float(sampling_weight)
        self.correction_weight = float(correction_weight)
        self.eps = float(eps)

    def _component(self, predicted: torch.Tensor, target: torch.Tensor,
                   eta: torch.Tensor, alpha_t: torch.Tensor,
                   alpha_s: torch.Tensor,
                   mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                torch.Tensor, Dict[str, torch.Tensor]]:
        delta = (alpha_s - alpha_t).clamp_min(self.eps)

        a_pred_sampling = eta * delta * predicted
        b_pred_sampling = eta * (1.0 - alpha_s * predicted)
        a_true_sampling = eta * delta * target
        b_true_sampling = eta * (1.0 - alpha_s * target)

        a_pred_correction = eta * alpha_t * predicted
        b_pred_correction = eta * (1.0 - alpha_t * predicted)
        a_true_correction = eta * alpha_t * target
        b_true_correction = eta * (1.0 - alpha_t * target)

        expanded_mask = mask
        while expanded_mask.ndim < predicted.ndim:
            expanded_mask = expanded_mask.unsqueeze(-1)
        expanded_mask = expanded_mask.expand_as(predicted)
        if not expanded_mask.any():
            zero = predicted.double().sum() * 0.0
            zero_count = torch.zeros((), device=predicted.device, dtype=torch.long)
            diagnostics = {
                key: zero_count for key in (
                    "valid_element_count", "kl_evaluation_count",
                    "invalid_parameter_count", "nan_kl_count", "inf_kl_count",
                    "tiny_negative_count", "materially_negative_count",
                    "zero_kl_count", "positive_kl_count")}
            diagnostics.update({
                "min_valid_kl": zero.detach(),
                "max_valid_kl": zero.detach(),
                "mean_valid_kl": zero.detach(),
            })
            return zero, zero, zero.detach(), zero.detach(), diagnostics

        def alpha_context():
            return (
                f"alpha_t=[{float(alpha_t.min().detach().cpu())},"
                f"{float(alpha_t.max().detach().cpu())}], "
                f"alpha_s=[{float(alpha_s.min().detach().cpu())},"
                f"{float(alpha_s.max().detach().cpu())}]")
        sampling_values, sampling_diag = beta_kl(
            a_pred_sampling, b_pred_sampling,
            a_true_sampling, b_true_sampling, self.eps,
            mask=expanded_mask, tensor_name="sampling_kl",
            context=alpha_context, return_diagnostics=True)
        correction_values, correction_diag = beta_kl(
            a_pred_correction, b_pred_correction,
            a_true_correction, b_true_correction, self.eps,
            mask=expanded_mask, tensor_name="correction_kl",
            context=alpha_context, return_diagnostics=True)

        sampling = sampling_values[expanded_mask].mean()
        correction = correction_values[expanded_mask].mean()
        parameter_values = torch.cat(tuple(
            value.double()[expanded_mask].reshape(-1) for value in (
                a_pred_sampling, b_pred_sampling, a_true_sampling, b_true_sampling,
                a_pred_correction, b_pred_correction,
                a_true_correction, b_true_correction)))

        valid_count = sampling_diag["valid_element_count"]
        evaluation_count = valid_count * 2
        diagnostics = {
            "valid_element_count": valid_count,
            "kl_evaluation_count": evaluation_count,
        }
        for key in ("invalid_parameter_count", "nan_kl_count", "inf_kl_count",
                    "tiny_negative_count", "materially_negative_count",
                    "zero_kl_count", "positive_kl_count"):
            diagnostics[key] = sampling_diag[key] + correction_diag[key]
        diagnostics["min_valid_kl"] = torch.minimum(
            sampling_diag["min_valid_kl"], correction_diag["min_valid_kl"])
        diagnostics["max_valid_kl"] = torch.maximum(
            sampling_diag["max_valid_kl"], correction_diag["max_valid_kl"])
        diagnostics["mean_valid_kl"] = (
            sampling_diag["mean_valid_kl"] + correction_diag["mean_valid_kl"]
        ) / 2.0
        return (sampling, correction, parameter_values.min().detach(),
                parameter_values.max().detach(), diagnostics)

    def forward(self, predicted_X: torch.Tensor, predicted_E: torch.Tensor,
                true_X: torch.Tensor, true_E: torch.Tensor,
                eta_X: torch.Tensor, eta_E: torch.Tensor,
                alpha_t: torch.Tensor, alpha_s: torch.Tensor,
                node_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_mask = node_mask
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        upper = torch.triu(torch.ones_like(edge_mask), diagonal=1).bool()
        edge_mask = edge_mask & upper

        alpha_X_t = alpha_t.view(-1, 1, 1)
        alpha_X_s = alpha_s.view(-1, 1, 1)
        alpha_E_t = alpha_t.view(-1, 1, 1, 1)
        alpha_E_s = alpha_s.view(-1, 1, 1, 1)

        X_sampling, X_correction, X_min, X_max, X_diag = self._component(
            predicted_X, true_X, eta_X, alpha_X_t, alpha_X_s, x_mask)
        E_sampling, E_correction, E_min, E_max, E_diag = self._component(
            predicted_E, true_E, eta_E, alpha_E_t, alpha_E_s, edge_mask)

        X_loss = (self.sampling_weight * X_sampling
                  + self.correction_weight * X_correction)
        E_loss = (self.sampling_weight * E_sampling
                  + self.correction_weight * E_correction)
        loss = X_loss + self.lambda_edge * E_loss

        valid_node_count = node_mask.sum()
        valid_edge_pair_count = edge_mask.sum()
        valid_element_count = (X_diag["valid_element_count"]
                               + E_diag["valid_element_count"])
        kl_evaluation_count = (X_diag["kl_evaluation_count"]
                               + E_diag["kl_evaluation_count"])
        if int(kl_evaluation_count.detach().cpu()) > 0:
            mean_valid_kl = (
                X_diag["mean_valid_kl"] * X_diag["kl_evaluation_count"]
                + E_diag["mean_valid_kl"] * E_diag["kl_evaluation_count"]
            ) / kl_evaluation_count
        else:
            mean_valid_kl = loss.detach() * 0.0

        metrics = {
            "loss": loss,
            "X_loss": X_loss,
            "E_loss": E_loss,
            "sampling_loss": X_sampling + self.lambda_edge * E_sampling,
            "correction_loss": X_correction + self.lambda_edge * E_correction,
            "min_beta_parameter": torch.minimum(X_min, E_min),
            "max_beta_parameter": torch.maximum(X_max, E_max),
            "nan_count": X_diag["invalid_parameter_count"] * 0,
            "inf_count": X_diag["invalid_parameter_count"] * 0,
            "valid_node_count": valid_node_count,
            "valid_edge_pair_count": valid_edge_pair_count,
            "valid_element_count": valid_element_count,
            "invalid_parameter_count": (X_diag["invalid_parameter_count"]
                                        + E_diag["invalid_parameter_count"]),
            "nan_kl_count": X_diag["nan_kl_count"] + E_diag["nan_kl_count"],
            "inf_kl_count": X_diag["inf_kl_count"] + E_diag["inf_kl_count"],
            "tiny_negative_count": (X_diag["tiny_negative_count"]
                                    + E_diag["tiny_negative_count"]),
            "materially_negative_count": (X_diag["materially_negative_count"]
                                          + E_diag["materially_negative_count"]),
            "zero_kl_count": X_diag["zero_kl_count"] + E_diag["zero_kl_count"],
            "positive_kl_count": (X_diag["positive_kl_count"]
                                  + E_diag["positive_kl_count"]),
            "min_valid_kl": torch.minimum(X_diag["min_valid_kl"],
                                           E_diag["min_valid_kl"]),
            "max_valid_kl": torch.maximum(X_diag["max_valid_kl"],
                                           E_diag["max_valid_kl"]),
            "mean_valid_kl": mean_valid_kl,
        }
        return loss, metrics


class BetaH1Loss(BetaH2Loss):
    """H1 KLUB: optional node channels plus valid strict-upper edges."""

    def forward(self, predicted_X: torch.Tensor, predicted_E: torch.Tensor,
                true_X: torch.Tensor, true_E: torch.Tensor,
                eta_X: torch.Tensor, eta_E: torch.Tensor,
                alpha_t: torch.Tensor, alpha_s: torch.Tensor,
                node_mask: torch.Tensor, diffuse_X: bool):
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_mask &= torch.triu(torch.ones_like(edge_mask), diagonal=1).bool()
        alpha_X_t = alpha_t.view(-1, 1, 1)
        alpha_X_s = alpha_s.view(-1, 1, 1)
        alpha_E_t = alpha_t.view(-1, 1, 1, 1)
        alpha_E_s = alpha_s.view(-1, 1, 1, 1)

        E_sampling, E_correction, E_min, E_max, E_diag = self._component(
            predicted_E, true_E, eta_E, alpha_E_t, alpha_E_s, edge_mask)
        E_loss = (self.sampling_weight * E_sampling
                  + self.correction_weight * E_correction)

        if diffuse_X:
            X_sampling, X_correction, X_min, X_max, X_diag = self._component(
                predicted_X, true_X, eta_X, alpha_X_t, alpha_X_s, node_mask)
            X_loss = (self.sampling_weight * X_sampling
                      + self.correction_weight * X_correction)
        else:
            X_loss = predicted_X.double().sum() * 0.0
            X_min, X_max = E_min.detach(), E_max.detach()
            X_diag = {key: torch.zeros_like(value)
                      for key, value in E_diag.items()}

        loss = X_loss + self.lambda_edge * E_loss
        return loss, {
            "X_loss": X_loss,
            "E_loss": E_loss,
            "min_beta_parameter": torch.minimum(X_min, E_min),
            "max_beta_parameter": torch.maximum(X_max, E_max),
            "beta_kl_negative_count": (
                X_diag["tiny_negative_count"]
                + X_diag["materially_negative_count"]
                + E_diag["tiny_negative_count"]
                + E_diag["materially_negative_count"]),
            "beta_kl_nonfinite_count": (
                X_diag["nan_kl_count"] + X_diag["inf_kl_count"]
                + E_diag["nan_kl_count"] + E_diag["inf_kl_count"]),
            "valid_node_count": (
                node_mask.sum() if diffuse_X else node_mask.sum() * 0),
            "valid_edge_pair_count": edge_mask.sum(),
        }
