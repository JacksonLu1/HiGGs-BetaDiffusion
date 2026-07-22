"""Conditional Graph Beta Diffusion for the HiGGs h2 stage only."""

from __future__ import annotations

import copy
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

import utils
from diffusion.beta_diffusion_utils import (
    apply_graph_masks,
    categorical_precondition_stats,
    discretize_graph,
    forward_beta_sample,
    fixed_h2_condition,
    logits_to_unit_probability,
    reverse_beta_step,
    shifted_clean_to_unit_probability,
    sigmoid_alpha,
    unit_probability_to_shifted_state,
)
from metrics.beta_train_metrics import BetaH2Loss
from models.transformer_model import GraphTransformer


class ConditionalGraphBetaDiffusionH2(pl.LightningModule):
    """GBD implementation that preserves the public HiGGs h2 sampling contract."""

    def __init__(self, cfg, dataset_infos, train_metrics=None,
                 sampling_metrics=None, visualization_tools=None,
                 extra_features=None, domain_features=None):
        super().__init__()
        if float(cfg.dataset.h) != 2.0:
            raise ValueError("ConditionalGraphBetaDiffusionH2 can only be used with dataset.h=2")

        self.cfg = cfg
        self.name = cfg.general.name
        self.T = int(cfg.model.diffusion_steps)
        self.dataset_info = dataset_infos
        self.node_dist = dataset_infos.nodes_dist
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features
        self.sampling_metrics = sampling_metrics

        source_input_dims = dict(dataset_infos.input_dims)
        source_output_dims = dict(dataset_infos.output_dims)
        if source_output_dims["E"] < 2:
            raise ValueError("h2 edge features must contain no-edge plus at least one edge class")

        self.Xdim_output = int(source_output_dims["X"])
        self.Edim_output = int(source_output_dims["E"])
        self.ydim_output = int(source_output_dims["y"])
        self.beta_Edim = self.Edim_output - 1

        # GBD diffuses only real edge channels; channel zero remains the reconstructed
        # no-edge class at the public interface.
        input_dims = dict(source_input_dims)
        output_dims = dict(source_output_dims)
        input_dims["E"] -= 1
        output_dims["E"] -= 1
        expected_y = self.ydim_output + 1
        if input_dims["y"] != expected_y:
            raise ValueError(
                "beta_h2 requires model.extra_features=null so y contains condition_y and time only "
                f"(expected {expected_y}, got {input_dims['y']})")

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.input_space = str(cfg.model.input_space)
        self.pre_condition = bool(cfg.model.pre_condition)
        self.sigmoid_start = float(cfg.model.sigmoid_start)
        self.sigmoid_end = float(cfg.model.sigmoid_end)
        self.sigmoid_power = float(cfg.model.sigmoid_power)
        self.node_scale = float(cfg.model.scale_shift.node[0])
        self.node_shift = float(cfg.model.scale_shift.node[1])
        self.edge_scale = float(cfg.model.scale_shift.edge[0])
        self.edge_shift = float(cfg.model.scale_shift.edge[1])
        self.eta_node = float(cfg.model.eta.node)
        self.eta_edge = float(cfg.model.eta.edge)
        self.eta_strategy = str(cfg.model.eta_strategy)
        if self.eta_strategy == "degree":
            warnings.warn(
                "beta_h2 eta_strategy=degree is experimental: reverse steps "
                "recompute eta from the current prediction, and this has not "
                "been proven to define one consistent Beta diffusion process. "
                "eta_strategy=fixed is the only formally supported mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.edge_threshold = float(cfg.model.edge_threshold)
        self.beta_eps = float(cfg.model.beta_eps)

        thresholds = cfg.model.get("degree_thresholds", [0.33, 0.66])
        multipliers = cfg.model.get("degree_eta_multipliers", [0.5, 1.0, 2.0])
        self.degree_thresholds = [float(value) for value in thresholds]
        self.degree_eta_multipliers = [float(value) for value in multipliers]
        if len(self.degree_eta_multipliers) != len(self.degree_thresholds) + 1:
            raise ValueError("degree_eta_multipliers must have one more value than degree_thresholds")

        self.loss_module = BetaH2Loss(
            lambda_edge=float(cfg.model.lambda_train[0]),
            sampling_weight=float(cfg.model.beta_loss.sampling_weight),
            correction_weight=float(cfg.model.beta_loss.correction_weight),
            eps=self.beta_eps,
        )

        node_prior = getattr(dataset_infos, "node_types", None)
        if node_prior is None or len(node_prior) != self.Xdim_output:
            node_prior = torch.full((self.Xdim_output,), 1.0 / self.Xdim_output)
        edge_prior = getattr(dataset_infos, "edge_types", None)
        if edge_prior is None or len(edge_prior) != self.Edim_output:
            edge_prior = torch.full((self.Edim_output,), 1.0 / self.Edim_output)
        self.register_buffer("node_channel_prior", torch.as_tensor(node_prior).float())
        self.register_buffer("edge_channel_prior", torch.as_tensor(edge_prior).float()[1:])

        try:
            checkpoint_dataset_infos = copy.copy(dataset_infos)
            if hasattr(checkpoint_dataset_infos, "datamodule"):
                checkpoint_dataset_infos.datamodule = None
            self.save_hyperparameters({
                "cfg": cfg,
                "dataset_infos": checkpoint_dataset_infos,
                "visualization_tools": visualization_tools,
                "extra_features": extra_features,
                "domain_features": domain_features,
                "model_family": "beta_h2",
                "stage": "h2",
            })
        except Exception as error:
            print(f"[HiGGs] beta_h2 hyperparameter serialization skipped: {error}")

        print("[HiGGs] Stage h2 model: Conditional Graph Beta Diffusion")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
            weight_decay=self.cfg.train.weight_decay)

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return sigmoid_alpha(t, self.sigmoid_start, self.sigmoid_end,
                             self.sigmoid_power)

    def _eta_tensors(self, clean_E: torch.Tensor,
                     node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n = node_mask.shape
        if self.eta_strategy == "fixed":
            eta_X = torch.full((batch_size, n, 1), self.eta_node,
                               device=clean_E.device, dtype=clean_E.dtype)
            eta_E = torch.full((batch_size, n, n, 1), self.eta_edge,
                               device=clean_E.device, dtype=clean_E.dtype)
        elif self.eta_strategy == "degree":
            degree = clean_E.sum(dim=-1).sum(dim=-1)
            degree = degree * node_mask
            max_degree = degree.max(dim=1, keepdim=True).values.clamp_min(1.0)
            normalized_degree = degree / max_degree
            thresholds = torch.tensor(self.degree_thresholds, device=clean_E.device,
                                      dtype=clean_E.dtype)
            bins = torch.bucketize(normalized_degree, thresholds)
            multipliers = torch.tensor(self.degree_eta_multipliers,
                                       device=clean_E.device, dtype=clean_E.dtype)
            node_multiplier = multipliers[bins] * node_mask
            # GBD Eq. 38 uses the more central endpoint for an undirected edge.
            edge_multiplier = torch.maximum(node_multiplier.unsqueeze(1),
                                            node_multiplier.unsqueeze(2))
            eta_X = (self.eta_node * node_multiplier).unsqueeze(-1)
            eta_E = (self.eta_edge * edge_multiplier).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported eta_strategy: {self.eta_strategy}")

        eta_X = eta_X.clamp_min(self.beta_eps)
        eta_E = eta_E.clamp_min(self.beta_eps)
        return eta_X, eta_E

    def _precondition(self, X_t: torch.Tensor, E_t: torch.Tensor,
                      alpha_t: torch.Tensor, eta_X: torch.Tensor,
                      eta_E: torch.Tensor, node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.pre_condition:
            return X_t, E_t
        mean_X, std_X = categorical_precondition_stats(
            alpha_t.view(-1, 1, 1), eta_X, self.node_channel_prior,
            self.node_shift, self.node_shift + self.node_scale,
            self.input_space, self.beta_eps)
        mean_E, std_E = categorical_precondition_stats(
            alpha_t.view(-1, 1, 1, 1), eta_E, self.edge_channel_prior,
            self.edge_shift, self.edge_shift + self.edge_scale,
            self.input_space, self.beta_eps)
        X_t = (X_t - mean_X) / std_X.clamp_min(self.beta_eps)
        E_t = (E_t - mean_E) / std_E.clamp_min(self.beta_eps)
        return apply_graph_masks(X_t, E_t, node_mask)

    def _predict_clean(self, X_t: torch.Tensor, E_t: torch.Tensor,
                       condition_y: torch.Tensor, t: torch.Tensor,
                       alpha_t: torch.Tensor, eta_X: torch.Tensor,
                       eta_E: torch.Tensor,
                       node_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        model_X, model_E = self._precondition(
            X_t, E_t, alpha_t, eta_X, eta_E, node_mask)
        global_condition = torch.cat((condition_y.float(), t.float()), dim=-1)
        prediction = self.model(model_X.float(), model_E.float(),
                                global_condition, node_mask)
        pred_X0 = unit_probability_to_shifted_state(
            logits_to_unit_probability(prediction.X),
            self.node_scale, self.node_shift)
        pred_E0 = unit_probability_to_shifted_state(
            logits_to_unit_probability(prediction.E),
            self.edge_scale, self.edge_shift)
        return apply_graph_masks(pred_X0, pred_E0, node_mask)

    def _forward_noise(self, X0: torch.Tensor, E0: torch.Tensor,
                       alpha_t: torch.Tensor, eta_X: torch.Tensor,
                       eta_E: torch.Tensor,
                       node_mask: torch.Tensor):
        X_t, X_diagnostics = forward_beta_sample(
            X0, alpha_t.view(-1, 1, 1), eta_X, self.beta_eps,
            self.input_space)
        E_t, E_diagnostics = forward_beta_sample(
            E0, alpha_t.view(-1, 1, 1, 1), eta_E, self.beta_eps,
            self.input_space)
        X_t, E_t = apply_graph_masks(X_t, E_t, node_mask)
        diagnostics = {
            "min_beta_parameter": torch.minimum(
                X_diagnostics.min_parameter, E_diagnostics.min_parameter),
            "max_beta_parameter": torch.maximum(
                X_diagnostics.max_parameter, E_diagnostics.max_parameter),
            "nan_count": X_diagnostics.nan_count + E_diagnostics.nan_count,
            "inf_count": X_diagnostics.inf_count + E_diagnostics.inf_count,
        }
        return X_t, E_t, diagnostics

    def _shared_step(self, data, stage: str):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X_discrete = dense_data.X.float()
        E_discrete = dense_data.E.float()
        X0 = unit_probability_to_shifted_state(
            X_discrete, self.node_scale, self.node_shift)
        E0 = unit_probability_to_shifted_state(
            E_discrete[..., 1:], self.edge_scale, self.edge_shift)
        X0, E0 = apply_graph_masks(X0, E0, node_mask)

        eta_X, eta_E = self._eta_tensors(E_discrete[..., 1:], node_mask)
        t = torch.rand((X0.size(0), 1), device=X0.device)
        t = t * (1.0 - 1.0 / self.T) + 1.0 / self.T
        s = (t - 1.0 / self.T).clamp_min(0.0)
        alpha_t = self._alpha(t)
        alpha_s = self._alpha(s)
        X_t, E_t, noise_diagnostics = self._forward_noise(
            X0, E0, alpha_t, eta_X, eta_E, node_mask)
        # Existing processed h2 caches may contain legacy random labels.  h2
        # has no semantic global condition, so canonicalize at the model
        # boundary as well as in dataset construction.
        condition_y = fixed_h2_condition(self.ydim_output).to(
            device=X0.device, dtype=X0.dtype).expand(X0.size(0), -1)
        pred_X0, pred_E0 = self._predict_clean(
            X_t, E_t, condition_y, t, alpha_t, eta_X, eta_E, node_mask)

        loss, metrics = self.loss_module(
            pred_X0, pred_E0, X0, E0, eta_X, eta_E,
            alpha_t, alpha_s, node_mask)
        metrics["min_beta_parameter"] = torch.minimum(
            metrics["min_beta_parameter"], noise_diagnostics["min_beta_parameter"])
        metrics["max_beta_parameter"] = torch.maximum(
            metrics["max_beta_parameter"], noise_diagnostics["max_beta_parameter"])
        metrics["nan_count"] = metrics["nan_count"] + noise_diagnostics["nan_count"]
        metrics["inf_count"] = metrics["inf_count"] + noise_diagnostics["inf_count"]

        prefix = "train" if stage == "train" else stage
        self.log(f"{prefix}/beta_h2_loss", loss, on_step=stage == "train",
                 on_epoch=True, batch_size=data.num_graphs)
        if stage == "train":
            self.log("train/beta_h2_X_loss", metrics["X_loss"], on_step=True,
                     on_epoch=True, batch_size=data.num_graphs)
            self.log("train/beta_h2_E_loss", metrics["E_loss"], on_step=True,
                     on_epoch=True, batch_size=data.num_graphs)
            self.log("train/beta_h2_sampling_loss", metrics["sampling_loss"],
                     on_step=True, on_epoch=True, batch_size=data.num_graphs)
            self.log("train/beta_h2_correction_loss", metrics["correction_loss"],
                     on_step=True, on_epoch=True, batch_size=data.num_graphs)
            for key in ("min_beta_parameter", "max_beta_parameter",
                        "nan_count", "inf_count", "valid_node_count",
                        "valid_edge_pair_count", "valid_element_count",
                        "invalid_parameter_count", "nan_kl_count",
                        "inf_kl_count", "tiny_negative_count",
                        "materially_negative_count", "zero_kl_count",
                        "positive_kl_count", "min_valid_kl",
                        "max_valid_kl", "mean_valid_kl"):
                self.log(f"diagnostics/{key}", metrics[key], on_step=True,
                         on_epoch=True, batch_size=data.num_graphs)
        return {"loss": loss, "metrics": metrics,
                "X_shape": tuple(X0.shape), "E_shape": tuple(E0.shape)}

    def training_step(self, data, batch_idx):
        return self._shared_step(data, "train")

    def validation_step(self, data, batch_idx):
        return self._shared_step(data, "val")["loss"]

    def test_step(self, data, batch_idx):
        return self._shared_step(data, "test")["loss"]

    def _sampling_condition(self, y: Optional[torch.Tensor], batch_size: int,
                            device: torch.device) -> torch.Tensor:
        if y is None:
            return fixed_h2_condition(self.ydim_output).to(device).expand(
                batch_size, -1)
        y = torch.as_tensor(y, device=device).float()
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if y.size(0) == 1 and batch_size > 1:
            y = y.expand(batch_size, -1)
        if y.shape != (batch_size, self.ydim_output):
            raise ValueError(
                f"Expected y shape {(batch_size, self.ydim_output)}, got {tuple(y.shape)}")
        return y

    @torch.no_grad()
    def sampling_batch_h2(self, batch_id: int, batch_size: int, keep_chain: int,
                          number_chain_steps: int, save_final: int = 1,
                          figures: bool = False, y: Optional[torch.Tensor] = None,
                          num_nodes=None, graph_number: int = 0):
        """Sample h2 graphs with the legacy HiGGs return structure."""
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device).long()
        elif isinstance(num_nodes, int):
            n_nodes = torch.full((batch_size,), num_nodes, device=self.device,
                                 dtype=torch.long)
        else:
            n_nodes = torch.as_tensor(num_nodes, device=self.device).long()
        n_max = int(n_nodes.max().item())
        node_mask = (torch.arange(n_max, device=self.device).unsqueeze(0)
                     < n_nodes.unsqueeze(1))
        condition_y = self._sampling_condition(y, batch_size, self.device)

        base_X0 = torch.full((batch_size, n_max, self.Xdim_output),
                             self.node_shift, device=self.device)
        base_E0 = torch.full((batch_size, n_max, n_max, self.beta_Edim),
                             self.edge_shift, device=self.device)
        base_X0, base_E0 = apply_graph_masks(base_X0, base_E0, node_mask)
        eta_X, eta_E = self._eta_tensors(
            torch.zeros_like(base_E0), node_mask)
        terminal_t = torch.ones((batch_size, 1), device=self.device)
        alpha_T = self._alpha(terminal_t)
        X_state, _ = forward_beta_sample(
            base_X0, alpha_T.view(-1, 1, 1), eta_X,
            self.beta_eps, self.input_space)
        E_state, _ = forward_beta_sample(
            base_E0, alpha_T.view(-1, 1, 1, 1), eta_E,
            self.beta_eps, self.input_space)
        X_state, E_state = apply_graph_masks(X_state, E_state, node_mask)

        times = torch.linspace(1.0, 0.0, self.T + 1, device=self.device)
        pred_X0, pred_E0 = base_X0, base_E0
        for step in range(self.T):
            t_value, s_value = times[step], times[step + 1]
            t = t_value.expand(batch_size, 1)
            alpha_t = self._alpha(t)
            alpha_s = self._alpha(s_value.expand(batch_size, 1))
            pred_X0, pred_E0 = self._predict_clean(
                X_state, E_state, condition_y, t, alpha_t,
                eta_X, eta_E, node_mask)

            if self.eta_strategy == "degree":
                predicted_edges = shifted_clean_to_unit_probability(
                    pred_E0, self.edge_scale, self.edge_shift)
                eta_X, eta_E = self._eta_tensors(predicted_edges, node_mask)
            X_state, _ = reverse_beta_step(
                X_state, pred_X0, alpha_t.view(-1, 1, 1),
                alpha_s.view(-1, 1, 1), eta_X, self.beta_eps,
                self.input_space)
            E_state, _ = reverse_beta_step(
                E_state, pred_E0, alpha_t.view(-1, 1, 1, 1),
                alpha_s.view(-1, 1, 1, 1), eta_E, self.beta_eps,
                self.input_space)
            X_state, E_state = apply_graph_masks(X_state, E_state, node_mask)

        node_values = shifted_clean_to_unit_probability(
            pred_X0, self.node_scale, self.node_shift)
        edge_values = shifted_clean_to_unit_probability(
            pred_E0, self.edge_scale, self.edge_shift)
        node_types, edge_types = discretize_graph(
            node_values, edge_values, node_mask, self.edge_threshold)

        predicted_graph_list = []
        networkx_graphs = []
        for index in range(batch_size):
            n = int(n_nodes[index].item())
            nodes = node_types[index, :n].cpu()
            edges = edge_types[index, :n, :n].cpu()
            graph = utils.to_networkx(nodes.numpy(), edges.numpy())
            predicted_graph_list.append([nodes, edges])
            networkx_graphs.append(graph)

        if figures and self.visualization_tools is not None:
            result_path = (Path.cwd() / "visualizations" /
                           f"sample_{graph_number:03d}" / "h2")
            os.makedirs(result_path, exist_ok=True)
            self.visualization_tools.visualize(
                result_path, predicted_graph_list,
                min(len(predicted_graph_list), int(save_final)),
                log="Sampled H2 Beta")

        return predicted_graph_list, networkx_graphs
