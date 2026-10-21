"""Conditional Graph Beta Diffusion for HiGGs Stage Two only."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn

import utils
from diffusion.beta_diffusion_utils import (
    apply_graph_masks,
    categorical_precondition_stats,
    forward_beta_sample,
    inverse_scale_shift,
    logits_to_unit_probability,
    reverse_beta_step,
    shifted_clean_to_unit_probability,
    sigmoid_alpha,
    unit_probability_to_shifted_state,
)
from metrics.beta_train_metrics import BetaH1Loss
from models.transformer_model import GraphTransformer


class ConditionalGraphBetaDiffusionH1(pl.LightningModule):
    """Generate an H1 community graph while keeping its H2 class condition fixed."""

    family = "beta_h1"
    stage = "h1"

    def __init__(self, cfg, dataset_infos, train_metrics=None,
                 sampling_metrics=None, visualization_tools=None,
                 extra_features=None, domain_features=None):
        super().__init__()
        if float(cfg.dataset.h) != 1.0:
            raise ValueError("ConditionalGraphBetaDiffusionH1 requires dataset.h=1")
        if str(cfg.model.get("eta_strategy", "fixed")) != "fixed":
            raise ValueError(
                "beta_h1 supports eta_strategy=fixed only; "
                "degree/betweenness/dynamic eta is not supported")

        self.cfg = cfg
        self.name = cfg.general.name
        self.T = int(cfg.model.diffusion_steps)
        self.dataset_info = dataset_infos
        self.node_dist = dataset_infos.nodes_dist
        self.sampling_metrics = sampling_metrics
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        input_dims = dict(dataset_infos.input_dims)
        output_dims = dict(dataset_infos.output_dims)
        self.Xdim_output = int(output_dims["X"])
        self.Edim_output = int(output_dims["E"])
        self.ydim_output = int(output_dims["y"])
        if self.Edim_output != 2:
            raise ValueError(
                "Beta-h1 current implementation only supports binary HiGGs edges")

        # The public HiGGs representation has [no-edge, edge]. Beta diffusion
        # models only the edge-present channel and reconstructs no-edge later.
        input_dims["E"] -= 1
        output_dims["E"] -= 1
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
        self.edge_threshold = float(cfg.model.edge_threshold)
        self.beta_eps = float(cfg.model.beta_eps)
        if not 0.0 <= self.edge_threshold <= 1.0:
            raise ValueError("edge_threshold must be in [0, 1]")

        self.diffuse_X = self.Xdim_output > 1
        self.loss_module = BetaH1Loss(
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
        self.register_buffer(
            "node_channel_prior", torch.as_tensor(node_prior).float())
        self.register_buffer(
            "edge_channel_prior", torch.as_tensor(edge_prior).float()[1:])

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
                "model_family": self.family,
                "stage": self.stage,
            })
        except Exception as error:
            print(f"[HiGGs] beta_h1 hyperparameter serialization skipped: {error}")
        print("[HiGGs] Stage h1 model: Conditional Graph Beta Diffusion")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_family"] = self.family
        checkpoint["stage"] = self.stage

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
            weight_decay=self.cfg.train.weight_decay)

    def _alpha(self, t):
        return sigmoid_alpha(
            t, self.sigmoid_start, self.sigmoid_end, self.sigmoid_power)

    def _eta_tensors(self, batch_size, n_nodes, reference):
        eta_X = torch.full(
            (batch_size, n_nodes, 1), self.eta_node,
            device=reference.device, dtype=reference.dtype)
        eta_E = torch.full(
            (batch_size, n_nodes, n_nodes, 1), self.eta_edge,
            device=reference.device, dtype=reference.dtype)
        return eta_X.clamp_min(self.beta_eps), eta_E.clamp_min(self.beta_eps)

    def _state_to_unit(self, state, scale, shift):
        shifted = torch.sigmoid(state) if self.input_space == "logit" else state
        return inverse_scale_shift(shifted, scale, shift).clamp(0.0, 1.0)

    def _shifted_to_state(self, shifted):
        if self.input_space == "logit":
            return torch.logit(
                shifted.clamp(self.beta_eps, 1.0 - self.beta_eps))
        return shifted

    def _condition(self, y, batch_size, device):
        if y is None:
            raise ValueError("beta_h1 sampling requires an explicit fixed y condition")
        condition = torch.as_tensor(y, device=device).float()
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.size(0) == 1 and batch_size > 1:
            condition = condition.expand(batch_size, -1)
        expected = (batch_size, self.ydim_output)
        if tuple(condition.shape) != expected:
            raise ValueError(
                f"Expected y shape {expected}, got {tuple(condition.shape)}")
        if not torch.isfinite(condition).all():
            raise ValueError("y condition contains NaN or Inf")
        return condition

    def _precondition(self, X_t, E_t, alpha_t, eta_X, eta_E, node_mask):
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
        return apply_graph_masks(
            (X_t - mean_X) / std_X.clamp_min(self.beta_eps),
            (E_t - mean_E) / std_E.clamp_min(self.beta_eps), node_mask)

    def _extra_data(self, X_t, E_t, condition_y, t, node_mask):
        public_X = self._state_to_unit(
            X_t, self.node_scale, self.node_shift) * node_mask.unsqueeze(-1)
        edge_present = self._state_to_unit(
            E_t, self.edge_scale, self.edge_shift)
        structural_edge = (
            edge_present > self.edge_threshold).type_as(edge_present)
        public_E = torch.cat((1.0 - structural_edge, structural_edge), dim=-1)
        valid_pair = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        diagonal = torch.eye(
            public_E.size(1), dtype=torch.bool,
            device=public_E.device).unsqueeze(0)
        public_E = public_E * (valid_pair & ~diagonal).unsqueeze(-1)
        noisy_data = {
            "X_t": public_X,
            "E_t": public_E,
            "y_t": condition_y,
            "node_mask": node_mask,
            "t": t,
        }
        if self.extra_features is None:
            extra_X = X_t.new_zeros((*X_t.shape[:-1], 0))
            extra_E = E_t.new_zeros((*E_t.shape[:-1], 0))
            extra_y = condition_y.new_zeros((condition_y.size(0), 0))
        else:
            extra = self.extra_features(noisy_data)
            extra_X, extra_E, extra_y = extra.X, extra.E, extra.y
        if self.domain_features is not None:
            domain = self.domain_features(noisy_data)
            extra_X = torch.cat((extra_X, domain.X), dim=-1)
            extra_E = torch.cat((extra_E, domain.E), dim=-1)
            extra_y = torch.cat((extra_y, domain.y), dim=-1)
        return extra_X, extra_E, extra_y

    def _predict_clean(self, X_t, E_t, condition_y, t, alpha_t,
                       eta_X, eta_E, node_mask, fixed_X=None):
        if (self.Xdim_output == 1 and X_t.size(1) == 1
                and fixed_X is not None):
            # There is no trainable strict-upper edge in a singleton graph.
            anchor = next(self.model.parameters()).sum() * 0.0
            return fixed_X + anchor, torch.zeros_like(E_t) + anchor
        model_X, model_E = self._precondition(
            X_t, E_t, alpha_t, eta_X, eta_E, node_mask)
        extra_X, extra_E, extra_y = self._extra_data(
            X_t, E_t, condition_y, t, node_mask)
        prediction = self.model(
            torch.cat((model_X, extra_X), dim=-1).float(),
            torch.cat((model_E, extra_E), dim=-1).float(),
            torch.cat((condition_y.float(), extra_y.float(), t.float()), dim=-1),
            node_mask,
        )
        pred_X0 = unit_probability_to_shifted_state(
            logits_to_unit_probability(prediction.X),
            self.node_scale, self.node_shift)
        pred_E0 = unit_probability_to_shifted_state(
            logits_to_unit_probability(prediction.E),
            self.edge_scale, self.edge_shift)
        pred_X0, pred_E0 = apply_graph_masks(pred_X0, pred_E0, node_mask)
        if fixed_X is not None:
            pred_X0 = fixed_X
        return pred_X0, pred_E0

    @staticmethod
    def _validate_diagnostics(diagnostics, context):
        if diagnostics is None:
            return
        if int((diagnostics.nan_count + diagnostics.inf_count).detach().cpu()):
            raise FloatingPointError(f"{context}: non-finite Beta parameters")

    def _shared_step(self, data, stage):
        dense, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch)
        dense = dense.mask(node_mask)
        X_discrete, E_discrete = dense.X.float(), dense.E.float()
        if E_discrete.size(-1) != 2:
            raise ValueError(
                "Beta-h1 current implementation only supports binary HiGGs edges")
        condition_y = self._condition(
            data.y, X_discrete.size(0), X_discrete.device)
        X0 = unit_probability_to_shifted_state(
            X_discrete, self.node_scale, self.node_shift)
        E0 = unit_probability_to_shifted_state(
            E_discrete[..., 1:], self.edge_scale, self.edge_shift)
        X0, E0 = apply_graph_masks(X0, E0, node_mask)
        eta_X, eta_E = self._eta_tensors(X0.size(0), X0.size(1), X0)
        t = torch.rand((X0.size(0), 1), device=X0.device)
        t = t * (1.0 - 1.0 / self.T) + 1.0 / self.T
        s = (t - 1.0 / self.T).clamp_min(0.0)
        alpha_t, alpha_s = self._alpha(t), self._alpha(s)

        X_diag = None
        if self.diffuse_X:
            X_t, X_diag = forward_beta_sample(
                X0, alpha_t.view(-1, 1, 1), eta_X,
                self.beta_eps, self.input_space)
        else:
            X_t = self._shifted_to_state(X0)
        E_t, E_diag = forward_beta_sample(
            E0, alpha_t.view(-1, 1, 1, 1), eta_E,
            self.beta_eps, self.input_space)
        self._validate_diagnostics(X_diag, "beta_h1/X forward")
        self._validate_diagnostics(E_diag, "beta_h1/E forward")
        X_t, E_t = apply_graph_masks(X_t, E_t, node_mask)
        pred_X0, pred_E0 = self._predict_clean(
            X_t, E_t, condition_y, t, alpha_t, eta_X, eta_E,
            node_mask, fixed_X=None if self.diffuse_X else X0)
        loss, metrics = self.loss_module(
            pred_X0, pred_E0, X0, E0, eta_X, eta_E,
            alpha_t, alpha_s, node_mask, self.diffuse_X)
        gamma_count = E_diag.gamma_clamp_count
        if X_diag is not None:
            gamma_count = gamma_count + X_diag.gamma_clamp_count

        prefix = "train" if stage == "train" else stage
        self.log(f"{prefix}/h1_X_KLUB", metrics["X_loss"],
                 on_step=stage == "train", on_epoch=True,
                 batch_size=data.num_graphs)
        self.log(f"{prefix}/h1_E_KLUB", metrics["E_loss"],
                 on_step=stage == "train", on_epoch=True,
                 batch_size=data.num_graphs)
        self.log(f"{prefix}/h1_total", loss,
                 on_step=stage == "train", on_epoch=True,
                 batch_size=data.num_graphs)
        if stage == "train":
            diagnostics = {
                "beta_param_min_valid": metrics["min_beta_parameter"],
                "beta_param_max_valid": metrics["max_beta_parameter"],
                "beta_kl_negative_count": metrics["beta_kl_negative_count"],
                "beta_kl_nonfinite_count": metrics["beta_kl_nonfinite_count"],
                "gamma_clamp_count": gamma_count,
            }
            for name, value in diagnostics.items():
                self.log(f"diagnostics/{name}", value.float(),
                         on_step=True, on_epoch=True,
                         batch_size=data.num_graphs)
        return {"loss": loss, "metrics": metrics,
                "condition_y": condition_y.detach()}

    def training_step(self, data, batch_idx):
        return self._shared_step(data, "train")

    def validation_step(self, data, batch_idx):
        return self._shared_step(data, "val")["loss"]

    def test_step(self, data, batch_idx):
        return self._shared_step(data, "test")["loss"]

    @torch.no_grad()
    def sampling_batch_h1(self, batch_id, batch_size, keep_chain,
                          number_chain_steps, num_nodes=None, y=None,
                          figures=False):
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device).long()
        elif isinstance(num_nodes, int):
            n_nodes = torch.full(
                (batch_size,), num_nodes, device=self.device, dtype=torch.long)
        else:
            n_nodes = torch.as_tensor(
                num_nodes, device=self.device, dtype=torch.long)
        n_max = int(n_nodes.max().item())
        node_mask = (
            torch.arange(n_max, device=self.device).unsqueeze(0)
            < n_nodes.unsqueeze(1))
        condition_y = self._condition(y, batch_size, self.device)
        original_condition = condition_y.clone()

        base_X0 = torch.full(
            (batch_size, n_max, self.Xdim_output), self.node_shift,
            device=self.device)
        if not self.diffuse_X:
            base_X0[node_mask] = self.node_shift + self.node_scale
        base_E0 = torch.full(
            (batch_size, n_max, n_max, 1), self.edge_shift,
            device=self.device)
        base_X0, base_E0 = apply_graph_masks(base_X0, base_E0, node_mask)
        eta_X, eta_E = self._eta_tensors(batch_size, n_max, base_X0)
        alpha_T = self._alpha(
            torch.ones((batch_size, 1), device=self.device))
        if self.diffuse_X:
            X_state, X_diag = forward_beta_sample(
                base_X0, alpha_T.view(-1, 1, 1), eta_X,
                self.beta_eps, self.input_space)
            self._validate_diagnostics(X_diag, "beta_h1/X terminal")
        else:
            X_state = self._shifted_to_state(base_X0)
        E_state, E_diag = forward_beta_sample(
            base_E0, alpha_T.view(-1, 1, 1, 1), eta_E,
            self.beta_eps, self.input_space)
        self._validate_diagnostics(E_diag, "beta_h1/E terminal")
        X_state, E_state = apply_graph_masks(X_state, E_state, node_mask)

        times = torch.linspace(1.0, 0.0, self.T + 1, device=self.device)
        pred_X0, pred_E0 = base_X0, base_E0
        for step in range(self.T):
            t = times[step].expand(batch_size, 1)
            alpha_t = self._alpha(t)
            alpha_s = self._alpha(times[step + 1].expand(batch_size, 1))
            pred_X0, pred_E0 = self._predict_clean(
                X_state, E_state, condition_y, t, alpha_t,
                eta_X, eta_E, node_mask,
                fixed_X=None if self.diffuse_X else base_X0)
            if self.diffuse_X:
                X_state, X_diag = reverse_beta_step(
                    X_state, pred_X0, alpha_t.view(-1, 1, 1),
                    alpha_s.view(-1, 1, 1), eta_X, self.beta_eps,
                    self.input_space)
                self._validate_diagnostics(X_diag, "beta_h1/X reverse")
            E_state, E_diag = reverse_beta_step(
                E_state, pred_E0, alpha_t.view(-1, 1, 1, 1),
                alpha_s.view(-1, 1, 1, 1), eta_E, self.beta_eps,
                self.input_space)
            self._validate_diagnostics(E_diag, "beta_h1/E reverse")
            X_state, E_state = apply_graph_masks(
                X_state, E_state, node_mask)
            if not torch.equal(condition_y, original_condition):
                raise AssertionError("Beta-h1 y changed during reverse sampling")

        if self.diffuse_X:
            node_values = shifted_clean_to_unit_probability(
                pred_X0, self.node_scale, self.node_shift)
            node_types = node_values.argmax(dim=-1)
        else:
            node_types = torch.zeros(
                (batch_size, n_max), dtype=torch.long, device=self.device)
        edge_values = shifted_clean_to_unit_probability(
            pred_E0, self.edge_scale, self.edge_shift).squeeze(-1)
        edge_types = (edge_values > self.edge_threshold).long()
        edge_types = torch.triu(edge_types, diagonal=1)
        edge_types = edge_types + edge_types.transpose(1, 2)
        edge_types = edge_types.masked_fill(
            ~(node_mask.unsqueeze(1) & node_mask.unsqueeze(2)), 0)

        graphs, networkx_graphs = [], []
        for index in range(batch_size):
            n = int(n_nodes[index].item())
            nodes = node_types[index, :n].cpu()
            edges = edge_types[index, :n, :n].cpu()
            graphs.append([nodes, edges])
            networkx_graphs.append(
                utils.to_networkx(nodes.numpy(), edges.numpy()))
        self._visualize(batch_id, graphs, figures, condition_y)
        return graphs, networkx_graphs

    def _visualize(self, batch_id, graphs, figures, y):
        if not figures or self.visualization_tools is None:
            return
        result_path = (
            Path.cwd() / "graphs" / self.name /
            f"epoch{self.current_epoch}_b{batch_id}")
        os.makedirs(result_path, exist_ok=True)
        labels = y.argmax(dim=-1).cpu().tolist()
        self.visualization_tools.visualize_grid(
            result_path, graphs, min(len(graphs), 15),
            log="Sampled Grid H1 Beta", labels=labels,
            largest_component=False)
