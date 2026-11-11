import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dgd"))

from diffusion.beta_diffusion_utils import (  # noqa: E402
    apply_graph_masks,
    shifted_clean_to_unit_probability,
    unit_probability_to_shifted_state,
    valid_upper_edge_mask,
)
from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1  # noqa: E402
from metrics.beta_train_metrics import BetaH1Loss  # noqa: E402


class _NodeDistribution:
    def sample_n(self, batch_size, device):
        return torch.full((batch_size,), 3, device=device, dtype=torch.long)


class _DatasetInfo:
    input_dims = {"X": 1, "E": 2, "y": 3}
    output_dims = {"X": 1, "E": 2, "y": 2}
    nodes_dist = _NodeDistribution()
    node_types = torch.tensor([1.0])
    edge_types = torch.tensor([0.9, 0.1])


def _cfg():
    return OmegaConf.create({
        "general": {"name": "test", "gpus": 0},
        "dataset": {"h": 1, "resolution": 1},
        "train": {"lr": 2.0e-4, "weight_decay": 1.0e-12},
        "model": {
            "family": "beta_h1",
            "type": "beta",
            "diffusion_steps": 2,
            "input_space": "logit",
            "pre_condition": False,
            "sigmoid_start": 10,
            "sigmoid_end": -13,
            "sigmoid_power": 1.0,
            "eta": {"node": 100, "edge": 100},
            "eta_strategy": "fixed",
            "scale_shift": {"node": [0.9, 0.09], "edge": [0.9, 0.09]},
            "beta_loss": {"sampling_weight": 0.99, "correction_weight": 0.01},
            "edge_threshold": 0.5,
            "beta_eps": 1.0e-6,
            "n_layers": 1,
            "hidden_mlp_dims": {"X": 16, "E": 16, "y": 16},
            "hidden_dims": {
                "dx": 16, "de": 16, "dy": 16, "n_head": 4,
                "dim_ffX": 16, "dim_ffE": 16, "dim_ffy": 16,
            },
            "lambda_train": [5, 0],
        },
    })


def test_unit_shifted_roundtrip_and_graph_masks():
    values = torch.tensor([0.01, 0.5, 0.99])
    shifted = unit_probability_to_shifted_state(values, 0.9, 0.09)
    restored = shifted_clean_to_unit_probability(shifted, 0.9, 0.09)
    assert torch.allclose(restored, values)

    node_mask = torch.tensor([[True, True, False]])
    X = torch.ones((1, 3, 1))
    E = torch.arange(9.0).reshape(1, 3, 3, 1)
    X, E = apply_graph_masks(X, E, node_mask)
    assert torch.equal(E, E.transpose(1, 2))
    assert torch.count_nonzero(torch.diagonal(E[0, :, :, 0])) == 0
    assert torch.count_nonzero(E[:, 2]) == 0
    assert torch.count_nonzero(E[:, :, 2]) == 0
    assert valid_upper_edge_mask(node_mask).sum().item() == 1


def test_single_class_h1_loss_is_edge_only_and_strict_upper():
    loss_module = BetaH1Loss(
        lambda_edge=5.0, sampling_weight=0.99,
        correction_weight=0.01, eps=1.0e-6)
    node_mask = torch.tensor([[True, True, True, False]])
    true_X = torch.full((1, 4, 1), 0.99)
    pred_X = torch.full((1, 4, 1), 0.25, requires_grad=True)
    true_E = torch.full((1, 4, 4, 1), 0.09)
    pred_E = torch.full((1, 4, 4, 1), 0.45, requires_grad=True)
    eta_X = torch.full_like(true_X, 100.0)
    eta_E = torch.full_like(true_E, 100.0)
    alpha_t = torch.tensor([[0.2]])
    alpha_s = torch.tensor([[0.3]])

    loss, metrics = loss_module(
        pred_X, pred_E, true_X, true_E, eta_X, eta_E,
        alpha_t, alpha_s, node_mask, diffuse_X=False)
    assert metrics["X_loss"].item() == 0.0
    assert metrics["valid_node_count"].item() == 0
    assert metrics["valid_edge_pair_count"].item() == 3
    assert torch.isfinite(loss)
    loss.backward()
    assert pred_E.grad is not None and torch.isfinite(pred_E.grad).all()
    assert pred_X.grad is not None
    assert torch.count_nonzero(pred_X.grad) == 0


def test_h1_sampling_schema_fixed_x_y_singleton_and_padding_safety():
    model = ConditionalGraphBetaDiffusionH1(
        _cfg(), _DatasetInfo(), extra_features=None, domain_features=None)
    model.eval()
    condition = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    original_condition = condition.clone()
    graphs, nx_graphs = model.sampling_batch_h1(
        batch_id=0, batch_size=2, keep_chain=0, number_chain_steps=1,
        num_nodes=torch.tensor([1, 3]), y=condition, figures=False)

    assert torch.equal(condition, original_condition)
    assert len(graphs) == len(nx_graphs) == 2
    for expected_nodes, graph in zip((1, 3), graphs):
        assert len(graph) == 2
        nodes, edges = graph
        assert nodes.shape == (expected_nodes,)
        assert edges.shape == (expected_nodes, expected_nodes)
        assert torch.count_nonzero(nodes) == 0
        assert torch.equal(edges, edges.T)
        assert torch.count_nonzero(torch.diagonal(edges)) == 0
        assert torch.isfinite(edges.float()).all()
    assert nx_graphs[0].number_of_nodes() == 1


def test_threshold_semantics_are_applied_in_unit_domain():
    model = ConditionalGraphBetaDiffusionH1(
        _cfg(), _DatasetInfo(), extra_features=None, domain_features=None)
    shifted = unit_probability_to_shifted_state(
        torch.tensor([0.49, 0.51]), model.edge_scale, model.edge_shift)
    unit = shifted_clean_to_unit_probability(
        shifted, model.edge_scale, model.edge_shift)
    assert torch.equal(
        (unit > model.edge_threshold).long(), torch.tensor([0, 1]))
