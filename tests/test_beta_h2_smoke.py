"""Lightweight train/sample/checkpoint smoke test for Conditional GBD h2."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import networkx as nx
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dgd"))

from diffusion_model_beta_h2 import ConditionalGraphBetaDiffusionH2  # noqa: E402
from diffusion.beta_diffusion_utils import discretize_graph  # noqa: E402


class FixedNodeDistribution:
    def __init__(self, n_nodes: int):
        self.n_nodes = int(n_nodes)

    def sample_n(self, batch_size: int, device):
        return torch.full((batch_size,), self.n_nodes, device=device,
                          dtype=torch.long)


class SmokeDatasetInfos:
    def __init__(self, node_classes: int, edge_classes: int,
                 y_classes: int, n_nodes: int):
        self.input_dims = {"X": node_classes, "E": edge_classes,
                           "y": y_classes + 1}
        self.output_dims = {"X": node_classes, "E": edge_classes,
                            "y": y_classes}
        self.nodes_dist = FixedNodeDistribution(n_nodes)
        self.node_types = torch.full((node_classes,), 1.0 / node_classes)
        self.edge_types = torch.full((edge_classes,), 1.0 / edge_classes)
        self.datamodule = None


def _synthetic_graph() -> Data:
    node_classes = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    undirected_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    row, col = [], []
    for source, target in undirected_edges:
        row.extend((source, target))
        col.extend((target, source))
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = F.one_hot(torch.ones(len(row), dtype=torch.long),
                          num_classes=2).float()
    return Data(
        x=F.one_hot(node_classes, num_classes=2).float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=F.one_hot(torch.tensor([0]), num_classes=2).float(),
    )


def _first_processed_graph(path: Path) -> Data:
    data, slices = torch.load(path)
    x_start, x_end = int(slices["x"][0]), int(slices["x"][1])
    e_start, e_end = int(slices["edge_index"][0]), int(slices["edge_index"][1])
    y_start, y_end = int(slices["y"][0]), int(slices["y"][1])
    return Data(
        x=data.x[x_start:x_end],
        edge_index=data.edge_index[:, e_start:e_end] - x_start,
        edge_attr=data.edge_attr[e_start:e_end],
        y=data.y[y_start:y_end],
    )


def _smoke_graph() -> Data:
    real_path = os.environ.get("HIGGS_SBM_H2_PROCESSED")
    if real_path and Path(real_path).is_file():
        graph = _first_processed_graph(Path(real_path))
        graph.source = "real_sbm_h2"
        return graph
    graph = _synthetic_graph()
    graph.source = "synthetic_fallback"
    return graph


def _config():
    return OmegaConf.create({
        "general": {"name": "beta_h2_smoke", "log_every_steps": 1,
                    "number_chain_steps": 4},
        "dataset": {"name": "sbm", "h": 2},
        "train": {"lr": 2.0e-4, "weight_decay": 1.0e-12},
        "model": {
            "family": "beta_h2", "type": "beta", "model": "graph_tf",
            "diffusion_steps": 5, "input_space": "logit",
            "pre_condition": False, "sigmoid_start": 10,
            "sigmoid_end": -13, "sigmoid_power": 1.0,
            "eta": {"node": 100, "edge": 100},
            "eta_strategy": "fixed",
            "degree_thresholds": [0.33, 0.66],
            "degree_eta_multipliers": [0.5, 1.0, 2.0],
            "scale_shift": {"node": [0.9, 0.09], "edge": [0.9, 0.09]},
            "beta_loss": {"sampling_weight": 0.99,
                          "correction_weight": 0.01},
            # Structural smoke only: threshold 0.5 is not a quality claim.
            "edge_threshold": 0.5, "beta_eps": 1.0e-6,
            "n_layers": 2, "extra_features": None,
            "hidden_mlp_dims": {"X": 32, "E": 32, "y": 32},
            "hidden_dims": {"dx": 32, "de": 32, "dy": 32,
                            "n_head": 4, "dim_ffX": 32,
                            "dim_ffE": 32, "dim_ffy": 32},
            "lambda_train": [5, 0],
        },
    })


def test_beta_h2_train_sample_and_checkpoint(tmp_path):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    graph = _smoke_graph()
    batch = Batch.from_data_list([graph])
    infos = SmokeDatasetInfos(
        node_classes=graph.x.size(-1), edge_classes=graph.edge_attr.size(-1),
        y_classes=graph.y.size(-1), n_nodes=graph.num_nodes)
    model = ConditionalGraphBetaDiffusionH2(_config(), infos)

    result = model.training_step(batch, 0)
    loss = result["loss"]
    assert torch.isfinite(loss)
    assert loss.dtype == torch.float64
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)

    predicted, nx_graphs = model.sampling_batch_h2(
        0, 1, 0, 4, 1, figures=False, num_nodes=graph.num_nodes)
    assert len(predicted) == len(nx_graphs) == 1
    node_types, adjacency = predicted[0]
    assert node_types.numel() > 0
    assert node_types.min() >= 0 and node_types.max() < graph.x.size(-1)
    assert torch.equal(adjacency, adjacency.t())
    assert torch.count_nonzero(torch.diagonal(adjacency)) == 0
    assert torch.isfinite(adjacency.float()).all()
    assert isinstance(nx_graphs[0], nx.Graph)
    assert not nx_graphs[0].is_multigraph()
    assert result["metrics"]["valid_node_count"] == graph.num_nodes
    assert result["metrics"]["valid_edge_pair_count"] == graph.num_nodes * (graph.num_nodes - 1) // 2
    assert result["metrics"]["nan_kl_count"] == 0
    assert result["metrics"]["inf_kl_count"] == 0
    assert result["metrics"]["materially_negative_count"] == 0

    checkpoint_path = tmp_path / "beta_h2_smoke.ckpt"
    torch.save({
        "state_dict": model.state_dict(),
        "hyper_parameters": dict(model.hparams),
        "pytorch-lightning_version": pl.__version__,
        "epoch": 0,
        "global_step": 1,
    }, checkpoint_path)
    reloaded = ConditionalGraphBetaDiffusionH2.load_from_checkpoint(checkpoint_path)
    predicted_reloaded, nx_reloaded = reloaded.sampling_batch_h2(
        0, 1, 0, 4, 1, figures=False, num_nodes=graph.num_nodes)
    assert len(predicted_reloaded) == len(nx_reloaded) == 1
    assert predicted_reloaded[0][0].numel() > 0

    print({
        "source": graph.source,
        "loss": float(loss.detach()),
        "X_shape": result["X_shape"],
        "E_shape": result["E_shape"],
        "sample_nodes": int(node_types.numel()),
        "sample_edges": int(torch.count_nonzero(adjacency).item() // 2),
        "symmetric": bool(torch.equal(adjacency, adjacency.t())),
        "checkpoint": str(checkpoint_path),
    })


def test_beta_h2_optional_modes():
    graph = _smoke_graph()
    batch = Batch.from_data_list([graph])
    infos = SmokeDatasetInfos(
        node_classes=graph.x.size(-1), edge_classes=graph.edge_attr.size(-1),
        y_classes=graph.y.size(-1), n_nodes=graph.num_nodes)
    for input_space, pre_condition, eta_strategy in (
            ("logit", True, "degree"),
            ("original", False, "fixed")):
        cfg = _config()
        cfg.model.input_space = input_space
        cfg.model.pre_condition = pre_condition
        cfg.model.eta_strategy = eta_strategy
        model = ConditionalGraphBetaDiffusionH2(cfg, infos)
        result = model.training_step(batch, 0)
        assert torch.isfinite(result["loss"])
        assert result["metrics"]["min_beta_parameter"] > 0
        assert result["metrics"]["nan_count"] == 0
        assert result["metrics"]["inf_count"] == 0
        sampled, _ = model.sampling_batch_h2(
            0, 1, 0, 4, 1, figures=False, num_nodes=graph.num_nodes)
        assert sampled[0][0].numel() > 0
        assert torch.equal(sampled[0][1], sampled[0][1].t())


def test_multiclass_edge_discretization():
    node_values = torch.tensor([[[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]])
    edge_values = torch.zeros((1, 3, 3, 3))
    edge_values[0, 0, 1, 2] = edge_values[0, 1, 0, 2] = 0.8
    edge_values[0, 1, 2, 0] = edge_values[0, 2, 1, 0] = 0.7
    node_types, edge_types = discretize_graph(
        node_values, edge_values, torch.tensor([[True, True, False]]), 0.5)
    assert node_types.tolist() == [[0, 1, -1]]
    assert edge_types[0, 0, 1].item() == 3
    assert torch.equal(edge_types, edge_types.transpose(1, 2))
    assert torch.count_nonzero(edge_types[0, 2]) == 0


if __name__ == "__main__":
    test_beta_h2_train_sample_and_checkpoint(ROOT / "tests" / ".tmp_beta_h2")
    test_beta_h2_optional_modes()
    test_multiclass_edge_discretization()
