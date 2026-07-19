"""Independent, read-only audit harness for the HiGGs beta-h2 implementation.

This file intentionally does not patch or rewrite production code.  The only
runtime monkeypatch wraps ``discretize_graph`` to capture the continuous model
output immediately before thresholding, and is restored before exit.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / ".tmp" / "independent_beta_h2_audit"
TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ["TEMP"] = str(TMP)
os.environ["TMP"] = str(TMP)
sys.path.insert(0, str(ROOT / "dgd"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402

import diffusion_model_beta_h2 as beta_model_module  # noqa: E402
from diffusion.beta_diffusion_utils import (  # noqa: E402
    apply_graph_masks,
    beta_kl,
    categorical_precondition_stats,
    fixed_h2_condition,
    forward_beta_sample,
    reverse_beta_step,
    scale_shift,
    positive_gamma_sample,
)
from diffusion_model_beta_h2 import ConditionalGraphBetaDiffusionH2  # noqa: E402


CHECKPOINT = (ROOT / "outputs" / "2026-07-17" / "11-25-47" /
              "checkpoints" / "beta_h2_fixed_kl_audit_epoch" / "last.ckpt")
REAL_DATA = Path(
    r"E:\HiGGs-main1\HiGGs-main\data\sbm\processed\proc_train_h2_r2_majority_max500_bs8.pt"
)
REAL_VALIDATION_DATA = Path(
    r"E:\HiGGs-main1\HiGGs-main\data\sbm\processed\proc_val_h2_r2_majority_max500_bs8.pt"
)
OUTPUT = ROOT / "tests" / "audit_beta_h2_results.json"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "q05": float(np.quantile(a, 0.05)),
        "q25": float(np.quantile(a, 0.25)),
        "median": float(np.quantile(a, 0.50)),
        "q75": float(np.quantile(a, 0.75)),
        "q95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def graph_summary(graphs):
    nodes = [g.number_of_nodes() for g in graphs]
    edges = [g.number_of_edges() for g in graphs]
    densities = [nx.density(g) for g in graphs]
    clustering = [nx.average_clustering(g) if g.number_of_nodes() else 0.0 for g in graphs]
    degrees = [degree for g in graphs for _, degree in g.degree()]
    isolated = sum(nx.number_of_isolates(g) for g in graphs)
    total_nodes = sum(nodes)
    connected = [g.number_of_nodes() > 0 and nx.is_connected(g) for g in graphs]
    complete = [n > 1 and m == n * (n - 1) // 2 for n, m in zip(nodes, edges)]
    return {
        "n_graphs": len(graphs),
        "nodes": summary(nodes),
        "edges": summary(edges),
        "density": summary(densities),
        "degree": summary(degrees),
        "clustering": summary(clustering),
        "connected_fraction": float(np.mean(connected)),
        "complete_fraction": float(np.mean(complete)),
        "edgeless_fraction": float(np.mean(np.asarray(edges) == 0)),
        "single_node_fraction": float(np.mean(np.asarray(nodes) == 1)),
        "isolated_node_fraction": float(isolated / max(total_nodes, 1)),
    }


def load_real_graphs(path=REAL_DATA):
    data, slices = torch.load(path, map_location="cpu")
    graphs = []
    ys = []
    for index in range(len(slices["x"]) - 1):
        x0, x1 = int(slices["x"][index]), int(slices["x"][index + 1])
        e0, e1 = int(slices["edge_index"][index]), int(slices["edge_index"][index + 1])
        n = x1 - x0
        edge_index = data.edge_index[:, e0:e1].cpu().numpy()
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        graphs.append(graph)
        y0, y1 = int(slices["y"][index]), int(slices["y"][index + 1])
        ys.append(data.y[y0:y1].reshape(-1).tolist())
    return graphs, ys


def synthetic_graph(n, edges, y_class):
    node_types = torch.arange(n) % 2
    x = F.one_hot(node_types, num_classes=2).float()
    directed = [(u, v) for u, v in edges for (u, v) in ((u, v), (v, u))]
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    edge_type = torch.ones(edge_index.size(1), dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=2).float()
    y = F.one_hot(torch.tensor([y_class]), num_classes=2).float()
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def mixed_batch_checks(model, device):
    graphs = [
        synthetic_graph(4, [(0, 1), (1, 2), (2, 3)], 0),
        synthetic_graph(7, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)], 1),
    ]
    batch = Batch.from_data_list(graphs).to(device)
    seen = {}
    original_noise = model._forward_noise
    original_loss = model.loss_module.forward

    def wrapped_noise(X0, E0, alpha_t, eta_X, eta_E, node_mask):
        seen["noise_eta_X_ptr"] = eta_X.data_ptr()
        seen["noise_eta_E_ptr"] = eta_E.data_ptr()
        seen["noise_eta_X"] = eta_X.detach().clone()
        seen["noise_eta_E"] = eta_E.detach().clone()
        return original_noise(X0, E0, alpha_t, eta_X, eta_E, node_mask)

    def wrapped_loss(pred_X, pred_E, true_X, true_E, eta_X, eta_E,
                     alpha_t, alpha_s, node_mask):
        seen["loss_eta_X_ptr"] = eta_X.data_ptr()
        seen["loss_eta_E_ptr"] = eta_E.data_ptr()
        seen["loss_eta_X"] = eta_X.detach().clone()
        seen["loss_eta_E"] = eta_E.detach().clone()
        seen["node_mask"] = node_mask.detach().clone()
        return original_loss(pred_X, pred_E, true_X, true_E, eta_X, eta_E,
                             alpha_t, alpha_s, node_mask)

    model._forward_noise = wrapped_noise
    model.loss_module.forward = wrapped_loss
    model.train()
    model.zero_grad(set_to_none=True)
    result = model._shared_step(batch, "audit")
    loss = result["loss"]
    loss.backward()
    finite_gradients = all(
        p.grad is None or torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )
    model._forward_noise = original_noise
    model.loss_module.forward = original_loss
    mask = seen["node_mask"]
    upper_pair_count = int(sum(int(n) * (int(n) - 1) // 2 for n in mask.sum(1)))
    return {
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "gradients_finite": bool(finite_gradients),
        "X_shape": list(result["X_shape"]),
        "E_shape": list(result["E_shape"]),
        "node_counts": mask.sum(1).cpu().tolist(),
        "valid_upper_edge_pairs": upper_pair_count,
        "eta_same_storage_forward_to_loss": bool(
            seen["noise_eta_X_ptr"] == seen["loss_eta_X_ptr"] and
            seen["noise_eta_E_ptr"] == seen["loss_eta_E_ptr"]
        ),
        "eta_same_values_forward_to_loss": bool(
            torch.equal(seen["noise_eta_X"], seen["loss_eta_X"]) and
            torch.equal(seen["noise_eta_E"], seen["loss_eta_E"])
        ),
        "metrics": {key: float(value.detach().cpu())
                    for key, value in result["metrics"].items()},
    }


def mode_checks(model, device):
    results = []
    base_cfg = copy.deepcopy(model.cfg)
    infos = model.dataset_info
    batch = Batch.from_data_list([
        synthetic_graph(4, [(0, 1), (1, 2), (2, 3)], 0),
        synthetic_graph(7, [(0, 1), (1, 2), (2, 4), (4, 6), (0, 6)], 1),
    ]).to(device)
    modes = [
        ("logit", False, "fixed"),
        ("original", False, "fixed"),
        ("logit", True, "degree"),
    ]
    for input_space, pre_condition, eta_strategy in modes:
        cfg = copy.deepcopy(base_cfg)
        cfg.model.input_space = input_space
        cfg.model.pre_condition = pre_condition
        cfg.model.eta_strategy = eta_strategy
        candidate = ConditionalGraphBetaDiffusionH2(cfg, infos).to(device)
        candidate.train()
        candidate.zero_grad(set_to_none=True)
        seed_all(177)
        out = candidate._shared_step(batch, "audit")
        out["loss"].backward()
        results.append({
            "input_space": input_space,
            "pre_condition": pre_condition,
            "eta_strategy": eta_strategy,
            "loss": float(out["loss"].detach().cpu()),
            "finite": bool(torch.isfinite(out["loss"]).item()),
            "nan_count": float(out["metrics"]["nan_count"].detach().cpu()),
            "inf_count": float(out["metrics"]["inf_count"].detach().cpu()),
            "min_beta_parameter": float(out["metrics"]["min_beta_parameter"].detach().cpu()),
            "max_beta_parameter": float(out["metrics"]["max_beta_parameter"].detach().cpu()),
        })
        del candidate
    return results


def numerical_checks(model, device):
    t = torch.tensor([[0.0], [1.0]], device=device)
    alpha = model._alpha(t)
    clean = torch.tensor([model.node_shift,
                          model.node_shift + model.node_scale], device=device).view(2, 1, 1)
    eta = torch.full_like(clean, model.eta_node)
    seed_all(11)
    original, forward_diag = forward_beta_sample(
        clean, alpha.view(2, 1, 1), eta, model.beta_eps, "original")
    logit, _ = forward_beta_sample(
        clean, alpha.view(2, 1, 1), eta, model.beta_eps, "logit")
    current = torch.full_like(clean, -4.0)
    pred = clean
    reverse_logit, reverse_diag = reverse_beta_step(
        current, pred,
        torch.full_like(clean, float(model._alpha(torch.tensor([[1.0]], device=device)))),
        torch.full_like(clean, float(model._alpha(torch.tensor([[0.8]], device=device)))),
        eta, model.beta_eps, "logit")
    probability = model.node_channel_prior
    mean, std = categorical_precondition_stats(
        alpha.view(2, 1, 1),
        torch.full((2, 1, 1), model.eta_node, device=device),
        probability, model.node_shift, model.node_shift + model.node_scale,
        "logit", model.beta_eps)
    formal_times = torch.linspace(1.0, 0.0, 101, device=device)
    formal_alpha = model._alpha(formal_times)
    deltas = formal_alpha[1:] - formal_alpha[:-1]
    low, high = model.node_shift, model.node_shift + model.node_scale
    ranges = []
    for index in list(range(3)) + list(range(97, 100)):
        a_min = model.eta_node * float(deltas[index]) * low
        a_max = model.eta_node * float(deltas[index]) * high
        b_min = model.eta_node * (1.0 - float(formal_alpha[index + 1]) * high)
        b_max = model.eta_node * (1.0 - float(formal_alpha[index + 1]) * low)
        ranges.append({"step": index, "alpha_t": float(formal_alpha[index]),
                       "alpha_s": float(formal_alpha[index + 1]),
                       "a_min": a_min, "a_max": a_max,
                       "b_min": b_min, "b_max": b_max})
    return {
        "alpha_t0": float(alpha[0]),
        "alpha_t1": float(alpha[1]),
        "forward_original_finite": bool(torch.isfinite(original).all().item()),
        "forward_logit_finite": bool(torch.isfinite(logit).all().item()),
        "forward_parameter_min": float(forward_diag.min_parameter),
        "forward_parameter_max": float(forward_diag.max_parameter),
        "reverse_logit_finite": bool(torch.isfinite(reverse_logit).all().item()),
        "reverse_parameter_min": float(reverse_diag.min_parameter),
        "reverse_parameter_max": float(reverse_diag.max_parameter),
        "precondition_mean_finite": bool(torch.isfinite(mean).all().item()),
        "precondition_std_finite": bool(torch.isfinite(std).all().item()),
        "precondition_std_min": float(std.min()),
        "formal_T100_reverse_parameter_ranges": ranges,
    }


def klub_precision_checks():
    """Compare production float32 KLUB arithmetic with a float64 Beta-KL baseline."""
    seed_all(20260717)
    eta = 100.0
    low, high = 0.09, 0.99
    alpha = torch.sigmoid(torch.linspace(-13.0, 10.0, 101))
    rows = []
    for step in (0, 10, 50, 99):
        delta = alpha[step + 1] - alpha[step]
        alpha_s = alpha[step + 1]
        for noise_scale in (1.0e-2, 1.0e-3):
            count = 50000
            target = torch.where(torch.rand(count) > 0.5,
                                 torch.tensor(high), torch.tensor(low))
            predicted = (target + noise_scale * torch.randn(count)).clamp(low, high)
            a_p = eta * delta * predicted
            b_p = eta * (1.0 - alpha_s * predicted)
            a_q = eta * delta * target
            b_q = eta * (1.0 - alpha_s * target)
            kl32 = beta_kl(a_p, b_p, a_q, b_q, 1.0e-6).double()
            ap, bp, aq, bq = (v.double() for v in (a_p, b_p, a_q, b_q))
            kl64 = (
                (torch.lgamma(aq) + torch.lgamma(bq) - torch.lgamma(aq + bq))
                - (torch.lgamma(ap) + torch.lgamma(bp) - torch.lgamma(ap + bp))
                + (ap - aq) * (torch.digamma(ap) - torch.digamma(ap + bp))
                + (bp - bq) * (torch.digamma(bp) - torch.digamma(ap + bp))
            )
            positive = kl64 > 1.0e-12
            rows.append({
                "step": step,
                "noise_scale": noise_scale,
                "mean_kl_float64": float(kl64.mean()),
                "mean_kl_production_float32": float(kl32.mean()),
                "positive_kl_zeroed_fraction": float(
                    ((kl32 == 0) & positive).sum() / positive.sum().clamp_min(1)),
            })

    gamma_rows = []
    for shape in (5.3e-6, 2.0e-5, 2.24e-4):
        samples = positive_gamma_sample(torch.full((200000,), shape))
        gamma_rows.append({
            "shape": shape,
            "fraction_exactly_legacy_beta_eps": float(
                (samples == 1.0e-6).float().mean()),
            "minimum": float(samples.min()),
            "log_finite": bool(torch.isfinite(samples.log()).all()),
        })
    return {"float32_vs_float64": rows, "gamma_floor": gamma_rows}


def capture_sample(model, threshold, seed, batch_size, y=None, num_nodes=None):
    captured = []
    original = beta_model_module.discretize_graph

    def wrapper(node_values, edge_values, node_mask, edge_threshold):
        node_types, edge_types = original(
            node_values, edge_values, node_mask, edge_threshold)
        captured.append({
            "node_values": node_values.detach().cpu(),
            "edge_values": edge_values.detach().cpu(),
            "node_mask": node_mask.detach().cpu(),
            "node_types": node_types.detach().cpu(),
            "edge_types": edge_types.detach().cpu(),
        })
        return node_types, edge_types

    old_threshold = model.edge_threshold
    beta_model_module.discretize_graph = wrapper
    model.edge_threshold = float(threshold)
    seed_all(seed)
    try:
        predicted, output_graphs = model.sampling_batch_h2(
            batch_id=seed, batch_size=batch_size, keep_chain=0,
            number_chain_steps=model.T, save_final=0, figures=False,
            y=y, num_nodes=num_nodes, graph_number=seed)
    finally:
        model.edge_threshold = old_threshold
        beta_model_module.discretize_graph = original
    assert len(captured) == 1
    cap = captured[0]
    raw_graphs = []
    continuous_edges = []
    for index in range(batch_size):
        n = int(cap["node_mask"][index].sum())
        adjacency = cap["edge_types"][index, :n, :n].numpy()
        graph = nx.from_numpy_array((adjacency > 0).astype(np.int8))
        raw_graphs.append(graph)
        upper = torch.triu(torch.ones((n, n), dtype=torch.bool), diagonal=1)
        values = cap["edge_values"][index, :n, :n].max(dim=-1).values
        continuous_edges.extend(values[upper].tolist())
    return {
        "raw_graphs": raw_graphs,
        "output_graphs": output_graphs,
        "continuous_edges": continuous_edges,
        "capture": cap,
        "predicted": predicted,
    }


def sampling_audit(model):
    threshold_results = {}
    for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        raw_graphs, output_graphs, continuous = [], [], []
        per_seed = []
        invariant_checks = []
        for seed in range(5):
            sampled = capture_sample(model, threshold, seed, 4)
            raw_graphs.extend(sampled["raw_graphs"])
            output_graphs.extend(sampled["output_graphs"])
            continuous.extend(sampled["continuous_edges"])
            cap = sampled["capture"]
            edge_types = cap["edge_types"]
            node_types = cap["node_types"]
            mask = cap["node_mask"]
            valid_edges = mask.unsqueeze(1) & mask.unsqueeze(2)
            diagonal = torch.diagonal(edge_types, dim1=1, dim2=2)
            invariant_checks.append({
                "symmetric": bool(torch.equal(edge_types, edge_types.transpose(1, 2))),
                "diagonal_zero": bool((diagonal == 0).all()),
                "padding_edges_zero": bool((edge_types.masked_select(~valid_edges) == 0).all()),
                "padding_nodes_minus_one": bool((node_types.masked_select(~mask) == -1).all()),
                "valid_node_classes_in_range": bool(
                    ((node_types.masked_select(mask) >= 0) &
                     (node_types.masked_select(mask) < model.Xdim_output)).all()),
                "continuous_finite": bool(torch.isfinite(cap["edge_values"]).all()),
            })
            per_seed.append({
                "seed": seed,
                "raw_density_mean": float(np.mean([nx.density(g) for g in sampled["raw_graphs"]])),
                "output_density_mean": float(np.mean([nx.density(g) for g in sampled["output_graphs"]])),
            })
        threshold_results[str(threshold)] = {
            "raw_before_lcc": graph_summary(raw_graphs),
            "public_output_after_lcc": graph_summary(output_graphs),
            "continuous_edge_score": summary(continuous),
            "per_seed": per_seed,
            "all_tensor_invariants_pass": bool(all(
                all(check.values()) for check in invariant_checks)),
            "tensor_invariants_by_seed": invariant_checks,
        }

    fixed_nodes = torch.full((20,), 8, device=model.device, dtype=torch.long)
    conditions = {
        "fixed_production_default": None,
        "onehot_0": torch.tensor([[1.0, 0.0]], device=model.device).expand(20, -1),
        "onehot_1": torch.tensor([[0.0, 1.0]], device=model.device).expand(20, -1),
    }
    condition_samples = {
        name: capture_sample(model, 0.5, 31415, 20, y=value, num_nodes=fixed_nodes)
        for name, value in conditions.items()
    }
    condition_results = {}
    for name, sampled in condition_samples.items():
        condition_results[name] = {
            "raw": graph_summary(sampled["raw_graphs"]),
            "continuous_edge_score": summary(sampled["continuous_edges"]),
        }
    c0 = condition_samples["onehot_0"]["capture"]["edge_values"]
    c1 = condition_samples["onehot_1"]["capture"]["edge_values"]
    cz = condition_samples["fixed_production_default"]["capture"]["edge_values"]
    e0 = condition_samples["onehot_0"]["capture"]["edge_types"]
    e1 = condition_samples["onehot_1"]["capture"]["edge_types"]
    mask = condition_samples["onehot_0"]["capture"]["node_mask"]
    valid = mask.unsqueeze(1) & mask.unsqueeze(2)
    upper = torch.triu(torch.ones_like(valid), diagonal=1).bool() & valid
    condition_results["paired_effect_same_noise"] = {
        "onehot0_vs_onehot1_edge_score_mae": float((c0 - c1).abs()[upper].mean()),
        "default_vs_onehot0_edge_score_mae": float((cz - c0).abs()[upper].mean()),
        "onehot0_vs_onehot1_discrete_pair_change_fraction": float((e0 != e1)[upper].float().mean()),
    }
    return threshold_results, condition_results


def checkpoint_routing_checks():
    """Exercise the actual HiGGs stage-2 loader for new and legacy checkpoints."""
    from omegaconf import OmegaConf

    sys.path.insert(0, str(ROOT / "higgs"))
    from higgs_utils import get_samplers

    legacy_checkpoint = Path(
        r"E:\HiGGs-main1\HiGGs-main\saved_models\sbm\sbm_h2.ckpt"
    )
    cases = {
        "legacy_missing_family": {
            "checkpoint": legacy_checkpoint,
            "stage": 2,
            "checkpoint_key": "h2_model",
            "general": {},
        },
        "beta_explicit_family": {
            "checkpoint": CHECKPOINT,
            "stage": 2,
            "checkpoint_key": "h2_model",
            "general": {"h2_model_family": "beta_h2"},
        },
        "h1_discrete": {
            "checkpoint": Path(
                r"E:\HiGGs-main1\HiGGs-main\saved_models\sbm\sbm_h1.ckpt"),
            "stage": 1,
            "checkpoint_key": "h1_model",
            "general": {},
        },
        "glue_discrete": {
            "checkpoint": Path(
                r"E:\HiGGs-main1\HiGGs-main\saved_models\sbm\sbm_glue.ckpt"),
            "stage": 1.5,
            "checkpoint_key": "glue_model",
            "general": {},
        },
    }
    results = {}
    for name, case in cases.items():
        cfg = OmegaConf.create({
            "general": {
                "name": f"audit_{name}",
                case["checkpoint_key"]: str(case["checkpoint"]),
                **case["general"],
            },
            "dataset": {"h": case["stage"]},
            "train": {},
            "model": {},
        })
        loaded_cfg, loaded_model = get_samplers(cfg)
        results[name] = {
            "checkpoint_exists": case["checkpoint"].exists(),
            "loaded_class": type(loaded_model).__name__,
            "loaded_checkpoint_path": str(loaded_cfg.general.test_only),
        }
        del loaded_model
    return results


def main():
    if (not CHECKPOINT.exists() or not REAL_DATA.exists()
            or not REAL_VALIDATION_DATA.exists()):
        raise FileNotFoundError({"checkpoint": str(CHECKPOINT),
                                 "real_data": str(REAL_DATA),
                                 "validation_data": str(REAL_VALIDATION_DATA)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(1234)
    model = ConditionalGraphBetaDiffusionH2.load_from_checkpoint(
        CHECKPOINT, map_location=device).to(device)
    model.eval()
    real_graphs, real_ys = load_real_graphs()
    validation_graphs, _ = load_real_graphs(REAL_VALIDATION_DATA)
    y_counts = {}
    for y in real_ys:
        key = str(int(np.argmax(y)))
        y_counts[key] = y_counts.get(key, 0) + 1

    mixed = mixed_batch_checks(model, device)
    modes = mode_checks(model, device)
    model.eval()
    numerical = numerical_checks(model, device)
    klub_precision = klub_precision_checks()
    thresholds, conditions = sampling_audit(model)
    routing = checkpoint_routing_checks()
    payload = {
        "environment": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_T": model.T,
            "formal_config_T": 100,
        },
        "real_train_h2": graph_summary(real_graphs),
        "real_validation_h2": graph_summary(validation_graphs),
        "threshold_selection_status": (
            "Candidates are diagnostics only; select on validation data, "
            "never test MMD. No final threshold selected."),
        "legacy_cached_real_train_y_counts": y_counts,
        "effective_h2_condition": fixed_h2_condition(model.ydim_output).tolist(),
        "mixed_batch_eta_and_gradient": mixed,
        "mode_smokes": modes,
        "numerical": numerical,
        "klub_precision": klub_precision,
        "threshold_sampling_100_each": thresholds,
        "condition_perturbation": conditions,
        "checkpoint_routing": routing,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"AUDIT_RESULT={OUTPUT}")


if __name__ == "__main__":
    main()
