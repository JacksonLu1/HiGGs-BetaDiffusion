from __future__ import annotations

import sys
import unittest
from pathlib import Path

import networkx as nx
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dgd"))
sys.path.insert(0, str(ROOT / "higgs"))

from interface_ablation import (  # noqa: E402
    adapt_h2_to_glue_class,
    adapt_h2_to_h1_conditions,
    apply_component_policy,
    apply_upstream_lcc,
    audit_glue_training_cache,
    unique_undirected_pairs,
)
from sampling import build_from_components  # noqa: E402


CONFIG_NAMES = {
    "C0": "sample_sbm_C0_current",
    "D1": "sample_sbm_D1_full_direct_eval",
    "D2": "sample_sbm_D2_lcc_direct_eval",
    "D3": "sample_sbm_D3_lcc_h1adapter_eval",
    "D4": "sample_sbm_D4_lcc_glueadapter_eval",
    "C3": "sample_sbm_C3_lcc_adapter_eval",
}


def _general_config(name):
    path = ROOT / "configs" / "general" / f"{CONFIG_NAMES[name]}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _changed_interface_fields(left, right):
    left_interface = _general_config(left)["interface"]
    right_interface = _general_config(right)["interface"]
    return {
        key for key in left_interface
        if left_interface[key] != right_interface[key]
    }


def _adjacency(n_nodes, edges=()):
    adjacency = torch.zeros((n_nodes, n_nodes), dtype=torch.long)
    for left, right in edges:
        adjacency[left, right] = 1
        adjacency[right, left] = 1
    return adjacency


def test_upstream_lcc_selects_three_node_component():
    nodes = torch.tensor([10, 11, 12, 20, 21], dtype=torch.long)
    adjacency = _adjacency(5, [(0, 1), (1, 2), (3, 4)])
    used_nodes, used_adjacency = apply_upstream_lcc(nodes, adjacency)
    assert torch.equal(used_nodes, torch.tensor([10, 11, 12]))
    assert torch.equal(
        used_adjacency, _adjacency(3, [(0, 1), (1, 2)]))


def test_upstream_lcc_all_isolated_matches_author_tie_behavior():
    nodes = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    used_nodes, used_adjacency = apply_upstream_lcc(
        nodes, _adjacency(4))
    assert used_nodes.shape == (1,)
    assert used_nodes.item() == 3
    assert used_adjacency.shape == (1, 1)


def test_upstream_lcc_single_node_is_safe():
    nodes = torch.tensor([7], dtype=torch.long)
    adjacency = _adjacency(1)
    used_nodes, used_adjacency = apply_upstream_lcc(nodes, adjacency)
    assert torch.equal(used_nodes, nodes)
    assert torch.equal(used_adjacency, adjacency)


def test_lcc_keeps_node_and_adjacency_indices_synchronized():
    nodes = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    adjacency = _adjacency(4, [(0, 2), (2, 3)])
    used_nodes, used_adjacency = apply_upstream_lcc(nodes, adjacency)
    assert torch.equal(used_nodes, torch.tensor([5, 7, 8]))
    assert torch.equal(
        used_adjacency, adjacency[[0, 2, 3]][:, [0, 2, 3]])


def test_full_policy_returns_input_objects_unchanged():
    nodes = torch.tensor([0, 1], dtype=torch.long)
    adjacency = _adjacency(2)
    used_nodes, used_adjacency, audit = apply_component_policy(
        nodes, adjacency, "full", label="test")
    assert used_nodes is nodes
    assert used_adjacency is adjacency
    assert audit["removed_nodes"] == 0


def test_direct_and_sbm_semantic_h2_to_h1_conditions():
    classes = torch.tensor([0, 1, 1], dtype=torch.long)
    direct = adapt_h2_to_h1_conditions(
        classes, 2, "direct", dataset_name="sbm")
    semantic = adapt_h2_to_h1_conditions(
        classes, 2, "sbm_semantic", dataset_name="sbm")
    assert torch.equal(
        direct, torch.tensor([[1, 0], [0, 1], [0, 1]]))
    assert torch.equal(
        semantic, torch.tensor([[1, 0], [1, 0], [1, 0]]))


def test_glue_semantic_adapter_requires_verified_class():
    audit = {"verified_present_class_index": 0}
    assert adapt_h2_to_glue_class(
        1, "sbm_semantic", configured_present_class_index=0,
        training_cache_audit=audit, dataset_name="sbm") == 0
    try:
        adapt_h2_to_glue_class(
            1, "sbm_semantic", configured_present_class_index=None,
            training_cache_audit=audit, dataset_name="sbm")
    except ValueError as error:
        assert "cannot be null" in str(error)
    else:
        raise AssertionError("null Glue semantic class did not fail")
    try:
        adapt_h2_to_glue_class(
            1, "sbm_semantic", configured_present_class_index=1,
            training_cache_audit=audit, dataset_name="sbm")
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched Glue semantic class did not fail")


def test_actual_original_glue_train_cache_verifies_class_zero():
    cache = (
        ROOT / "data" / "sbm" / "processed"
        / "proc_train_X_r1_majority_max500_bs8.pt")
    audit = audit_glue_training_cache(str(cache))
    assert audit["sample_count"] == 80
    assert audit["one_hot_histogram"] == {"1,0": 80}
    assert audit["class_histogram"] == {"0": 80}
    assert audit["verified_present_class_index"] == 0


def test_pair_dedup_removes_reverse_and_self_loop():
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 1, 0]], dtype=torch.long)
    edge_types = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    assert unique_undirected_pairs(edge_index, edge_types) == [
        (0, 1, 1), (0, 2, 1)]


def test_glue_aggregation_preserves_intra_and_only_adds_inter_edges():
    h1_nodes = [
        torch.tensor([0, 0], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
    ]
    h1_adjs = [
        _adjacency(2, [(0, 1)]),
        _adjacency(2, [(0, 1)]),
    ]
    mappings = [
        {"start_ind": 0, "end_ind": 2, "n_nodes": 2},
        {"start_ind": 2, "end_ind": 4, "n_nodes": 2},
    ]
    node_to_class = {index: 0 for index in range(4)}
    glue_pair = nx.Graph()
    glue_pair.add_nodes_from(range(4))
    glue_pair.add_edges_from([(0, 1), (2, 3), (1, 2)])

    graph, _, audit = build_from_components(
        h1_nodes, h1_adjs, [glue_pair], mappings, node_to_class,
        [(0, 1)], strict_assertions=True, return_audit=True)

    assert graph.has_edge(0, 1)
    assert graph.has_edge(2, 3)
    assert graph.has_edge(1, 2)
    assert audit["intra_h1_edges_changed"] == 0
    assert audit["invalid_intra_added_edge_count"] == 0
    assert audit["glue_added_edge_count"] == 1


def test_single_variable_config_differences():
    assert _changed_interface_fields("C0", "D1") == {"force_eval"}
    assert _changed_interface_fields("D1", "D2") == {
        "h2_component_policy", "h1_component_policy"}
    assert _changed_interface_fields("D2", "D3") == {
        "h2_to_h1_condition_mode"}
    assert _changed_interface_fields("D2", "D4") == {
        "h2_to_glue_condition_mode", "glue_present_class_index"}
    assert _changed_interface_fields("D3", "C3") == {
        "h2_to_glue_condition_mode", "glue_present_class_index"}
    assert _changed_interface_fields("D4", "C3") == {
        "h2_to_h1_condition_mode"}


def test_ablation_configs_share_all_non_interface_fields_and_checkpoints():
    configs = [_general_config(name) for name in CONFIG_NAMES]
    baselines = [
        {key: value for key, value in config.items() if key != "interface"}
        for config in configs
    ]
    assert all(config == baselines[0] for config in baselines)
    checkpoints = [
        (config["h2_model"], config["h1_model"], config["glue_model"])
        for config in configs
    ]
    assert all(value == checkpoints[0] for value in checkpoints)


def test_hydra_composition_does_not_change_model_dataset_or_train():
    composed = []
    config_dir = str((ROOT / "configs").resolve())
    for name in CONFIG_NAMES.values():
        with initialize_config_dir(
                config_dir=config_dir, version_base="1.1"):
            cfg = compose(
                config_name="sample",
                overrides=[
                    f"general={name}",
                    "dataset=sbm_r1",
                    "model=beta_h2",
                ],
            )
        composed.append(OmegaConf.to_container(
            cfg, resolve=True, enum_to_str=True))
    for section in ("model", "dataset", "train"):
        expected = composed[0][section]
        assert all(config[section] == expected for config in composed)


class InterfaceAblationTests(unittest.TestCase):
    def test_upstream_lcc_selects_three_node_component(self):
        test_upstream_lcc_selects_three_node_component()

    def test_upstream_lcc_all_isolated_matches_author_tie_behavior(self):
        test_upstream_lcc_all_isolated_matches_author_tie_behavior()

    def test_upstream_lcc_single_node_is_safe(self):
        test_upstream_lcc_single_node_is_safe()

    def test_lcc_keeps_node_and_adjacency_indices_synchronized(self):
        test_lcc_keeps_node_and_adjacency_indices_synchronized()

    def test_full_policy_returns_input_objects_unchanged(self):
        test_full_policy_returns_input_objects_unchanged()

    def test_direct_and_sbm_semantic_h2_to_h1_conditions(self):
        test_direct_and_sbm_semantic_h2_to_h1_conditions()

    def test_glue_semantic_adapter_requires_verified_class(self):
        test_glue_semantic_adapter_requires_verified_class()

    def test_actual_original_glue_train_cache_verifies_class_zero(self):
        test_actual_original_glue_train_cache_verifies_class_zero()

    def test_pair_dedup_removes_reverse_and_self_loop(self):
        test_pair_dedup_removes_reverse_and_self_loop()

    def test_glue_aggregation_preserves_intra_and_only_adds_inter_edges(self):
        test_glue_aggregation_preserves_intra_and_only_adds_inter_edges()

    def test_single_variable_config_differences(self):
        test_single_variable_config_differences()

    def test_ablation_configs_share_non_interface_fields_and_checkpoints(self):
        test_ablation_configs_share_all_non_interface_fields_and_checkpoints()

    def test_hydra_composition_does_not_change_model_dataset_or_train(self):
        test_hydra_composition_does_not_change_model_dataset_or_train()


if __name__ == "__main__":
    unittest.main()
