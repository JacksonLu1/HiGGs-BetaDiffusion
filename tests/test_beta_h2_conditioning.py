"""Regression tests for the fixed, non-semantic HiGGs h2 condition."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dgd"))
sys.path.insert(0, str(ROOT / "higgs"))

from datasets.cora_dataset import CORADataset  # noqa: E402
from datasets.fb_hierarchies import FBHierarchiesDataset  # noqa: E402
from datasets.sbm_hierarchies import SBMDataset  # noqa: E402
from diffusion.beta_diffusion_utils import fixed_h2_condition  # noqa: E402
from diffusion_model_beta_h2 import ConditionalGraphBetaDiffusionH2  # noqa: E402
import sampling as higgs_sampling  # noqa: E402


DATA_ROOT = Path(os.environ.get(
    "HIGGS_AUDIT_DATA_ROOT", r"E:\HiGGs-main1\HiGGs-main\data"))


def _assert_fixed_dataset(dataset_class, root: Path, kwargs):
    expected = fixed_h2_condition(2)
    for stage in ("train", "val", "test"):
        dataset = dataset_class(stage=stage, root=str(root), h=2, **kwargs)
        assert len(dataset) > 0
        for index in (0, len(dataset) - 1):
            assert torch.equal(dataset[index].y.cpu(), expected)
        # Re-instantiation must not restore the random labels in legacy caches.
        repeated = dataset_class(stage=stage, root=str(root), h=2, **kwargs)
        assert torch.equal(repeated[0].y.cpu(), expected)


def test_all_h2_datasets_use_fixed_condition():
    _assert_fixed_dataset(
        SBMDataset, DATA_ROOT / "sbm",
        dict(max_size=500, resolution=2, n_samples=24, n_workers=1,
             batch_size=8, regime="majority"))
    _assert_fixed_dataset(
        CORADataset, DATA_ROOT / "cora_r2",
        dict(max_size=500, resolution=2, n_samples=240, n_workers=1,
             batch_size=8, regime="majority"))
    _assert_fixed_dataset(
        FBHierarchiesDataset, DATA_ROOT / "fb",
        dict(max_size=400, resolution=10, n_samples=240, n_workers=1,
             batch_size=8, regime="majority"))


def test_standalone_beta_default_condition():
    model = SimpleNamespace(ydim_output=2)
    condition = ConditionalGraphBetaDiffusionH2._sampling_condition(
        model, None, batch_size=4, device=torch.device("cpu"))
    assert torch.equal(condition, fixed_h2_condition(2).expand(4, -1))


def test_higgs_full_sampling_passes_fixed_condition():
    captured = {}

    class FakeBeta:
        ydim_output = 2

        def to(self, device):
            return self

        def eval(self):
            return self

        def sampling_batch_h2(self, **kwargs):
            captured.update(kwargs)
            graph = nx.Graph()
            graph.add_node(0)
            return [[torch.tensor([0]), torch.zeros((1, 1), dtype=torch.long)]], [graph]

    fake = FakeBeta()
    old_class = higgs_sampling.ConditionalGraphBetaDiffusionH2
    old_loader = higgs_sampling.get_samplers
    higgs_sampling.ConditionalGraphBetaDiffusionH2 = FakeBeta
    higgs_sampling.get_samplers = lambda cfg: (cfg, fake)
    cfg = OmegaConf.create({
        "general": {"gpus": 0},
        "dataset": {"h": 2},
        "model": {"extra_features": None},
    })
    try:
        higgs_sampling.get_h2_graph(cfg, intermediate_figures=False)
    finally:
        higgs_sampling.ConditionalGraphBetaDiffusionH2 = old_class
        higgs_sampling.get_samplers = old_loader
    assert torch.equal(captured["y"].cpu(), fixed_h2_condition(2))


if __name__ == "__main__":
    test_all_h2_datasets_use_fixed_condition()
    test_standalone_beta_default_condition()
    test_higgs_full_sampling_passes_fixed_condition()
    print("beta_h2 conditioning tests passed")
