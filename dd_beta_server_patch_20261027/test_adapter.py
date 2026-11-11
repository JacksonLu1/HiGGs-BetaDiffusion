"""CPU-only numerical tests of DD's isolated node-count prior; no real datasets."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'HiGGs-main/dgd/diffusion/distributions.py'


def load_adapter():
    spec = importlib.util.spec_from_file_location('_baseline_distribution', BASE)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    graph = types.ModuleType('datasets.graph_collection')
    class FakeInfos:
        def __init__(self, datamodule, config):
            self.n_nodes = torch.tensor([0., .25, .75], dtype=torch.float64)
    graph.GraphCollectionInfos = FakeInfos
    spec = importlib.util.spec_from_file_location('_dd_prior_test', HERE / 'payload/dgd/datasets/dd_beta_adapter.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'diffusion.distributions': baseline, 'datasets.graph_collection': graph}):
        spec.loader.exec_module(module)
    return module, baseline


class PriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter, cls.baseline = load_adapter()

    def test_fractional_probabilities_and_out_of_range(self):
        prior = self.adapter.DDDistributionNodes(torch.tensor([0., .25, .75], dtype=torch.float64))
        actual = prior.log_prob(torch.tensor([1, 2, 3, -1]))
        expected = torch.log(torch.tensor([.25, .75, 0., 0.], dtype=torch.float64) + 1e-30)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())

    def test_sampling_distribution_and_rng_are_unchanged(self):
        hist = torch.tensor([0., .25, .75])
        new = self.adapter.DDDistributionNodes(hist)
        old = self.baseline.DistributionNodes(hist)
        with torch.random.fork_rng():
            torch.manual_seed(1234)
            a = old.sample_n(200, 'cpu')
            torch.manual_seed(1234)
            b = new.sample_n(200, 'cpu')
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(old.prob, new.prob)
        torch.testing.assert_close(old.m.probs, new.m.probs)

    def test_empty_counts_and_wrong_rank(self):
        prior = self.adapter.DDDistributionNodes(torch.tensor([0., 1.]))
        self.assertEqual(tuple(prior.log_prob(torch.empty(0, dtype=torch.long)).shape), (0,))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.zeros((1, 1), dtype=torch.long))

    def test_only_dd_scope(self):
        with self.assertRaises(ValueError):
            self.adapter.DDBetaDatasetInfos(None, types.SimpleNamespace(name='planar'))
        infos = self.adapter.DDBetaDatasetInfos(None, types.SimpleNamespace(name='dd'))
        self.assertIsInstance(infos.nodes_dist, self.adapter.DDDistributionNodes)


if __name__ == '__main__':
    unittest.main(verbosity=2)
