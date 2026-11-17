"""CPU-only tests of the H1 noisy-state k-core feature extractor."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dgd"))
from diffusion.h1_kcore_features import (
    h1_kcore_options, masked_h1_kcore_features,
)


def adjacency(n, edges):
    graph = torch.zeros((n, n))
    for u, v in edges:
        graph[u, v] = graph[v, u] = 1
    return graph


def extract(adj, mask=None, t=None, *, k=2, max_t=0.5):
    if mask is None:
        mask = torch.ones(adj.shape[:2], dtype=torch.bool, device=adj.device)
    if t is None:
        t = torch.zeros(adj.shape[0], device=adj.device)
    return masked_h1_kcore_features(adj, mask, t, k=k, max_t=max_t)


class H1KCoreFeatureTests(unittest.TestCase):
    def test_options_default_disabled_and_invalid(self):
        self.assertEqual(h1_kcore_options(), {"enabled": False, "k": 3, "max_t": 0.5})
        self.assertEqual(h1_kcore_options({"enabled": True, "k": 2, "max_t": 1}),
                         {"enabled": True, "k": 2, "max_t": 1.0})
        for invalid in (True, {"enabled": "yes"}, {"k": 0}, {"k": True},
                        {"max_t": True}, {"max_t": float("nan")},
                        {"max_t": float("inf")}, {"max_t": 1.1}, {"bogus": 2}):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                h1_kcore_options(invalid)

    def test_known_core_numbers_and_distinct_connected_components(self):
        # Two triangles, one pendant on the first, one isolate.
        a = adjacency(8, [(0, 1), (1, 2), (2, 0), (2, 6),
                          (3, 4), (4, 5), (5, 3)])
        x, e = extract(a[None])
        expected_cores = torch.tensor([2, 2, 2, 2, 2, 2, 1, 0])
        torch.testing.assert_close(x[0, :, 0], expected_cores.float() / 7)
        torch.testing.assert_close(x[0, :, 1], (expected_cores >= 2).float())
        expected_pairs = torch.zeros(8, 8)
        for start in (0, 3):
            expected_pairs[start:start + 3, start:start + 3] = 1 - torch.eye(3)
        torch.testing.assert_close(e[0, :, :, 0], expected_pairs)

    def test_pair_relation_is_not_limited_to_adjacent_nodes(self):
        a = adjacency(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        _, e = extract(a[None])
        self.assertEqual(a[0, 2], 0)
        self.assertEqual(e[0, 0, 2, 0], 1)

    def test_time_gate_per_graph_inclusive_and_no_rng_or_mutation(self):
        a = adjacency(3, [(0, 1), (1, 2), (2, 0)]).expand(3, -1, -1).clone()
        a.requires_grad_(True)
        t = torch.tensor([[0.01], [0.5], [0.51]], requires_grad=True)
        before_a = a.detach().clone()
        before_rng = torch.random.get_rng_state()
        x, e = extract(a, t=t)
        self.assertTrue(torch.equal(a, before_a))
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        self.assertFalse(x.requires_grad)
        self.assertFalse(e.requires_grad)
        self.assertTrue(torch.equal(x[0], x[1]))
        self.assertTrue(torch.equal(e[0], e[1]))
        self.assertEqual(x[0].count_nonzero().item(), 6)
        self.assertEqual(e[0].count_nonzero().item(), 6)
        self.assertEqual(x[2].count_nonzero().item(), 0)
        self.assertEqual(e[2].count_nonzero().item(), 0)

    def test_padding_diagonal_and_node_permutation(self):
        # Non-contiguous valid slots; contents in padding and diagonal are ignored.
        valid = torch.tensor([True, False, True, True, False, True])
        a = adjacency(6, [(0, 2), (2, 3), (3, 0), (3, 5)])
        a[1] = 9
        a[:, 4] = -2
        a.fill_diagonal_(4)
        x, e = extract(a[None], valid[None])
        keep = valid.nonzero(as_tuple=True)[0]
        ref_x, ref_e = extract(a[keep][:, keep][None])
        torch.testing.assert_close(x[:, keep], ref_x)
        torch.testing.assert_close(e[:, keep][:, :, keep], ref_e)
        self.assertFalse(x[:, ~valid].any())
        self.assertFalse(e[:, ~valid].any())
        self.assertFalse(e[:, :, ~valid].any())
        self.assertFalse(e[:, torch.arange(6), torch.arange(6)].any())
        self.assertTrue(torch.equal(e, e.transpose(1, 2)))
        p = torch.tensor([4, 2, 0, 5, 3, 1])
        px, pe = extract(a[p][:, p][None], valid[p][None])
        torch.testing.assert_close(px, x[:, p])
        torch.testing.assert_close(pe, e[:, p][:, :, p])

    def test_singleton_empty_and_high_k(self):
        for n in (0, 1, 4):
            a = torch.zeros((2, n, n), dtype=torch.bool)
            x, e = extract(a)
            self.assertEqual(tuple(x.shape), (2, n, 2))
            self.assertEqual(tuple(e.shape), (2, n, n, 1))
            self.assertFalse(x.any())
            self.assertFalse(e.any())
        a = adjacency(3, [(0, 1), (1, 2), (2, 0)])
        x, e = extract(a[None], k=4)
        self.assertTrue(torch.equal(x[0, :, 0], torch.ones(3)))
        self.assertFalse(x[..., 1].any())
        self.assertFalse(e.any())

    def test_recomputes_features_when_noisy_graph_changes(self):
        before = adjacency(3, [(0, 1), (1, 2)])
        after = adjacency(3, [(0, 1), (1, 2), (2, 0)])
        x, e = extract(torch.stack((before, after)))
        self.assertFalse(x[0, :, 1].any())
        self.assertFalse(e[0].any())
        self.assertTrue(x[1, :, 1].all())
        self.assertEqual(e[1].count_nonzero().item(), 6)

    def test_invalid_real_edges_and_time(self):
        for pair in ((1, 0), (.1, .1), (float("nan"), 1)):
            a = torch.tensor([[[0., pair[0]], [pair[1], 0.]]])
            with self.assertRaises(ValueError):
                extract(a)
        for invalid_t in (torch.tensor([float("nan")]), torch.tensor([1.1]),
                          torch.tensor([-.01]), torch.zeros(1, 2)):
            with self.assertRaises(ValueError):
                extract(torch.zeros(1, 2, 2), t=invalid_t)


if __name__ == "__main__":
    unittest.main()
