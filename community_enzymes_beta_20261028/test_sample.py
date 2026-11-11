"""CPU-only sampling-adapter tests; no model training or GPU jobs are launched."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "payload/tools"))
import sc_beta_sample as sample

interface_path = HERE.parents[1] / "HiGGs-main/higgs/interface_ablation.py"
loader = importlib.util.spec_from_file_location("local_interface_for_sample_test", interface_path)
interface = importlib.util.module_from_spec(loader)
loader.loader.exec_module(interface)


def simple_graph(n):
    adjacency = torch.zeros((n, n), dtype=torch.long)
    for i in range(n - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1
    return adjacency


class PairTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(dataset=SimpleNamespace(name="community_small"),
                                   general=SimpleNamespace(gpus=0))
        self.model = SimpleNamespace(Xdim_output=1, Edim_output=2, ydim_output=2)
        self.adjs = [simple_graph(n) for n in (2, 3, 4)]
        self.nodes = [torch.zeros(n, dtype=torch.long) for n in (2, 3, 4)]
        self.edge_index = torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 1, 2]])
        self.edge_types = torch.ones(5, dtype=torch.long)
        self.cache = {"one_hot_histogram": {"0,1": 64}}

    def prepare(self, **changes):
        args = dict(cfg=self.cfg, model=self.model, h1_adjs=self.adjs, h1_nodes=self.nodes,
                    edge_index=self.edge_index, edge_types=self.edge_types, batch_size=1,
                    return_audit=True, interface=interface, cache_audit=self.cache)
        args.update(changes)
        return sample.prepare_pairs(**args)

    def test_pair_uniqueness_actual_padding_and_membership(self):
        data, loader, batch, count, remainder, pairs, audit = self.prepare()
        self.assertEqual(pairs, [(0, 1), (1, 2)])
        self.assertEqual((batch, count, remainder), (1, 2, 0))
        self.assertEqual(audit["membership_padding"], 7)
        self.assertEqual(audit["glue_condition_histogram_after_adapter"], {"0,1": 2})
        self.assertEqual(tuple(data[0].pos.shape), (7, 7))
        self.assertTrue(torch.equal(data[0].pos[:5, :5].bool(),
                                   (torch.arange(5) < 2)[:, None] == (torch.arange(5) < 2)[None, :]))
        self.assertFalse(data[0].pos[5:].any())
        self.assertFalse(data[0].pos[:, 5:].any())
        self.assertEqual(data[0].num_nodes, 5)
        self.assertEqual(data[1].num_nodes, 7)
        self.assertTrue(all(torch.equal(d.y, torch.tensor([[0.0, 1.0]])) for d in data))
        self.assertTrue(all(d.x.shape[1] == 1 and torch.all(d.x == 1) for d in data))
        self.assertEqual(len(list(loader)), 2)
        self.assertEqual(data[0].edge_index.max().item(), 4)

    def test_empty_h2_is_not_rejected_or_resampled(self):
        result = self.prepare(edge_index=torch.empty((2, 0), dtype=torch.long),
                              edge_types=torch.empty(0, dtype=torch.long))
        data, loader, _, count, remainder, pairs, audit = result
        self.assertEqual((data, loader, count, remainder, pairs), ([], [], 0, 0, []))
        self.assertEqual(audit["membership_padding"], 0)

    def test_edgeless_h1_pair_keeps_both_node_sets(self):
        adjs = [torch.zeros_like(adj) for adj in self.adjs]
        data, *_ = self.prepare(h1_adjs=adjs)
        self.assertEqual(data[0].num_nodes, 5)
        self.assertEqual(tuple(data[0].edge_index.shape), (2, 0))
        self.assertEqual(tuple(data[0].edge_attr.shape), (0, 2))

    def test_wrong_dataset_batch_category_or_node_labels_refused(self):
        self.cfg.dataset.name = "dd"
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.cfg.dataset.name = "enzymes"
        with self.assertRaises(RuntimeError):
            self.prepare(batch_size=2)
        with self.assertRaises(RuntimeError):
            self.prepare(edge_types=torch.full((5,), 2, dtype=torch.long))
        nodes = [n.clone() for n in self.nodes]
        nodes[0][0] = 1
        with self.assertRaises(RuntimeError):
            self.prepare(h1_nodes=nodes)

    def test_conditions_fixed_training_class0_and_correct_width(self):
        result = sample.fixed_h1_conditions(torch.tensor([0, 1, 1]), 2,
            sample.CONDITION_MODE, dataset_name="enzymes")
        self.assertTrue(torch.equal(result, torch.tensor([[1, 0]] * 3)))
        with self.assertRaises(RuntimeError):
            sample.fixed_h1_conditions(torch.tensor([0]), 3,
                sample.CONDITION_MODE, dataset_name="enzymes")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=str(HERE))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reports = {}
        for stage, (_, _, monitor) in sample.STAGES.items():
            directory = self.root / stage
            directory.mkdir()
            best = directory / "best.ckpt"
            best.write_bytes(b"mock checkpoint only")
            report = dict(stage=stage, epoch=499, global_step=1000, monitor=monitor,
                          manifest_sha256="abc", strict_reload=True, strict_best_reload=True,
                          best=str(best), best_sha256=sample.sha(best))
            path = directory / "formal_result.json"
            path.write_text(json.dumps(report))
            self.reports[stage] = (path, report)

    def test_complete_reports_accepted(self):
        self.assertEqual(set(sample.read_reports(self.root, "enzymes", "abc")), set(sample.STAGES))

    def test_partial_epoch_changed_hash_and_manifest_refused(self):
        path, report = self.reports["h1"]
        for field, bad in (("epoch", 498), ("strict_best_reload", False),
                           ("manifest_sha256", "other"), ("best_sha256", "wrong")):
            changed = dict(report)
            changed[field] = bad
            path.write_text(json.dumps(changed))
            with self.assertRaises(RuntimeError):
                sample.read_reports(self.root, "enzymes", "abc")
        path.write_text(json.dumps(report))


if __name__ == "__main__":
    unittest.main(verbosity=2)
