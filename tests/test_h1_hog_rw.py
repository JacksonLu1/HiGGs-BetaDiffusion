"""CPU-only feature/regression smoke. No real dataset or model downloads."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dgd"))
sys.path.insert(0, str(ROOT / "tests"))
from diffusion.hog_rw_features import (
    HOG_RW_ALGORITHM, get_rw_feat, hog_rw_options, masked_hog_rw_features,
)

REQUIRE_INTEGRATION = "--require-integration" in sys.argv
if REQUIRE_INTEGRATION:
    sys.argv.remove("--require-integration")


class FeatureTests(unittest.TestCase):
    def test_options(self):
        self.assertFalse(hog_rw_options()["enabled"])
        for bad in ({"enabled": "false"}, {"rw_depth": True},
                    {"rw_depth": 0}, {"unknown": 3}):
            with self.assertRaises((TypeError, ValueError)):
                hog_rw_options(bad)

    def test_matches_independent_matrix_power_reference(self):
        # Test chains, odd cycles, bipartite graphs, isolated and singleton nodes.
        graphs = [torch.zeros(1, 1), torch.zeros(4, 4)]
        for n, edges in [(4, [(0, 1), (1, 2)]),
                         (3, [(0, 1), (1, 2), (2, 0)]),
                         (4, [(0, 2), (0, 3), (1, 2), (1, 3)])]:
            a = torch.zeros(n, n)
            for i, j in edges:
                a[i, j] = a[j, i] = 1
            graphs.append(a)
        for a in graphs:
            a = a.unsqueeze(0)
            p = a / (a.sum(-1, keepdim=True) + 1e-8)
            powers = torch.stack([torch.linalg.matrix_power(p, j)
                                  for j in range(2, 6)], dim=1)
            expected_x = powers.diagonal(dim1=2, dim2=3).permute(0, 2, 1)
            index = (powers <= 0).sum(1)
            expected_e = torch.nn.functional.one_hot(index, 5).float()
            x, e = get_rw_feat(4, a)
            self.assertTrue(torch.allclose(x, expected_x, atol=1e-6))
            self.assertTrue(torch.equal(e.permute(0, 2, 3, 1), expected_e))

    def test_masks_symmetry_padding_and_permutation(self):
        a = torch.zeros(1, 5, 5)
        a[0, 0, 1] = a[0, 1, 0] = 1
        a[0, 1, 2] = a[0, 2, 1] = 1
        mask = torch.tensor([[True, True, True, False, False]])
        x, e = masked_hog_rw_features(a, mask, 4)
        sx, se = masked_hog_rw_features(a[:, :3, :3], mask[:, :3], 4)
        self.assertTrue(torch.equal(x[:, :3], sx))
        self.assertTrue(torch.equal(e[:, :3, :3], se))
        self.assertEqual(int(x[:, 3:].count_nonzero()), 0)
        self.assertEqual(int(e[:, 3:].count_nonzero()), 0)
        self.assertEqual(int(e[:, :, 3:].count_nonzero()), 0)
        self.assertEqual(int(e[:, torch.arange(5), torch.arange(5)].count_nonzero()), 0)
        self.assertTrue(torch.equal(e, e.transpose(1, 2)))
        perm = torch.tensor([4, 2, 0, 3, 1])
        px, pe = masked_hog_rw_features(a[:, perm][:, :, perm], mask[:, perm], 4)
        self.assertTrue(torch.allclose(px, x[:, perm], atol=1e-6))
        self.assertTrue(torch.equal(pe, e[:, perm][:, :, perm]))

    def test_singleton_empty_and_no_rng_consumption(self):
        for n in (1, 3):
            state = torch.get_rng_state()
            x, e = masked_hog_rw_features(torch.zeros(2, n, n),
                                         torch.ones(2, n, dtype=torch.bool), 4)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertTrue(torch.isfinite(x).all() and torch.isfinite(e).all())
            self.assertEqual(tuple(x.shape), (2, n, 4))
            self.assertEqual(tuple(e.shape), (2, n, n, 5))

    def test_invalid_inputs_fail(self):
        mask = torch.ones(1, 2, dtype=torch.bool)
        for a in (torch.tensor([[[0., 1.], [0., 0.]]]),
                  torch.full((1, 2, 2), .1), torch.full((1, 2, 2), float("nan"))):
            with self.assertRaises(ValueError):
                masked_hog_rw_features(a, mask, 4)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1
            from test_beta_h1_semantics import _cfg, _DatasetInfo
        except ModuleNotFoundError as error:
            if REQUIRE_INTEGRATION:
                raise
            raise unittest.SkipTest(f"Integration dependency unavailable: {error}")
        cls.Model = ConditionalGraphBetaDiffusionH1
        cls.cfg_factory = staticmethod(_cfg)
        cls.Info = _DatasetInfo

    def model(self, enabled=True, twohop=False):
        cfg = self.cfg_factory()
        if enabled:
            cfg.model.hog_rw = {"enabled": True, "rw_depth": 4}
        if twohop:
            cfg.model.two_hop = {"enabled": True, "channels": 4,
                                 "normalization": "mean", "warm_start_checkpoint": None}
        return self.Model(cfg, self.Info()).eval()

    def test_dimensions_and_dataset_info_not_mutated(self):
        info = self.Info()
        before = dict(info.input_dims)
        base, model = self.model(False), self.model(True)
        self.assertEqual(info.input_dims, before)
        self.assertEqual(model.model.mlp_in_X[0].in_features,
                         base.model.mlp_in_X[0].in_features + 4)
        self.assertEqual(model.model.mlp_in_E[0].in_features,
                         base.model.mlp_in_E[0].in_features + 5)
        self.assertEqual(model.model.out_dim_X, base.model.out_dim_X)
        self.assertEqual(model.model.out_dim_E, base.model.out_dim_E)
        self.assertEqual(model.model.out_dim_y, base.model.out_dim_y)
        self.assertFalse(model.two_hop_options["enabled"])

    def test_mean_twohop_is_retained(self):
        from models.h1_twohop_transformer import H1TwoHopGraphTransformer
        model = self.model(twohop=True)
        self.assertIsInstance(model.model, H1TwoHopGraphTransformer)
        self.assertTrue(model.two_hop_options["enabled"])
        self.assertEqual(model.model.two_hop_edge.normalization, "mean")
        self.assertEqual(model.model.two_hop_edge.channels, 4)
        self.assertTrue(any("two_hop_edge" in name for name in model.state_dict()))

    def test_disabled_matches_prepatch_backbone(self):
        path = os.environ.get("H1_HOG_BASELINE_FILE")
        if not path:
            if REQUIRE_INTEGRATION:
                self.fail("Set H1_HOG_BASELINE_FILE to the pre-patch backup")
            self.skipTest("No pre-patch backup path provided")
        spec = importlib.util.spec_from_file_location("h1_pre_hog_baseline", path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        torch.manual_seed(613)
        old = original.ConditionalGraphBetaDiffusionH1(self.cfg_factory(), self.Info()).eval()
        old_rng = torch.get_rng_state()
        torch.manual_seed(613)
        new = self.model(False)
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        new.load_state_dict(old.state_dict(), strict=True)
        mask = torch.ones(1, 3, dtype=torch.bool)
        x, e = torch.zeros(1, 3, 1), torch.zeros(1, 3, 3, 1)
        y, t = torch.tensor([[1., 0.]]), torch.tensor([[.5]])
        eta_x, eta_e = old._eta_tensors(1, 3, x)
        args = (x, e, y, t, old._alpha(t), eta_x, eta_e, mask)
        for a, b in zip(old._predict_clean(*args), new._predict_clean(*args)):
            self.assertTrue(torch.equal(a, b))

    def test_hog_disabled_preserves_prepatch_mean_model(self):
        path = os.environ.get("H1_HOG_BASELINE_FILE")
        if not path:
            if REQUIRE_INTEGRATION:
                self.fail("Set H1_HOG_BASELINE_FILE to the pre-patch backup")
            self.skipTest("No pre-patch backup path provided")
        spec = importlib.util.spec_from_file_location("mean_pre_hog_stack", path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        cfg = self.cfg_factory()
        cfg.model.two_hop = {"enabled": True, "channels": 4,
                             "normalization": "mean", "warm_start_checkpoint": None}
        torch.manual_seed(614)
        old = original.ConditionalGraphBetaDiffusionH1(cfg, self.Info()).eval()
        state = torch.get_rng_state()
        torch.manual_seed(614)
        new = self.model(False, twohop=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        new.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            old.model.two_hop_edge.output[-1].weight.fill_(.01)
            new.model.two_hop_edge.output[-1].weight.fill_(.01)
        mask = torch.ones(1, 3, dtype=torch.bool)
        x, e = torch.zeros(1, 3, 1), torch.zeros(1, 3, 3, 1)
        y, t = torch.tensor([[1., 0.]]), torch.tensor([[.5]])
        eta_x, eta_e = old._eta_tensors(1, 3, x)
        args = (x, e, y, t, old._alpha(t), eta_x, eta_e, mask)
        for a, b in zip(old._predict_clean(*args), new._predict_clean(*args)):
            self.assertTrue(torch.equal(a, b))

    def test_sampling_output_schema_and_fixed_condition(self):
        model = self.model(twohop=True)
        y = torch.tensor([[1., 0.], [0., 1.]])
        graphs, _ = model.sampling_batch_h1(
            batch_id=0, batch_size=2, keep_chain=0, number_chain_steps=1,
            num_nodes=torch.tensor([1, 3]), y=y.clone(), figures=False)
        for n, (x, e) in zip((1, 3), graphs):
            self.assertEqual(tuple(x.shape), (n,))
            self.assertEqual(tuple(e.shape), (n, n))
            self.assertTrue(torch.equal(e, e.T))
            self.assertEqual(int(e.diagonal().count_nonzero()), 0)
            self.assertTrue(torch.isfinite(e.float()).all())

    def test_checkpoint_metadata_and_mismatch(self):
        model = self.model()
        ckpt = {"hyper_parameters": {"cfg": copy.deepcopy(model.cfg)}}
        model.on_save_checkpoint(ckpt)
        self.assertEqual(ckpt["h1_hog_rw"]["algorithm"], HOG_RW_ALGORITHM)
        restored = self.model()
        restored.on_load_checkpoint(ckpt)
        restored.load_state_dict(model.state_dict(), strict=True)
        with self.assertRaises(RuntimeError):
            self.model(False).on_load_checkpoint(ckpt)
        broken = copy.deepcopy(ckpt)
        broken["h1_hog_rw"]["rw_depth"] = 5
        with self.assertRaises(RuntimeError):
            self.model().on_load_checkpoint(broken)

    def test_stack_metadata_and_v1_hog_only_compatibility(self):
        model = self.model(twohop=True)
        ckpt = {"hyper_parameters": {"cfg": copy.deepcopy(model.cfg)}}
        model.on_save_checkpoint(ckpt)
        self.assertEqual(ckpt["h1_hog_rw"]["version"], 2)
        self.assertIs(ckpt["h1_hog_rw"]["two_hop"], True)
        self.assertEqual(ckpt["h1_two_hop"]["normalization"], "mean")
        restored = self.model(twohop=True)
        restored.on_load_checkpoint(ckpt)
        restored.load_state_dict(model.state_dict(), strict=True)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))
        for field, value in (("two_hop", False), ("version", 1)):
            bad = copy.deepcopy(ckpt)
            bad["h1_hog_rw"][field] = value
            with self.assertRaises(RuntimeError):
                self.model(twohop=True).on_load_checkpoint(bad)
        with self.assertRaises(RuntimeError):
            self.model().on_load_checkpoint(ckpt)
        old_mean = self.model(False, twohop=True)
        old_mean_ckpt = {"hyper_parameters": {"cfg": copy.deepcopy(old_mean.cfg)}}
        old_mean.on_save_checkpoint(old_mean_ckpt)
        with self.assertRaises(RuntimeError):
            self.model(twohop=True).on_load_checkpoint(old_mean_ckpt)
        single = self.model()
        legacy = {"hyper_parameters": {"cfg": copy.deepcopy(single.cfg)}}
        single.on_save_checkpoint(legacy)
        legacy["h1_hog_rw"]["version"] = 1
        single.on_load_checkpoint(legacy)

    def test_synthetic_train_validation_checkpoint_reload(self):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        from torch_geometric.data import Batch, Data
        from torch.utils.data import DataLoader
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        graph = Data(x=torch.ones(3, 1), edge_index=edge_index,
                     edge_attr=torch.tensor([[0., 1.]]).repeat(4, 1),
                     y=torch.tensor([[1., 0.]]))
        loader = DataLoader([graph] * 4, batch_size=2, collate_fn=Batch.from_data_list)
        with tempfile.TemporaryDirectory(prefix="h1_hog_synthetic_") as directory:
            model = self.model(twohop=True).train()
            callback = ModelCheckpoint(dirpath=directory, monitor="val/h1_total",
                                       mode="min", save_top_k=1, save_last=True)
            trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1,
                              limit_train_batches=2, limit_val_batches=2,
                              num_sanity_val_steps=0, logger=False,
                              enable_progress_bar=False, callbacks=[callback])
            trainer.fit(model, loader, loader)
            for key in ("train/h1_total", "val/h1_total"):
                self.assertTrue(torch.isfinite(trainer.callback_metrics[key]))
            for path in (callback.best_model_path, callback.last_model_path):
                self.assertTrue(Path(path).is_file())
            restored = self.Model.load_from_checkpoint(callback.last_model_path)
            self.assertEqual(restored.hog_rw_options, model.hog_rw_options)
            self.assertTrue(restored.two_hop_options["enabled"])
            self.assertEqual(restored.model.two_hop_edge.normalization, "mean")
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), restored.state_dict()[key].cpu()))


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False).result
    if not result.wasSuccessful():
        sys.exit(1)
    print("PASS: HOG_RW_FEATURE_TESTS")
    if REQUIRE_INTEGRATION:
        print("PASS: HOG_RW_INTEGRATION_AND_SYNTHETIC_SMOKE")
        print("PASS: TWOHOP_MEAN_PLUS_HOG_RW_INTEGRATION")
    else:
        print("Integration is optional in this mode; inspect skipped tests above.")
    print("No real dataset training, H2/Glue sampling, or metric evaluation performed.")
