"""CPU integration checks; no real data, full chain, or model downloads."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from test_beta_h1_semantics import _cfg, _DatasetInfo
from diffusion.beta_diffusion_utils import unit_probability_to_shifted_state
from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1

REQUIRE_BASELINE = "--require-baseline" in sys.argv
if REQUIRE_BASELINE:
    sys.argv.remove("--require-baseline")


def config(enabled=None):
    cfg = _cfg()
    cfg.model.diffusion_steps = 4
    cfg.model.hog_rw = {"enabled": True, "rw_depth": 4}
    cfg.model.two_hop = {"enabled": True, "channels": 4,
                         "normalization": "mean", "warm_start_checkpoint": None}
    if enabled is not None:
        cfg.model.h1_kcore = {"enabled": enabled, "k": 3, "max_t": 0.5}
    return cfg


def checkpoint(model):
    state = {"hyper_parameters": {"cfg": copy.deepcopy(model.cfg)}}
    model.on_save_checkpoint(state)
    return state


def noisy_clique(model):
    mask = torch.ones(2, 4, dtype=torch.bool)
    probs = (1.0 - torch.eye(4)).view(1, 4, 4, 1).expand(2, -1, -1, -1)
    x = model._shifted_to_state(unit_probability_to_shifted_state(
        torch.ones(2, 4, 1), model.node_scale, model.node_shift))
    e = model._shifted_to_state(unit_probability_to_shifted_state(
        probs, model.edge_scale, model.edge_shift))
    return x, e, torch.tensor([[1., 0.], [1., 0.]]), torch.tensor([[.25], [.75]]), mask


class IntegrationTests(unittest.TestCase):
    def test_noisy_state_time_gate_and_widths(self):
        info = _DatasetInfo()
        original_widths = dict(info.input_dims)
        base = ConditionalGraphBetaDiffusionH1(config(False), info).eval()
        model = ConditionalGraphBetaDiffusionH1(config(True), info).eval()
        self.assertEqual(info.input_dims, original_widths)
        self.assertEqual(model.model.mlp_in_X[0].in_features,
                         base.model.mlp_in_X[0].in_features + 2)
        self.assertEqual(model.model.mlp_in_E[0].in_features,
                         base.model.mlp_in_E[0].in_features + 1)
        self.assertEqual(model.model.two_hop_edge.normalization, "mean")
        args = noisy_clique(model)
        bx, be, by = base._extra_data(*args)
        x, e, y = model._extra_data(*args)
        self.assertTrue(torch.equal(bx, x[..., :-2]))
        self.assertTrue(torch.equal(be, e[..., :-1]))
        self.assertTrue(torch.equal(by, y))
        self.assertTrue(torch.equal(x[0, :, -2:], torch.ones(4, 2)))
        self.assertEqual(int(x[1, :, -2:].count_nonzero()), 0)
        self.assertEqual(int(e[0, :, :, -1].count_nonzero()), 12)
        self.assertEqual(int(e[1, :, :, -1].count_nonzero()), 0)

    def test_checkpoint_separates_baseline_and_new_inputs(self):
        old = ConditionalGraphBetaDiffusionH1(config(), _DatasetInfo())
        disabled = ConditionalGraphBetaDiffusionH1(config(False), _DatasetInfo())
        enabled = ConditionalGraphBetaDiffusionH1(config(True), _DatasetInfo())
        old_meta, new_meta = checkpoint(old), checkpoint(enabled)
        self.assertNotIn("h1_kcore", old_meta)
        disabled.on_load_checkpoint(old_meta)
        disabled.load_state_dict(old.state_dict(), strict=True)
        enabled.on_load_checkpoint(new_meta)
        for model, meta in ((enabled, old_meta), (disabled, new_meta)):
            with self.assertRaisesRegex(RuntimeError, "k-core"):
                model.on_load_checkpoint(meta)
        mismatch = copy.deepcopy(new_meta)
        mismatch["h1_kcore"]["max_t"] = .75
        with self.assertRaisesRegex(RuntimeError, "k-core"):
            enabled.on_load_checkpoint(mismatch)

    def test_off_matches_immutable_archive(self):
        root = Path(__file__).resolve().parents[2]
        path = Path(os.environ.get("H1_KCORE_BASELINE_FILE") or (
            root / "server_source_20261111_120828_683654" / "dgd/diffusion_model_beta_h1.py"))
        if not path.is_file():
            if REQUIRE_BASELINE:
                self.fail("H1_KCORE_BASELINE_FILE must name the verified original H1 source")
            self.skipTest("Unmodified local snapshot unavailable")
        spec = importlib.util.spec_from_file_location("h1_before_kcore", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(357)
        old = module.ConditionalGraphBetaDiffusionH1(config(), _DatasetInfo()).eval()
        old_rng = torch.get_rng_state()
        torch.manual_seed(357)
        new = ConditionalGraphBetaDiffusionH1(config(False), _DatasetInfo()).eval()
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        new.on_load_checkpoint(checkpoint(old))
        new.load_state_dict(old.state_dict(), strict=True)
        x, e, y, t, mask = noisy_clique(old)
        eta_x, eta_e = old._eta_tensors(x.size(0), x.size(1), x)
        args = (x, e, y, t, old._alpha(t), eta_x, eta_e, mask)
        for a, b in zip(old._predict_clean(*args), new._predict_clean(*args)):
            self.assertTrue(torch.equal(a, b))

    def test_train_validation_checkpoint_and_reverse_sample(self):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        from torch.utils.data import DataLoader
        from torch_geometric.data import Batch, Data

        torch.manual_seed(358)
        pairs = [(i, j) for i in range(4) for j in range(4) if i != j]
        graph = Data(x=torch.ones(4, 1), edge_index=torch.tensor(pairs).T.contiguous(),
                     edge_attr=torch.tensor([[0., 1.]]).repeat(len(pairs), 1),
                     y=torch.tensor([[1., 0.]]))
        loader = DataLoader([graph] * 2, batch_size=2, collate_fn=Batch.from_data_list)
        with tempfile.TemporaryDirectory(prefix="h1_kcore_synthetic_") as directory:
            model = ConditionalGraphBetaDiffusionH1(config(True), _DatasetInfo())
            cb = ModelCheckpoint(dirpath=directory, monitor="val/h1_total",
                                 mode="min", save_top_k=1, save_last=True)
            trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1,
                              num_sanity_val_steps=0, logger=False,
                              enable_progress_bar=False, enable_model_summary=False,
                              callbacks=[cb])
            trainer.fit(model, loader, loader)
            for key in ("train/h1_total", "val/h1_total"):
                self.assertTrue(torch.isfinite(trainer.callback_metrics[key]))
            restored = model.load_from_checkpoint(cb.last_model_path, strict=True).eval()
            self.assertEqual(restored.kcore_options, model.kcore_options)
            y = torch.tensor([[1., 0.], [0., 1.]])
            graphs, _ = restored.sampling_batch_h1(
                batch_id=0, batch_size=2, keep_chain=0, number_chain_steps=1,
                num_nodes=torch.tensor([1, 4]), y=y, figures=False)
            for n, (x, e) in zip((1, 4), graphs):
                self.assertEqual(tuple(x.shape), (n,))
                self.assertEqual(tuple(e.shape), (n, n))
                self.assertTrue(torch.equal(e, e.T))
                self.assertEqual(int(e.diagonal().count_nonzero()), 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
