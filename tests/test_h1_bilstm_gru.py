import copy
from pathlib import Path
import tempfile
import unittest

import pytorch_lightning as pl
import torch

from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1
from models.bilstm_edge import BiLSTMEdgeUpdate
from test_beta_h1_semantics import _DatasetInfo, _cfg


class BiLSTMGRUTests(unittest.TestCase):
    def test_relabeling_padding_and_gradients(self):
        torch.manual_seed(29)
        model = BiLSTMEdgeUpdate(8, 5, 4).double()
        model.PAIR_CHUNK = 3
        edges = torch.randn(3, 7, 7, 8, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1] * 7, [1, 0, 1, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0]]).bool()
        global_features = torch.zeros(3, 5, dtype=edges.dtype)
        # Allow gradients to reach the paths despite the default zero MLP output.
        with torch.no_grad():
            model.output[-1].weight.normal_(std=.1)
        output = model(edges, global_features, mask)
        perm = torch.tensor([3, 5, 0, 6, 4, 1, 2])
        relabeled = model(edges[:, perm][:, :, perm], global_features, mask[:, perm])
        torch.testing.assert_close(relabeled, output[:, perm][:, :, perm], rtol=1e-10, atol=1e-12)
        self.assertTrue(torch.equal(output, output.transpose(1, 2)))
        valid = mask[:, None, :] & mask[:, :, None] & ~torch.eye(7, dtype=torch.bool)
        self.assertFalse(output[~valid].any())
        output.square().mean().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertTrue(parameter.grad.abs().max() > 0, name)

    def test_model_checkpoint_modes(self):
        def make(mode, aggregation):
            cfg = _cfg()
            cfg.model.two_hop = dict(enabled=True, channels=4, normalization=aggregation, gate_mode=mode)
            cfg.model.hog_rw = dict(enabled=True, rw_depth=4)
            return ConditionalGraphBetaDiffusionH1(cfg, _DatasetInfo())

        gru = make('gru', 'mean')
        bilstm = make('bilstm_gru', 'bilstm')
        with tempfile.TemporaryDirectory() as temp:
            for model, version in ((gru, 4), (bilstm, 5)):
                saved = dict(state_dict=model.state_dict(), hyper_parameters=dict(model.hparams),
                             **{'pytorch-lightning_version': pl.__version__})
                model.on_save_checkpoint(saved)
                self.assertEqual(saved['h1_two_hop']['version'], version)
                path = Path(temp)/f'model-{version}.ckpt'
                torch.save(saved, path)
                restored = type(model).load_from_checkpoint(path, map_location='cpu', strict=True)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
                other = bilstm if model is gru else gru
                with self.assertRaises(RuntimeError):
                    other.on_load_checkpoint(copy.deepcopy(saved))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
