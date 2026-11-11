"""Regress the real Glue reverse sampler's Boolean-mask contract on CPU.

The tiny diffusion horizon is a synthetic software test only. No checkpoint,
production source, server process or real-data training is changed.
"""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1] / 'HiGGs-main'
OLD_PACKAGE = HERE
sys.path[:0] = [str(CHECKOUT / 'dgd'), str(OLD_PACKAGE / 'payload/tools')]

import torch
import tg_beta_sample as legacy



def load_interface():
    spec = importlib.util.spec_from_file_location('glue_bool_regression_interface', CHECKOUT / 'higgs/interface_ablation.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def line_graph(n):
    value = torch.zeros((n, n), dtype=torch.long)
    for i in range(n - 1):
        value[i, i + 1] = value[i + 1, i] = 1
    return value


def prepared():
    return legacy.prepare_pairs(
        SimpleNamespace(dataset=SimpleNamespace(name='tree'), general=SimpleNamespace(gpus=0)),
        SimpleNamespace(Xdim_output=1, Edim_output=2, ydim_output=2),
        [line_graph(n) for n in (2, 3, 4)],
        [torch.zeros(n, dtype=torch.long) for n in (2, 3, 4)],
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        torch.ones(4, dtype=torch.long), 1, True,
        interface=load_interface(), cache_audit={'one_hot_histogram': {'0,1': 64}})


class RealGlueSamplerTests(unittest.TestCase):
    def test_real_reverse_sampler_old_fails_fixed_succeeds(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from datasets.abstract_dataset import AbstractDatasetInfos
        from diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
        from diffusion_model_discrete import DiscreteDenoisingDiffusion
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='glue_bool_', dir=HERE) as directory:
            config_dir = Path(directory) / 'configs'
            shutil.copytree(CHECKOUT / 'configs', config_dir)
            with initialize_config_dir(version_base='1.1', config_dir=str(config_dir)):
                cfg = compose(config_name='config', overrides=['dataset=sbm', 'model=original_glue'])
            OmegaConf.set_struct(cfg, False)
            cfg.dataset.name = 'tree'
            cfg.dataset.h = 1.5
            cfg.general.name = 'synthetic_glue_bool_regression'
            cfg.general.number_chain_steps = 2
            cfg.model.diffusion_steps = 4  # Test only; formal Glue remains 400.
            infos = AbstractDatasetInfos()
            infos.name = 'tree'
            infos.n_nodes = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1.])
            infos.node_types = torch.tensor([1.])
            infos.edge_types = torch.tensor([0.6, 0.4])
            infos.complete_infos(infos.n_nodes, infos.node_types)
            result = prepared()
            loader = result[1]
            dm = SimpleNamespace(train_dataloader=lambda: loader)
            extra = ExtraFeatures('all', infos)
            domain = DummyExtraFeatures()
            infos.compute_input_output_dims(dm, extra, domain)
            model = DiscreteDenoisingDiffusion(cfg=cfg, dataset_infos=infos,
                train_metrics=TrainAbstractMetricsDiscrete(), sampling_metrics=TrainAbstractMetricsDiscrete(),
                visualization_tools=None, extra_features=extra, domain_features=domain)
            model.eval()
            self.assertEqual((model.Xdim_output, model.Edim_output, model.ydim_output), (1, 2, 2))
            parameters_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            old_batch = next(iter(loader))
            self.assertEqual(old_batch.pos.dtype, torch.bool)
            old_batch.pos = old_batch.pos.float()
            self.assertEqual(old_batch.pos.dtype, torch.float32)
            with torch.no_grad(), self.assertRaisesRegex(TypeError, 'invert|Boolean-type'):
                model.sampling_batch_glue(0, 1, 0, 2, data=old_batch, figures=False)
            batch = next(iter(loader))
            self.assertEqual(batch.pos.dtype, torch.bool)
            with torch.no_grad():
                generated, nx_graphs = model.sampling_batch_glue(0, 1, 0, 2, data=batch, figures=False)
            self.assertEqual(len(generated), 1)
            self.assertEqual(len(nx_graphs), 1)
            self.assertEqual(len(nx_graphs[0]), 5)
            self.assertEqual(tuple(generated[0][1].shape), (5, 5))
            expected = torch.zeros((5, 5), dtype=torch.long)
            expected[:2, :2] = line_graph(2)
            expected[2:, 2:] = line_graph(3)
            membership = (torch.arange(5) < 2)[:, None] == (torch.arange(5) < 2)[None, :]
            self.assertTrue(torch.equal(generated[0][1][membership], expected[membership]))
            self.assertTrue(torch.equal(generated[0][1], generated[0][1].T))
            self.assertTrue(all(torch.equal(before, model.state_dict()[key])
                                for key, before in parameters_before.items()))
            print('PASS: real Glue reverse sampler reproduces float-mask TypeError; Boolean fix completes; intra-edges and weights unchanged')


if __name__ == '__main__':
    unittest.main(verbosity=2)


