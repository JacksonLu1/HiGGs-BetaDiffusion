"""Tiny real Glue reverse-sampler CPU integration; not the server failure case."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1] / 'HiGGs-main'
sys.path[:0] = [str(HERE), str(CHECKOUT / 'dgd'),
               str(HERE.parent / 'four_benchmarks_beta_20261101/payload/dgd')]


class GlueIntegration(unittest.TestCase):
    def test_real_reverse_sampler_completes_forced_solver_fallback(self):
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from datasets.abstract_dataset import AbstractDatasetInfos
        from diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
        from four_benchmark_models import FourBenchmarkGlue
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from stable_eigen import make_guarded_eigenfeatures
        torch.set_num_threads(1)
        with initialize_config_dir(version_base='1.1', config_dir=str(CHECKOUT / 'configs')):
            cfg = compose(config_name='config', overrides=['dataset=sbm', 'model=original_glue'])
        OmegaConf.set_struct(cfg, False)
        cfg.dataset.name = 'ego'
        cfg.dataset.h = 1.5
        cfg.general.name = 'synthetic_ego_eigh_guard'
        cfg.model.diffusion_steps = 4  # Software test only; real Glue remains400.
        data = Data(x=torch.ones((4, 1)), y=torch.tensor([[0., 1.]]),
                    edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
                    edge_attr=torch.tensor([[0., 1.]]).repeat(4, 1))
        side = torch.tensor([0, 0, 1, 1])
        data.pos = side[:, None] == side[None, :]
        loader = DataLoader([data], batch_size=1)
        info = AbstractDatasetInfos()
        info.name = 'ego'
        info.n_nodes = torch.tensor([0., 0., 0., 0., 1.])
        info.node_types = torch.tensor([1.])
        info.edge_types = torch.tensor([.6, .4])
        info.complete_infos(info.n_nodes, info.node_types)
        extra, domain = ExtraFeatures('all', info), DummyExtraFeatures()
        info.compute_input_output_dims(SimpleNamespace(train_dataloader=lambda: loader), extra, domain)
        model = FourBenchmarkGlue(cfg, info, TrainAbstractMetricsDiscrete(),
            TrainAbstractMetricsDiscrete(), None, extra, domain)
        model.eval()
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        events = []
        with tempfile.TemporaryDirectory(dir=HERE) as folder:
            guard = make_guarded_eigenfeatures(model.extra_features.eigenfeatures,
                                               events.append, diagnostic_dir=folder)
            def nonconverge(matrix, UPLO='L'):
                raise torch._C._LinAlgError('linalg.eigh: The algorithm failed to converge')
            guard.solver.primary_solver = nonconverge
            model.extra_features.eigenfeatures = guard
            with torch.no_grad():
                generated, graphs = model.sampling_batch_glue(
                    0, 1, 0, 2, data=next(iter(loader)), figures=False)
            self.assertEqual(len(graphs), 1)
            self.assertEqual(len(graphs[0]), 4)
            adjacency = generated[0][1]
            self.assertTrue(torch.equal(adjacency, adjacency.T))
            self.assertEqual(int(adjacency[0, 1]), 1)
            self.assertEqual(int(adjacency[2, 3]), 1)
            self.assertGreaterEqual(len(events), 4)
            self.assertTrue(all(e['event'] == 'eigh_cpu_float64_fallback' for e in events))
            self.assertTrue(all(Path(e['matrix_file']).is_file() for e in events))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in weights.items()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
