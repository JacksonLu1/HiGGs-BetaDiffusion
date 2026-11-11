"""Synthetic CPU model tests, not a server/real-data training claim."""
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1] / 'HiGGs-main'
HOG = HERE.parent / 'h1_twohop_mean_hog_rw_20261022/payload'


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='four_cpu_', dir=HERE)
        cls.root = Path(cls.temp.name)
        shutil.copytree(CHECKOUT / 'configs', cls.root / 'configs')
        shutil.copytree(CHECKOUT / 'dgd', cls.root / 'dgd',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for relative in ('dgd/diffusion_model_beta_h1.py', 'dgd/diffusion/hog_rw_features.py',
                         'configs/model/beta_h1_hog_rw.yaml', 'configs/model/beta_h1_twohop_mean_hog_rw.yaml'):
            shutil.copy2(HOG / relative, cls.root / relative)
        shutil.copy2(HERE / 'payload/dgd/four_benchmark_models.py', cls.root / 'dgd/four_benchmark_models.py')
        sys.path[:0] = [str(cls.root / 'dgd'), str(HERE / 'payload/tools')]
        import torch
        import wandb
        torch.set_num_threads(1)
        torch.manual_seed(0)
        wandb.init(project='synthetic-four-benchmark-test', mode='disabled')

    @classmethod
    def tearDownClass(cls):
        import wandb
        wandb.finish()
        cls.temp.cleanup()

    def build(self, stage, n=4):
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from datasets.abstract_dataset import AbstractDatasetInfos
        from diffusion.distributions import DistributionNodes
        from diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
        from four_benchmark_models import model_class
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        group = {'h2': 'beta_h2', 'h1': 'beta_h1_twohop_mean_hog_rw', 'glue': 'original_glue'}[stage]
        with initialize_config_dir(version_base='1.1', config_dir=str(self.root / 'configs')):
            cfg = compose(config_name='config', overrides=['dataset=sbm', 'model=' + group])
        OmegaConf.set_struct(cfg, False)
        cfg.dataset.name = 'ego_small'
        cfg.dataset.h = {'h2': 2, 'h1': 1, 'glue': 1.5}[stage]
        cfg.general.name = 'synthetic_four_models'
        cfg.general.sample_every_val = 10**9
        cfg.general.number_chain_steps = 2
        cfg.model.diffusion_steps = 4  # Test only; formal horizons remain 100/100/400.
        xdim = 2 if stage == 'h2' else 1
        nodes = torch.zeros(n, dtype=torch.long)
        edges = [(u, v) for i in range(n - 1) for u, v in ((i, i + 1), (i + 1, i))]
        data = Data(x=torch.nn.functional.one_hot(nodes, xdim).float(),
                    edge_index=torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous(),
                    edge_attr=torch.tensor([[0., 1.]]).repeat(len(edges), 1),
                    y=torch.tensor([[0., 1.]]) if stage == 'glue' else torch.tensor([[1., 0.]]))
        if stage == 'glue':
            side = torch.arange(n) < max(1, n // 2)
            data.pos = side[:, None] == side[None, :]
        loader = DataLoader([data, data.clone()], batch_size=2)
        infos = AbstractDatasetInfos()
        infos.name = 'ego_small'
        infos.n_nodes = torch.zeros(max(4, n) + 1)
        infos.n_nodes[n] = 1
        infos.node_types = torch.tensor([1., 0.]) if xdim == 2 else torch.tensor([1.])
        infos.edge_types = torch.tensor([0.6, 0.4])
        infos.complete_infos(infos.n_nodes, infos.node_types)
        infos.nodes_dist = DistributionNodes(infos.n_nodes)
        extra = DummyExtraFeatures() if stage == 'h2' else ExtraFeatures('all', infos)
        domain = DummyExtraFeatures()
        infos.compute_input_output_dims(SimpleNamespace(train_dataloader=lambda: loader), extra, domain)
        model = model_class(stage)(cfg=cfg, dataset_infos=infos,
                    train_metrics=TrainAbstractMetricsDiscrete(),
                    sampling_metrics=TrainAbstractMetricsDiscrete(),
                    visualization_tools=None, extra_features=extra, domain_features=domain)
        return model, next(iter(loader))

    def check_backward(self, model, batch):
        import torch
        result = model.training_step(batch, 0)
        loss = result['loss'] if isinstance(result, dict) else result
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        return loss

    def test_all_stages_finite_training_and_strict_reload(self):
        import torch
        import pytorch_lightning as pl
        from four_benchmark_models import model_class
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        for stage in ('h2', 'h1', 'glue'):
            with self.subTest(stage=stage):
                model, batch = self.build(stage)
                self.check_backward(model, batch)
                checkpoint = {'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams),
                              'pytorch-lightning_version': pl.__version__, 'epoch': 0, 'global_step': 1}
                model.on_save_checkpoint(checkpoint)
                path = self.root / (stage + '.ckpt')
                torch.save(checkpoint, path)
                loaded = model_class(stage).load_from_checkpoint(str(path), map_location='cpu', strict=True,
                    train_metrics=TrainAbstractMetricsDiscrete(), sampling_metrics=TrainAbstractMetricsDiscrete())
                self.assertEqual(model.state_dict().keys(), loaded.state_dict().keys())
                self.assertTrue(all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items()))
                if stage == 'h1':
                    self.assertEqual(checkpoint['h1_two_hop']['normalization'], 'mean')
                    self.assertEqual(checkpoint['h1_hog_rw']['rw_depth'], 4)
                    self.assertEqual({k: getattr(loaded.model, 'mlp_in_' + k)[0].in_features for k in ('X', 'E', 'y')},
                                     {'X': 12, 'E': 6, 'y': 14})

    def test_singleton_h2_h1_backward_and_sampling(self):
        import torch
        for stage in ('h2', 'h1'):
            with self.subTest(stage=stage):
                model, batch = self.build(stage, n=1)
                loss = self.check_backward(model, batch)
                self.assertTrue(torch.isfinite(loss))
                model.eval()
                with torch.no_grad():
                    if stage == 'h2':
                        predicted, graphs = model.sampling_batch_h2(0, 2, 0, 2, 0, False, num_nodes=1)
                    else:
                        predicted, graphs = model.sampling_batch_h1(0, 2, 0, 2,
                            y=torch.tensor([[1., 0.]]).repeat(2, 1), num_nodes=1, figures=False)
                self.assertEqual(len(graphs), 2)
                self.assertTrue(all(len(g) == 1 and g.number_of_edges() == 0 for g in graphs))
                self.assertTrue(all(tuple(row[1].shape) == (1, 1) for row in predicted))

    def test_non_singleton_forward_is_bit_identical_and_padding_has_no_parameters(self):
        import torch
        from four_benchmark_models import adapt_backbone
        from models.transformer_model import GraphTransformer
        model, _ = self.build('h2')
        backbone = model.model
        backbone.eval()
        X = torch.randn(2, 3, 2)
        E = torch.randn(2, 3, 3, 1)
        E = (E + E.transpose(1, 2)) / 2
        y = torch.randn(2, 3)
        mask = torch.ones(2, 3, dtype=torch.bool)
        with torch.no_grad():
            actual = backbone(X, E, y, mask)
            expected = GraphTransformer.forward(backbone, X, E, y, mask)
        self.assertTrue(all(torch.equal(getattr(actual, k), getattr(expected, k)) for k in ('X', 'E', 'y')))
        backbone.__class__ = GraphTransformer
        before = {k: v.clone() for k, v in backbone.state_dict().items()}
        rng = torch.get_rng_state().clone()
        adapt_backbone(backbone)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before.keys(), backbone.state_dict().keys())
        self.assertTrue(all(torch.equal(v, backbone.state_dict()[k]) for k, v in before.items()))

    def test_glue_empty_edge_ce_is_zero_and_nonempty_unchanged(self):
        import torch
        from four_benchmark_models import EmptyEdgeCrossEntropy
        from metrics.abstract_metrics import CrossEntropyMetric
        empty = EmptyEdgeCrossEntropy()
        value = empty(torch.empty(0, 2), torch.empty(0, 2))
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(float(value), 0.)
        pred = torch.tensor([[0., 1.], [2., 1.]])
        true = torch.tensor([[0., 1.], [1., 0.]])
        self.assertTrue(torch.equal(EmptyEdgeCrossEntropy()(pred, true), CrossEntropyMetric()(pred, true)))

    def test_actual_glue_reverse_sampler_boolmask_preserves_internal_edges(self):
        import torch
        model, batch = self.build('glue', n=4)
        # This sampler is deliberately batch-size one, as in the formal protocol.
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        one = Data(x=batch.x[:4], y=batch.y[:1],
                   edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
                   edge_attr=torch.tensor([[0., 1.]]).repeat(4, 1))
        side = torch.tensor([0, 0, 1, 1])
        one.pos = side[:, None] == side[None, :]
        data = next(iter(DataLoader([one], batch_size=1)))
        model.eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            generated, graphs = model.sampling_batch_glue(0, 1, 0, 2, data=data, figures=False)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(len(graphs[0]), 4)
        edges = generated[0][1]
        self.assertTrue(torch.equal(edges, edges.T))
        self.assertEqual(int(edges[0, 1]), 1)
        self.assertEqual(int(edges[2, 3]), 1)
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))

    def test_diffusion_steps_are_inherited_without_changes(self):
        from four_benchmark_models import FourBenchmarkH1, FourBenchmarkH2, FourBenchmarkGlue
        from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1
        from diffusion_model_beta_h2 import ConditionalGraphBetaDiffusionH2
        from diffusion_model_discrete import DiscreteDenoisingDiffusion
        self.assertIs(FourBenchmarkH1._predict_clean, ConditionalGraphBetaDiffusionH1._predict_clean)
        self.assertIs(FourBenchmarkH2._predict_clean, ConditionalGraphBetaDiffusionH2._predict_clean)
        self.assertIs(FourBenchmarkGlue.sampling_batch_glue, DiscreteDenoisingDiffusion.sampling_batch_glue)


if __name__ == '__main__':
    unittest.main(verbosity=2)
