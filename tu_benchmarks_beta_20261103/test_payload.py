"""Synthetic CPU model tests, not a server/real-data training claim."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
CHECKOUT = HERE.parents[1] / 'HiGGs-main'
HOG = HERE.parent / 'h1_twohop_mean_hog_rw_20261022/payload'
ORIGINAL = Path('C:/path/to/HiGGs-main1/HiGGs-main/deliverables/tu_benchmarks_20261102/files')
sys.path.insert(0, str(HERE / 'payload/tools'))
import tu_beta_common as common
import tu_beta_train as driver


class ProtocolTests(unittest.TestCase):
    def test_original_frozen_identity_and_all_counts(self):
        for name, spec in common.SPECS.items():
            manifest = ORIGINAL / f'data_sources/splits/{name}_r1_s1234_l0.json'
            config = ORIGINAL / f'configs/dataset/{name}.yaml'
            self.assertEqual(common.sha(manifest), spec['manifest_sha256'])
            self.assertEqual(common.sha(config), spec['config_sha256'])
            frozen = json.loads(manifest.read_text())
            common.validate_manifest(name, frozen)
            records = {r['id']: r for r in frozen['graphs']}
            counts = {'2': [], '1': [], '1.5': []}
            for split in ('train', 'val', 'test'):
                communities = pairs = 0
                for graph_id in frozen['splits'][split]:
                    parts = frozen['partitions'][str(graph_id)]
                    owner = {v: i for i, part in enumerate(parts) for v in part}
                    communities += len(parts)
                    pairs += len({tuple(sorted((owner[a], owner[b])))
                                  for a, b in records[graph_id]['edges'] if owner[a] != owner[b]})
                counts['2'].append(len(frozen['splits'][split]))
                counts['1'].append(communities)
                counts['1.5'].append(pairs)
            self.assertEqual(counts, spec['cache_counts'])
            self.assertEqual({h: c[0] for h, c in counts.items()}, spec['train_samples'])

    def test_sampling_seed_and_component_policies_match_original_actual_sampler(self):
        for spec in common.SPECS.values():
            self.assertEqual(spec['sampling_seed'], 0)
            self.assertEqual(spec['component_policy'], {'h2': 'upstream_lcc', 'h1': 'upstream_lcc'})
        source = (HERE / 'payload/tools/tu_beta_sample.py').read_text()
        self.assertIn('interface.seed_sampling(SPECS[args.dataset]', source)
        self.assertIn('"sampling_seed": SPECS[args.dataset]', source)
        self.assertNotIn('seed_sampling(1234)', source)
        # Only hierarchy creation uses1234; assembly must not retain full H2/H1.
        self.assertNotIn('"h2_component_policy": "full"', source)
        self.assertNotIn('"h1_component_policy": "full"', source)

    def test_upstream_lcc_ties_match_original_numpy_semantics(self):
        import importlib.util
        import numpy as np
        import scipy.sparse as sp
        import torch
        spec = importlib.util.spec_from_file_location('tu_test_interface', CHECKOUT / 'higgs/interface_ablation.py')
        interface = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interface)
        for n, edges in ((1, []), (4, [(0, 1), (2, 3)]), (4, []), (5, [(0, 1), (1, 2), (3, 4)])):
            adjacency = torch.zeros(n, n, dtype=torch.long)
            for left, right in edges:
                adjacency[left, right] = adjacency[right, left] = 1
            nodes = torch.arange(n)
            _, components = sp.csgraph.connected_components(adjacency.numpy())
            _, counts = np.unique(components, return_counts=True)
            mask = torch.as_tensor(np.isin(components, counts.argsort()[-1:]))
            used_nodes, used_edges = interface.apply_upstream_lcc(nodes, adjacency)
            self.assertTrue(torch.equal(used_nodes, nodes[mask]))
            self.assertTrue(torch.equal(used_edges, adjacency[mask][:, mask]))


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='tu_config_', dir=HERE)
        cls.root = Path(cls.temp.name)
        shutil.copytree(CHECKOUT / 'configs', cls.root / 'configs')
        for name in common.SPECS:
            shutil.copy2(ORIGINAL / f'configs/dataset/{name}.yaml', cls.root / f'configs/dataset/{name}.yaml')
        for source in (HOG / 'configs/model').glob('*.yaml'):
            shutil.copy2(source, cls.root / 'configs/model' / source.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_dataset_stage_and_mode_configs(self):
        for dataset, spec in common.SPECS.items():
            item = {'manifest': str(self.root / f'{dataset}.json'),
                    'source': str(self.root / f'{dataset}_source.json'),
                    'datadir': str(self.root / 'data' / dataset),
                    'splits': spec['splits'], 'resolution': 1, 'max_size': spec['max_size']}
            for stage in ('h2', 'h1', 'glue'):
                for mode in ('smoke', 'train'):
                    with self.subTest(dataset=dataset, stage=stage, mode=mode):
                        cfg = driver.make_config(self.root, dataset, stage, mode, self.root / 'outputs', item)
                        self.assertEqual(cfg.train.batch_size, spec['batch_size'][stage])
                        self.assertEqual(cfg.train.seed, 0)
                        self.assertEqual(cfg.train.n_epochs, 1 if mode == 'smoke' else spec['epochs'])
                        self.assertEqual(cfg.model.diffusion_steps, 400 if stage == 'glue' else 100)
                        self.assertEqual(cfg.general.check_val_every_n_epochs, 1 if mode == 'smoke' else 10)
                        self.assertIsNone(cfg.general.resume)
                        self.assertIsNone(cfg.general.test_only)
                        self.assertFalse(cfg.general.run_test_after_fit)
                        if stage == 'h1':
                            self.assertTrue(cfg.model.hog_rw.enabled)
                            self.assertEqual(cfg.model.hog_rw.rw_depth, 4)
                            self.assertEqual(cfg.model.two_hop.normalization, 'mean')
                            self.assertEqual(cfg.model.two_hop.channels, 16)
                            self.assertIsNone(cfg.model.two_hop.warm_start_checkpoint)
                        elif stage == 'glue':
                            self.assertEqual(cfg.model.family, 'original_glue')


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='tu_cpu_', dir=HERE)
        cls.root = Path(cls.temp.name)
        cls.saved_sys_path = list(sys.path)
        cls.saved_module_names = set(sys.modules)
        cls.saved_package_paths = {}
        shutil.copytree(CHECKOUT / 'configs', cls.root / 'configs')
        shutil.copytree(CHECKOUT / 'dgd', cls.root / 'dgd',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for relative in ('dgd/diffusion_model_beta_h1.py', 'dgd/diffusion/hog_rw_features.py',
                         'configs/model/beta_h1_hog_rw.yaml', 'configs/model/beta_h1_twohop_mean_hog_rw.yaml'):
            shutil.copy2(HOG / relative, cls.root / relative)
        shutil.copy2(HERE / 'payload/dgd/tu_benchmark_models.py', cls.root / 'dgd/tu_benchmark_models.py')
        shutil.copy2(HERE / 'payload/dgd/datasets/tu_benchmark_beta.py',
                     cls.root / 'dgd/datasets/tu_benchmark_beta.py')
        for name in common.SPECS:
            shutil.copy2(ORIGINAL / f'configs/dataset/{name}.yaml', cls.root / f'configs/dataset/{name}.yaml')
        sys.path[:0] = [str(cls.root / 'dgd'), str(HERE / 'payload/tools')]
        # Earlier installer tests intentionally imported the production data
        # adapter read-only. Its cached packages keep production __path__, so
        # sys.path alone does not expose HOG/adapter files in this CPU fixture.
        # Extend only these cached package search paths for this test class and
        # restore them below; installed production files are never changed.
        for package_name, relative in (('diffusion', 'dgd/diffusion'), ('datasets', 'dgd/datasets')):
            package = sys.modules.get(package_name)
            if package is not None:
                cls.saved_package_paths[package_name] = list(package.__path__)
                package.__path__ = [str(cls.root / relative), *package.__path__]
        import torch
        import wandb
        torch.set_num_threads(1)
        torch.manual_seed(0)
        wandb.init(project='synthetic-tu-benchmark-test', mode='disabled')

    @classmethod
    def tearDownClass(cls):
        import wandb
        wandb.finish()
        for name, paths in cls.saved_package_paths.items():
            sys.modules[name].__path__ = paths
        sys.path[:] = cls.saved_sys_path
        # Do not leave modules whose __file__ points into the soon-deleted
        # disposable fixture available to later tests in combined discovery.
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if (name not in cls.saved_module_names and path
                    and Path(path).resolve().is_relative_to(cls.root.resolve())):
                sys.modules.pop(name, None)
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
        from tu_benchmark_models import model_class
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        group = {'h2': 'beta_h2', 'h1': 'beta_h1_twohop_mean_hog_rw', 'glue': 'original_glue'}[stage]
        with initialize_config_dir(version_base='1.1', config_dir=str(self.root / 'configs')):
            cfg = compose(config_name='config', overrides=['dataset=sbm', 'model=' + group])
        OmegaConf.set_struct(cfg, False)
        cfg.dataset.name = 'imdb_binary'
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
        infos.name = 'imdb_binary'
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
        from tu_benchmark_models import model_class
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
        from tu_benchmark_models import adapt_backbone
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
        from tu_benchmark_models import EmptyEdgeCrossEntropy
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
        from tu_benchmark_models import TUBenchmarkH1, TUBenchmarkH2, TUBenchmarkGlue
        from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1
        from diffusion_model_beta_h2 import ConditionalGraphBetaDiffusionH2
        from diffusion_model_discrete import DiscreteDenoisingDiffusion
        self.assertIs(TUBenchmarkH1._predict_clean, ConditionalGraphBetaDiffusionH1._predict_clean)
        self.assertIs(TUBenchmarkH2._predict_clean, ConditionalGraphBetaDiffusionH2._predict_clean)
        self.assertIs(TUBenchmarkGlue.sampling_batch_glue, DiscreteDenoisingDiffusion.sampling_batch_glue)

    def test_real_frozen_record_subsets_train_validate_reload_all_three_by_three(self):
        """Nine tiny CPU fits on actual source records, NOT complete caches/formal training."""
        import torch
        from torch_geometric.data import Data, InMemoryDataset
        from pytorch_lightning import Trainer, seed_everything
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger
        from datasets.tu_benchmark_beta import FrozenCollectionDataModule, TUBenchmarkBetaInfos
        from tu_benchmark_models import model_class
        from diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        for name, spec in common.SPECS.items():
            manifest = ORIGINAL / f'data_sources/splits/{name}_r1_s1234_l0.json'
            frozen = json.loads(manifest.read_text())
            records = {r['id']: r for r in frozen['graphs']}
            cache_root = self.root / 'actual_record_subsets' / name
            processed = cache_root / 'processed'
            processed.mkdir(parents=True)
            selected = {}
            for split in ('train', 'val', 'test'):
                graph_id = next(gid for gid in frozen['splits'][split]
                                if 12 <= records[gid]['n'] <= 40
                                and 2 <= len(frozen['partitions'][str(gid)]) <= 8)
                selected[split] = graph_id
                record, parts = records[graph_id], frozen['partitions'][str(graph_id)]
                member = {v: i for i, part in enumerate(parts) for v in part}
                links = sorted({tuple(sorted((member[a], member[b])))
                                for a, b in record['edges'] if member[a] != member[b]})
                self.assertTrue(links)
                for h in ('2', '1', '1.5'):
                    samples = []
                    if h == '2':
                        subsets = [(list(range(len(parts))), links, None)]
                    elif h == '1':
                        subsets = [(part, record['edges'], i) for i, part in enumerate(parts)]
                    else:
                        subsets = [(parts[a] + parts[b], record['edges'], len(parts[a])) for a, b in links]
                    for vertices, source_edges, info in subsets:
                        order = {v: i for i, v in enumerate(vertices)}
                        undirected = [(order[a], order[b]) for a, b in source_edges if a in order and b in order]
                        directed = sorted(undirected + [(b, a) for a, b in undirected])
                        width = 2 if h == '2' else 1
                        data = Data(x=torch.nn.functional.one_hot(torch.zeros(len(vertices), dtype=torch.long), width).float(),
                                    edge_index=torch.tensor(directed, dtype=torch.long).reshape(-1, 2).t().contiguous(),
                                    edge_attr=torch.tensor([[0., 1.]]).repeat(len(directed), 1),
                                    y=torch.tensor([[0., 1.]]) if h == '1.5' else torch.tensor([[1., 0.]]),
                                    idx=len(samples), source_graph_id=torch.tensor([graph_id]))
                        if h == '1':
                            data.community_id = torch.tensor([info])
                        elif h == '1.5':
                            side = torch.arange(len(vertices)) < info
                            data.pos = side[:, None] == side[None, :]
                        samples.append(data)
                    if h == '1.5':
                        maximum = max(d.num_nodes for d in samples)
                        for data in samples:
                            data.pos = torch.nn.functional.pad(data.pos,
                                (0, maximum-data.num_nodes, 0, maximum-data.num_nodes), value=False)
                    path = processed / f'proc_{split}_v1_h{h}_max{spec["max_size"]}_{spec["manifest_sha256"]}.pt'
                    torch.save(InMemoryDataset.collate(samples), path)
            item = {'datadir': str(cache_root), 'source': str(self.root / 'unused_source.json'),
                    'manifest': str(manifest), 'splits': spec['splits'], 'resolution': 1,
                    'max_size': spec['max_size']}
            for stage in ('h2', 'h1', 'glue'):
                with self.subTest(dataset=name, stage=stage):
                    seed_everything(0, workers=True)
                    output = self.root / 'real_subset_cpu_fits' / name / stage
                    cfg = driver.make_config(self.root, name, stage, 'smoke', output, item)
                    cfg.train.progress_bar = False
                    dm = FrozenCollectionDataModule(cfg)
                    dm.prepare_data()
                    self.assertTrue(all(int(d.source_graph_id.item()) == selected['train']
                                        for d in dm.dataloaders['train'].dataset))
                    infos = TUBenchmarkBetaInfos(dm, cfg.dataset)
                    extra = DummyExtraFeatures() if stage == 'h2' else ExtraFeatures('all', infos)
                    domain = DummyExtraFeatures()
                    infos.compute_input_output_dims(dm, extra, domain)
                    klass = model_class(stage)
                    model = klass(cfg=cfg, dataset_infos=infos,
                        train_metrics=TrainAbstractMetricsDiscrete(), sampling_metrics=TrainAbstractMetricsDiscrete(),
                        visualization_tools=None, extra_features=extra, domain_features=domain)
                    callback = ModelCheckpoint(dirpath=str(output / 'checkpoints'),
                        monitor=common.STAGES[stage][2], mode='min', save_top_k=1, save_last=True)
                    trainer = Trainer(accelerator='cpu', devices=1, max_epochs=1,
                        limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0,
                        enable_model_summary=False, enable_progress_bar=False, log_every_n_steps=1,
                        logger=CSVLogger(str(output / 'logs'), name='metrics'), callbacks=[callback])
                    trainer.fit(model, datamodule=dm)
                    self.assertEqual(trainer.global_step, 1)
                    self.assertTrue(torch.isfinite(callback.best_model_score))
                    loaded = klass.load_from_checkpoint(callback.best_model_path, map_location='cpu', strict=True,
                        train_metrics=TrainAbstractMetricsDiscrete(), sampling_metrics=TrainAbstractMetricsDiscrete())
                    self.assertEqual(loaded.dataset_info.manifest_sha256, spec['manifest_sha256'])
                    self.assertTrue(all(torch.isfinite(p).all() for p in loaded.parameters()))
                    self.assertEqual(model.state_dict().keys(), loaded.state_dict().keys())
                    self.assertTrue(all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items()))
                    print('REAL_RECORD_SUBSET_CPU_SMOKE', name, stage, 'steps=1 validation=1 strict_reload=True')


if __name__ == '__main__':
    unittest.main(verbosity=2)
