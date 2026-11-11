"""Tiny synthetic CPU forward/backward and strict reload, not a real-data smoke."""
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / 'payload/tools'))
import sc_beta_train as driver


def main():
    with tempfile.TemporaryDirectory(prefix='cpu_models_', dir=PACKAGE) as directory:
        root = Path(directory)
        checkout = PACKAGE.parents[1] / 'HiGGs-main'
        shutil.copytree(checkout / 'configs', root / 'configs')
        (root / 'dgd').mkdir()
        for name in ('utils.py',):
            shutil.copy2(checkout / 'dgd' / name, root / 'dgd' / name)
        for name in ('diffusion', 'models', 'metrics'):
            shutil.copytree(checkout / 'dgd' / name, root / 'dgd' / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('diffusion_model_beta_h1.py', 'diffusion_model_beta_h2.py', 'diffusion_model_discrete.py'):
            shutil.copy2(checkout / 'dgd' / name, root / 'dgd' / name)
        (root / 'dgd/datasets').mkdir()
        (root / 'dgd/datasets/__init__.py').touch()
        shutil.copy2(checkout / 'dgd/datasets/abstract_dataset.py', root / 'dgd/datasets/abstract_dataset.py')
        shutil.copy2(PACKAGE / 'payload/dgd/datasets/small_collection_beta.py',
                     root / 'dgd/datasets/small_collection_beta.py')
        hog = PACKAGE.parent / 'h1_twohop_mean_hog_rw_20261022/payload'
        for relative in ('dgd/diffusion_model_beta_h1.py', 'dgd/diffusion/hog_rw_features.py',
                         'configs/model/beta_h1_hog_rw.yaml', 'configs/model/beta_h1_twohop_mean_hog_rw.yaml'):
            shutil.copy2(hog / relative, root / relative)
        for dataset in ('community_small', 'enzymes'):
            shutil.copy2(Path('C:/path/to/HiGGs-main1/HiGGs-main/configs/dataset') / (dataset + '.yaml'),
                         root / 'configs/dataset' / (dataset + '.yaml'))
        sys.path.insert(0, str(root / 'dgd'))
        import torch
        import pytorch_lightning as pl
        import wandb
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from datasets.abstract_dataset import AbstractDatasetInfos
        from datasets.small_collection_beta import SmallCollectionNodePrior
        from diffusion.extra_features import ExtraFeatures, DummyExtraFeatures
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from pytorch_lightning.callbacks import ModelCheckpoint
        wandb.log = lambda *args, **kwargs: None
        torch.set_num_threads(1)
        torch.manual_seed(1234)
        for stage in ('h2', 'h1', 'glue'):
            item = {'datadir': str(root / 'data'), 'source': str(root / 'raw.pkl'),
                    'manifest': str(root / 'manifest.json'), 'splits': {'test': 20}}
            cfg = driver.make_config(root, 'community_small', stage, 'smoke', root / stage, item)
            xdim = 2 if stage == 'h2' else 1
            infos = AbstractDatasetInfos()
            infos.name = 'community_small'
            infos.n_nodes = torch.tensor([0., 0., 0., 0., 1.])
            infos.node_types = torch.tensor([1., 0.]) if xdim == 2 else torch.tensor([1.])
            infos.edge_types = torch.tensor([0.5, 0.5])
            infos.complete_infos(infos.n_nodes, infos.node_types)
            infos.nodes_dist = SmallCollectionNodePrior(infos.n_nodes)
            edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
            x = torch.nn.functional.one_hot(torch.zeros(4, dtype=torch.long), xdim).float()
            y = torch.tensor([[0., 1.]]) if stage == 'glue' else torch.tensor([[1., 0.]])
            data = Data(x=x, y=y, edge_index=edges,
                        edge_attr=torch.tensor([[0., 1.]]).repeat(edges.shape[1], 1))
            if stage == 'glue':
                membership = torch.tensor([0, 0, 1, 1])
                data.pos = membership[:, None] == membership[None, :]
            loader = DataLoader([data, data.clone()], batch_size=2)
            dm = SimpleNamespace(train_dataloader=lambda: loader)
            extra = DummyExtraFeatures() if stage == 'h2' else ExtraFeatures('all', infos)
            domain = DummyExtraFeatures()
            infos.compute_input_output_dims(dm, extra, domain)
            klass = driver.model_class(stage)
            model = klass(cfg=cfg, dataset_infos=infos,
                          train_metrics=TrainAbstractMetricsDiscrete(),
                          sampling_metrics=TrainAbstractMetricsDiscrete(),
                          visualization_tools=None, extra_features=extra, domain_features=domain)
            batch = next(iter(loader))
            output = model.training_step(batch, 0)
            loss = output['loss'] if isinstance(output, dict) else output
            assert torch.isfinite(loss).all(), (stage, loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

            class TinyDataModule(pl.LightningDataModule):
                def train_dataloader(self):
                    return loader

                def val_dataloader(self):
                    return loader

            callback = ModelCheckpoint(dirpath=str(root / stage), monitor=driver.STAGES[stage][2],
                                       mode='min', save_last=True, save_top_k=1)
            trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1,
                                  limit_train_batches=1, limit_val_batches=1,
                                  check_val_every_n_epoch=1, val_check_interval=1.0,
                                  num_sanity_val_steps=0, logger=False,
                                  enable_progress_bar=False, enable_model_summary=False,
                                  callbacks=[callback])
            trainer.fit(model, datamodule=TinyDataModule())
            assert callback.best_model_path and torch.isfinite(callback.best_model_score)
            checkpoint = {'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams),
                          'pytorch-lightning_version': pl.__version__, 'epoch': 0, 'global_step': 1}
            model.on_save_checkpoint(checkpoint)
            path = root / (stage + '.ckpt')
            torch.save(checkpoint, path)
            reloaded = klass.load_from_checkpoint(str(path), map_location='cpu', cfg=cfg, strict=True,
                          train_metrics=TrainAbstractMetricsDiscrete(),
                          sampling_metrics=TrainAbstractMetricsDiscrete())
            assert all(torch.equal(v, reloaded.state_dict()[k]) for k, v in model.state_dict().items())
            if stage == 'h1':
                widths = {k: getattr(reloaded.model, 'mlp_in_' + k)[0].in_features for k in ('X', 'E', 'y')}
                assert widths == {'X': 12, 'E': 6, 'y': 14}, widths
                assert checkpoint['h1_hog_rw']['rw_depth'] == 4
                assert checkpoint['h1_two_hop']['normalization'] == 'mean'
            print('PASS: synthetic CPU', stage, 'finite loss/gradients, one-epoch train/val and strict reload', float(loss))
        print('No real dataset training, GPU/server execution or sampling was performed.')


if __name__ == '__main__':
    main()
