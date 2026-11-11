"""Isolated training entry for frozen four benchmark collections; no shared source edits.

This file deliberately does not invoke dgd/main.py: an existing DD experiment
can keep training while these additive entry points are installed and used.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

from qb_beta_common import ROOT, SPECS, STAGES, baseline, check_hashes, get_item, sha


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', choices=tuple(SPECS), required=True)
    parser.add_argument('--stage', choices=tuple(STAGES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke', 'train'), required=True)
    return parser.parse_args()


def make_config(root, dataset, stage, mode, output, item):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    h, group, _ = STAGES[stage]
    # Compose each checkpoint's actual model group. H2 configuration must never
    # overwrite the H1 mean/HOG model during training or checkpoint reload.
    with initialize_config_dir(version_base='1.1', config_dir=str(root / 'configs')):
        cfg = compose(config_name='config', overrides=[
            'dataset=' + dataset, 'model=' + group,
        ])
    OmegaConf.set_struct(cfg, False)
    cfg.dataset.h = float(h) if h == '1.5' else int(h)
    cfg.dataset.datadir = item['datadir']
    cfg.dataset.manifest = item['manifest']
    cfg.dataset.source = item['source']
    cfg.general.name = dataset + '_' + stage + '_mean_hog_' + mode
    cfg.general.wandb = 'disabled'
    cfg.general.gpus = 1
    cfg.general.resume = None
    cfg.general.test_only = None
    cfg.general.check_val_every_n_epochs = 1 if mode == 'smoke' else 10
    cfg.general.val_check_interval = 1.0
    cfg.general.sample_every_val = 1000000000
    cfg.general.log_every_steps = 1 if mode == 'smoke' else 10
    cfg.general.samples_to_generate = 4
    cfg.general.samples_to_save = 0
    cfg.general.chains_to_save = 0
    cfg.general.final_model_samples_to_generate = item['splits']['test']
    cfg.general.run_test_after_fit = False
    cfg.general.limit_train_batches = 2 if mode == 'smoke' else 1.0
    cfg.general.limit_val_batches = 2 if mode == 'smoke' else 1.0
    cfg.train.n_epochs = 1 if mode == 'smoke' else SPECS[dataset]['epochs']
    cfg.train.batch_size = 32
    cfg.train.num_workers = 0
    cfg.train.lr = 0.0002
    cfg.train.seed = 0
    cfg.train.ema_decay = 0
    cfg.train.save_model = True
    # Make the family explicit for the new Glue checkpoint. Its model, 400-step
    # transition, optimizer and loss are still the existing original_glue group.
    if stage == 'glue':
        cfg.model.family = 'original_glue'
    assert cfg.dataset.name == dataset
    assert cfg.dataset.resolution == item['resolution'] and cfg.dataset.max_size == item['max_size']
    assert cfg.train.weight_decay == 1e-12 and cfg.train.clip_grad is None
    assert cfg.train.optimizer == 'adamw' and cfg.train.amsgrad is True
    assert cfg.model.diffusion_steps == (400 if stage == 'glue' else 100)
    if stage != 'glue':
        assert cfg.model.pre_condition is False and cfg.model.eta_strategy == 'fixed'
        assert cfg.model.edge_threshold == 0.5
    if stage == 'h1':
        assert cfg.model.two_hop.enabled and cfg.model.two_hop.normalization == 'mean'
        assert cfg.model.two_hop.channels == 16
        assert cfg.model.two_hop.warm_start_checkpoint is None
        assert cfg.model.hog_rw.enabled and cfg.model.hog_rw.rw_depth == 4
    OmegaConf.resolve(cfg)
    return cfg


def model_class(stage):
    from four_benchmark_models import model_class as select
    return select(stage)


def read_losses(logger_dir, prefix):
    path = Path(logger_dir) / 'metrics.csv'
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    losses = {k: [float(row[k]) for row in rows if row.get(k)]
              for k in (prefix + '/train_loss', prefix + '/val_loss')}
    assert all(values and all(math.isfinite(v) for v in values)
               for values in losses.values()), losses
    return losses


def train(args):
    import fcntl

    root, output = args.root.resolve(), args.output.resolve()
    assert root == ROOT, 'Use only the modified Linux experiment root'
    assert (root / 'outputs') in output.parents and output != root / 'outputs'
    assert not args.output.is_symlink() and not output.exists(), 'Never overwrite an existing run'
    item = get_item(root, args.dataset)
    before = baseline(root, args.dataset)
    before[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())

    lock_path = root / 'logs/training' / (args.dataset + '_' + args.stage + '_mean_hog.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Keep this descriptor alive through strict checkpoint reload and reporting.
    assert not output.exists()
    os.chdir(root)
    sys.path[:0] = [str(root / 'dgd'), str(root)]

    import torch
    import wandb
    from omegaconf import OmegaConf
    from pytorch_lightning import Callback, Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from datasets.four_benchmark_beta import FrozenCollectionDataModule, FourBenchmarkBetaInfos
    from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
    from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
    from analysis.visualization import DiscreteNodeTypeVisualization

    assert torch.cuda.is_available(), 'CUDA unavailable; refusing CPU training'
    torch.cuda.set_device(0)
    free = torch.cuda.mem_get_info()[0]
    assert free >= 8 * 1024**3, 'Less than 8 GiB free GPU memory'
    print('Physical GPU:', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
    print('Logical GPU: cuda:0', torch.cuda.get_device_name(0), flush=True)
    print('Free GPU memory GiB:', free / 1024**3, '; no utilization restriction', flush=True)
    seed_everything(0, workers=True)
    cfg = make_config(root, args.dataset, args.stage, args.mode, output, item)
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    (output / '.hydra').mkdir()
    OmegaConf.save(cfg, output / '.hydra/config.yaml')
    # Some preserved discrete-model logging is unconditional. Disabled mode is
    # local/no-network and does not add an online WandB dependency to the run.
    wandb.init(project='higgs-small-beta-local', name=cfg.general.name,
               mode='disabled', dir=str(output))

    dm = FrozenCollectionDataModule(cfg)
    dm.prepare_data()
    infos = FourBenchmarkBetaInfos(dm, cfg.dataset)
    extra = (DummyExtraFeatures() if args.stage == 'h2' else
             ExtraFeatures(cfg.model.extra_features, dataset_info=infos))
    domain = DummyExtraFeatures()
    infos.compute_input_output_dims(dm, extra, domain)
    klass = model_class(args.stage)
    model = klass(cfg=cfg, dataset_infos=infos,
                  train_metrics=TrainAbstractMetricsDiscrete(),
                  sampling_metrics=TrainAbstractMetricsDiscrete(),
                  visualization_tools=DiscreteNodeTypeVisualization(),
                  extra_features=extra, domain_features=domain)
    assert infos.manifest_sha256 == item['manifest_sha256']
    if args.stage == 'h1':
        widths = {k: getattr(model.model, 'mlp_in_' + k)[0].in_features for k in ('X', 'E', 'y')}
        assert widths == {'X': 12, 'E': 6, 'y': 14}, widths
    else:
        widths = None

    prefix = 'smoke' if args.mode == 'smoke' else 'formal'

    class FiniteAudit(Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            assert loss is not None and torch.isfinite(loss).all(), 'Nonfinite training loss'
            module.log(prefix + '/train_loss', loss.detach(), on_step=False,
                       on_epoch=True, batch_size=batch.num_graphs)

        def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            assert loss is not None and torch.isfinite(loss).all(), 'Nonfinite validation loss'
            module.log(prefix + '/val_loss', loss.detach(), on_step=False,
                       on_epoch=True, batch_size=batch.num_graphs)

        def on_after_backward(self, trainer, module):
            assert all(torch.isfinite(p.grad).all() for p in module.parameters()
                       if p.grad is not None), 'Nonfinite gradient'

    monitor = STAGES[args.stage][2]
    callback = ModelCheckpoint(dirpath=str(output / 'checkpoints' / cfg.general.name),
                               filename='{epoch}', monitor=monitor, mode='min',
                               save_top_k=2, save_last=True,
                               every_n_epochs=cfg.general.check_val_every_n_epochs)
    logger = CSVLogger(str(output / 'logs'), name='metrics', version='run')
    trainer = Trainer(accelerator='gpu', devices=1, max_epochs=cfg.train.n_epochs,
                      gradient_clip_val=cfg.train.clip_grad,
                      limit_train_batches=min(2, len(dm.train_dataloader())) if args.mode == 'smoke' else 1.0,
                      limit_val_batches=min(2, len(dm.val_dataloader())) if args.mode == 'smoke' else 1.0,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      val_check_interval=1.0, num_sanity_val_steps=0,
                      enable_progress_bar=cfg.train.progress_bar,
                      log_every_n_steps=cfg.general.log_every_steps,
                      callbacks=[FiniteAudit(), callback], logger=logger)
    print('START', args.dataset, args.stage, args.mode,
          'epochs=', cfg.train.n_epochs, 'batch=32 seed=0',
          'manifest=', item['manifest_sha256'], flush=True)
    trainer.fit(model, datamodule=dm)
    # This entry does not call Trainer.test(), component tests or sampling.
    last = output / 'checkpoints' / cfg.general.name / 'last.ckpt'
    best = Path(callback.best_model_path).resolve()
    assert output in best.parents and best.is_file() and last.is_file()
    assert callback.best_model_score is not None and math.isfinite(float(callback.best_model_score))
    from datasets.four_benchmark_beta import load_cpu
    last_state, best_state = load_cpu(last), load_cpu(best)
    epochs = cfg.train.n_epochs
    expected_steps = min(2, len(dm.train_dataloader())) if args.mode == 'smoke' else len(dm.train_dataloader()) * epochs
    assert int(last_state['epoch']) == epochs - 1
    assert int(last_state['global_step']) == expected_steps
    assert 0 <= int(best_state['epoch']) < epochs

    for state, path in ((last_state, last), (best_state, best)):
        loaded_cfg = state['hyper_parameters']['cfg']
        loaded_cfg = loaded_cfg if OmegaConf.is_config(loaded_cfg) else OmegaConf.create(loaded_cfg)
        assert loaded_cfg.dataset.name == args.dataset
        assert float(loaded_cfg.dataset.h) == float(STAGES[args.stage][0])
        assert loaded_cfg.general.resume is None and loaded_cfg.general.test_only is None
        assert int(loaded_cfg.train.seed) == 0
        reloaded = klass.load_from_checkpoint(str(path), map_location='cpu', cfg=loaded_cfg, strict=True,
                     train_metrics=TrainAbstractMetricsDiscrete(), sampling_metrics=TrainAbstractMetricsDiscrete())
        assert reloaded.dataset_info.manifest_sha256 == item['manifest_sha256']
        assert list(reloaded.dataset_info.training_graph_ids) == list(infos.training_graph_ids)
        assert type(reloaded.node_dist).__module__ == 'datasets.four_benchmark_beta'
        assert all(torch.isfinite(p).all() for p in reloaded.parameters())
        if args.stage == 'h1':
            assert state['h1_two_hop']['normalization'] == 'mean'
            assert state['h1_two_hop']['warm_start'] is None
            assert state['h1_hog_rw']['rw_depth'] == 4
            assert state['h1_hog_rw']['two_hop'] is True
            assert {k: getattr(reloaded.model, 'mlp_in_' + k)[0].in_features
                    for k in ('X', 'E', 'y')} == widths
        del reloaded
    losses = read_losses(logger.log_dir, prefix)
    check_hashes(before)
    result = {
        'dataset': args.dataset, 'stage': args.stage, 'epoch': int(last_state['epoch']),
        'global_step': int(last_state['global_step']), 'monitor': monitor,
        'best_score': float(callback.best_model_score), 'best': str(best), 'last': str(last),
        'strict_reload': True, 'strict_best_reload': True,
        'best_epoch': int(best_state['epoch']), 'best_sha256': sha(best), 'last_sha256': sha(last),
        'manifest_sha256': item['manifest_sha256'],
        'two_hop': last_state.get('h1_two_hop'), 'hog': last_state.get('h1_hog_rw'),
        'input_widths': widths, 'train_seed': 0, 'batch_size': 32,
        'diffusion_steps': int(cfg.model.diffusion_steps),
        'losses': losses if args.mode == 'smoke' else {
            k: {'count': len(v), 'last': v[-1]} for k, v in losses.items()},
        'inputs_unchanged': True,
    }
    report = output / ('smoke_result.json' if args.mode == 'smoke' else 'formal_result.json')
    with report.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    with (output / 'input_hash_check.json').open('x', encoding='utf-8') as handle:
        json.dump({'unchanged': True, 'sha256': before}, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print('PASS:', args.dataset.upper() + '_' + args.stage.upper() + '_MEAN_HOG_' +
          ('REAL_SMOKE' if args.mode == 'smoke' else 'FORMAL_TRAINING_COMPLETE'), flush=True)
    print('No test evaluation, H2/Glue sampling or full-chain graph sampling was performed.', flush=True)
    wandb.finish()
    lock.close()


if __name__ == '__main__':
    train(parse_args())
