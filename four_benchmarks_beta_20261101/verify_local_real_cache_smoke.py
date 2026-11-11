"""Local CPU integration QA only, not a server or formal-training entry."""
import json
import math
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL_FIXTURE = Path('C:/path/to/HiGGs-main1/HiGGs-main/tmp/small_benchmark_stage_20261101')


def main():
    from test_cpu_models import ModelTests
    ModelTests.setUpClass()
    try:
        root = ModelTests.root
        shutil.copy2(HERE / 'payload/dgd/datasets/four_benchmark_beta.py',
                     root / 'dgd/datasets/four_benchmark_beta.py')
        for name in ('ego_small', 'proteins', 'msrc21', 'ego'):
            shutil.copy2(ORIGINAL_FIXTURE / ('configs/dataset/' + name + '.yaml'),
                         root / ('configs/dataset/' + name + '.yaml'))
        import torch
        from pytorch_lightning import Trainer, seed_everything, Callback
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger
        from datasets.four_benchmark_beta import FrozenCollectionDataModule, FourBenchmarkBetaInfos
        from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        import qb_beta_train as driver

        class FiniteLoss(Callback):
            def on_after_backward(self, trainer, model):
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

        results = []
        for name, spec in driver.SPECS.items():
            item = {'datadir': str(ORIGINAL_FIXTURE / ('data/' + name + '_r1_s1234_l0')),
                    'source': str(ORIGINAL_FIXTURE / ('data_sources/small_benchmarks/' + name + '.json')),
                    'manifest': str(ORIGINAL_FIXTURE / ('data_sources/splits/' + name + '_r1_s1234_l0.json')),
                    'splits': spec['splits'], 'resolution': 1, 'max_size': spec['max_size']}
            for stage in ('h2', 'h1', 'glue'):
                seed_everything(0, workers=True)
                output = root / 'real_smoke' / name / stage
                cfg = driver.make_config(root, name, stage, 'smoke', output, item)
                cfg.train.progress_bar = False
                dm = FrozenCollectionDataModule(cfg)
                dm.prepare_data()
                infos = FourBenchmarkBetaInfos(dm, cfg.dataset)
                extra = DummyExtraFeatures() if stage == 'h2' else ExtraFeatures('all', infos)
                domain = DummyExtraFeatures()
                infos.compute_input_output_dims(dm, extra, domain)
                klass = driver.model_class(stage)
                model = klass(cfg=cfg, dataset_infos=infos,
                    train_metrics=TrainAbstractMetricsDiscrete(), sampling_metrics=TrainAbstractMetricsDiscrete(),
                    visualization_tools=None, extra_features=extra, domain_features=domain)
                callback = ModelCheckpoint(dirpath=str(output / 'checkpoints'),
                    monitor=driver.STAGES[stage][2], mode='min', save_top_k=1, save_last=True)
                trainer = Trainer(accelerator='cpu', devices=1, max_epochs=1,
                    limit_train_batches=min(2, len(dm.train_dataloader())),
                    limit_val_batches=min(2, len(dm.val_dataloader())),
                    num_sanity_val_steps=0, enable_model_summary=False, enable_progress_bar=False,
                    logger=CSVLogger(str(output / 'logs'), name='metrics'),
                    log_every_n_steps=1, callbacks=[FiniteLoss(), callback])
                trainer.fit(model, datamodule=dm)
                assert trainer.global_step == min(2, len(dm.train_dataloader()))
                assert math.isfinite(float(callback.best_model_score))
                loaded = klass.load_from_checkpoint(callback.best_model_path, map_location='cpu', strict=True,
                    train_metrics=TrainAbstractMetricsDiscrete(), sampling_metrics=TrainAbstractMetricsDiscrete())
                assert loaded.dataset_info.manifest_sha256 == spec['manifest_sha256']
                assert all(torch.isfinite(p).all() for p in loaded.parameters())
                results.append({'dataset': name, 'stage': stage, 'steps': trainer.global_step,
                    'score': float(callback.best_model_score), 'strict_reload': True,
                    'diffusion_steps': cfg.model.diffusion_steps})
                print('REAL_CACHE_CPU_SMOKE', json.dumps(results[-1]), flush=True)
                del loaded, trainer, model, dm, infos
        assert len(results) == 12
        print('PASS: 12 actual-cache CPU training/validation/checkpoint reload checks; no server/formal training')
    finally:
        ModelTests.tearDownClass()


if __name__ == '__main__':
    main()
