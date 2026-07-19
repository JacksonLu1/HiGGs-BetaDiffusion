"""
Adaption of DiGress (https://github.com/cvignac/DiGress) for HiGGs hierarchy stages
Should be run from parent directory, i.e.
`python dgd/main.py **kwargs`
"""

import os
import sys
import atexit
import pathlib
import threading
import warnings
from collections.abc import Mapping
from datetime import datetime
import pickle
import torch


class Tee:
    """Mirror text output to multiple streams."""

    def __init__(self, *streams, lock=None):
        self.streams = streams
        self.lock = lock or threading.RLock()

    def write(self, obj):
        with self.lock:
            for stream in self.streams:
                stream.write(obj)
                stream.flush()
        return len(obj)

    def flush(self):
        with self.lock:
            for stream in self.streams:
                stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        return self.streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.streams[0], "errors", "strict")


def setup_console_logging(log_root):
    """Tee stdout/stderr to a per-run file and restore them at process exit."""
    os.makedirs(log_root, exist_ok=True)
    console_path = os.path.join(log_root, "console.log")
    log_file = open(console_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    lock = threading.RLock()
    stdout_tee = Tee(original_stdout, log_file, lock=lock)
    stderr_tee = Tee(original_stderr, log_file, lock=lock)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    def cleanup():
        if sys.stdout is stdout_tee:
            sys.stdout = original_stdout
        if sys.stderr is stderr_tee:
            sys.stderr = original_stderr
        if not log_file.closed:
            log_file.flush()
            log_file.close()

    atexit.register(cleanup)
    return console_path


torch.multiprocessing.set_sharing_strategy('file_system')

import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger


import utils
from datasets import fb_hierarchies, cora_dataset, sbm_hierarchies
from datasets.spectre_dataset import SBMDataModule

from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from analysis.spectre_utils import FBSamplingMetrics, ShetlandSamplingMetrics, CoraSamplingMetrics, SBMSamplingMetrics

from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion_model_beta_h2 import ConditionalGraphBetaDiffusionH2
from analysis.visualization import DiscreteNodeTypeVisualization
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from tqdm import tqdm

warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def get_model_class(cfg):
    """Select beta diffusion only for h2; missing family stays backward compatible."""
    family = cfg.model.get('family', 'digress')
    if float(cfg.dataset.h) == 2.0 and family == 'beta_h2':
        return ConditionalGraphBetaDiffusionH2
    return DiscreteDenoisingDiffusion


def log_beta_h2_training_audit(cfg, datamodule, dataset_infos):
    """Print the resolved SBM h2 data and node-count prior before training."""
    try:
        is_h2 = float(cfg.dataset.h) == 2.0
    except (TypeError, ValueError):
        is_h2 = False
    if (cfg.dataset.name != "sbm" or not is_h2
            or cfg.model.get("family", "digress") != "beta_h2"):
        return

    print("\n[HiGGs] Beta-H2 training data audit")
    print(f"[HiGGs] dataset.name = {cfg.dataset.name}")
    print(f"[HiGGs] dataset.h = {cfg.dataset.h}")
    print(f"[HiGGs] dataset.resolution = {cfg.dataset.resolution}")
    print(f"[HiGGs] dataset.regime = {cfg.dataset.regime}")
    print(f"[HiGGs] dataset.max_size = {cfg.dataset.max_size}")

    for split in ("train", "val", "test"):
        dataset = datamodule.dataloaders[split].dataset
        processed_path = os.path.abspath(
            dataset.processed_paths[dataset.file_idx])
        print(f"[HiGGs] {split} processed path = {processed_path}")

    node_prob = dataset_infos.nodes_dist.prob.detach().cpu()
    node_values = torch.arange(node_prob.numel(), dtype=node_prob.dtype)
    node_mean = float((node_values * node_prob).sum().item())
    print(f"[HiGGs] dataset_infos.nodes_dist.prob = {node_prob.tolist()}")
    print(f"[HiGGs] max_n_nodes = {dataset_infos.max_n_nodes}")
    print(f"[HiGGs] H2 node-count mean = {node_mean:.6f}\n")


def logger_hyperparameters(params):
    """Keep logger metadata serializable without changing checkpoint hparams."""
    cfg = params.get('cfg') if isinstance(params, Mapping) else getattr(params, 'cfg', None)
    if omegaconf.OmegaConf.is_config(cfg):
        cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    return {'cfg': cfg}


class ConfigCSVLogger(CSVLogger):
    def log_hyperparams(self, params):
        super().log_hyperparams(logger_hyperparameters(params))


class ConfigTensorBoardLogger(TensorBoardLogger):
    def log_hyperparams(self, params, metrics=None):
        super().log_hyperparams(logger_hyperparameters(params), metrics)


class ConfigWandbLogger(WandbLogger):
    def log_hyperparams(self, params):
        super().log_hyperparams(logger_hyperparameters(params))


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    model = get_model_class(cfg).load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)
    model = get_model_class(cfg).load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if cfg.dataset.subsample:
        extra_label = "_subsampled"
    else:
        extra_label = ""

    kwargs = {'name': f"h{cfg.dataset.h}-{cfg.dataset.regime}-" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), 'project': f'higgs_{cfg.dataset.name}{extra_label}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'mode': cfg.general.wandb}
    wandb_entity = os.environ.get("WANDB_ENTITY")
    if wandb_entity:
        kwargs['entity'] = wandb_entity
    wandb.init(**kwargs)
    wandb.save('*.txt')

    wandb.log({"Type":"Training"})

    return cfg


def build_lightning_loggers(cfg, log_root):
    """Create isolated local logs and connect Lightning to the active W&B run."""
    loggers = [
        ConfigCSVLogger(save_dir=log_root, name="csv", version="metrics"),
        ConfigTensorBoardLogger(save_dir=log_root, name="tensorboard", version="events"),
    ]
    if cfg.general.wandb != "disabled" and wandb.run is not None:
        loggers.append(ConfigWandbLogger(experiment=wandb.run, log_model=False))
    return loggers

@hydra.main(version_base='1.1', config_path='../configs', config_name='config') #
def main(cfg: DictConfig):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
    log_root = os.path.abspath(os.path.join(os.getcwd(), "logs", cfg.general.name, run_id))
    console_path = setup_console_logging(log_root)
    print(f"Console log saved to: {console_path}")
    print("Get past imports")

    dataset_config = cfg["dataset"]
    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)
    lightning_loggers = build_lightning_loggers(cfg, log_root)

    if dataset_config["name"] in ['sbm_dgd']:
        # Using dgd as default here is like using HiGGs only for h2
        # We only implement here for the SBM dataset, as others are well beyond DGD memory limits

        if dataset_config['name'] == 'sbm_dgd':
            datamodule = SBMDataModule(cfg)
            sampling_metrics = SBMSamplingMetrics(datamodule.dataloaders)
            dataset_infos = sbm_hierarchies.SBMDatasetInfos(datamodule,
                                                          dataset_config)

        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = DiscreteNodeTypeVisualization()

        print(f"Extra features")
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        print(f"Computing in/out dims")
        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}


    elif dataset_config["name"] in ["fb_hierarchies", "shetland", "cora", "sbm"]:
        # Get datamodules, sampling metrics and dataset infos for each of the datasets we use
        # Shetland is actually the road network of Iceland, and code is still functional, but not included in paper

        if dataset_config["name"] == "fb_hierarchies":
            datamodule = fb_hierarchies.FBHierarchiesDataModule(cfg)
            datamodule.prepare_data()
            sampling_metrics = FBSamplingMetrics(datamodule.dataloaders)
            dataset_infos = fb_hierarchies.FBDatasetInfos(datamodule,
                                                        dataset_config)

        elif dataset_config["name"] == "shetland":
            from datasets import shetland_dataset

            datamodule = shetland_dataset.ShetlandDataModule(cfg)
            datamodule.prepare_data()
            sampling_metrics = ShetlandSamplingMetrics(datamodule.dataloaders)
            dataset_infos = shetland_dataset.ShetlandDatasetInfos(datamodule,
                                                        dataset_config)

        elif dataset_config["name"] == "cora":
            datamodule = cora_dataset.CORADataModule(cfg)
            datamodule.prepare_data()
            sampling_metrics = CoraSamplingMetrics(datamodule.dataloaders)
            dataset_infos = cora_dataset.CORADatasetInfos(datamodule,
                                                          dataset_config)

        elif dataset_config["name"] == "sbm":
            datamodule = sbm_hierarchies.SBMDataModule(cfg)
            datamodule.prepare_data()
            sampling_metrics = SBMSamplingMetrics(datamodule.dataloaders)
            dataset_infos = sbm_hierarchies.SBMDatasetInfos(datamodule,
                                                          dataset_config)

        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = DiscreteNodeTypeVisualization()

        print(f"Initialising extra features")
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        print(f"Computing in/out dims")
        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

        if cfg.dataset.dataset_testing:
            quit()

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    log_beta_h2_training_audit(cfg, datamodule, dataset_infos)


    # Config can be over-written if model is resumed, so store these from existing config
    # if testing only one stage (h2) for the model
    n_to_generate = cfg.general.final_model_samples_to_generate
    bs = cfg.general.samples_to_generate
    n_batches = int(n_to_generate / bs)
    remainder = n_to_generate % n_batches

    print("\nFinished building datasets!\n")
    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    print("\nGetting model")
    model_class = get_model_class(cfg)
    model = model_class(cfg=cfg, **model_kwargs)
    print(f"Got model: {model}")

    callbacks = []
    if cfg.train.save_model:
        checkpoint_monitor = ('val/beta_h2_loss' if model_class is ConditionalGraphBetaDiffusionH2
                              else 'val/epoch_NLL_stable')
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor=checkpoint_monitor,
                                              save_top_k=2,
                                              save_last=True,
                                              mode='min',
                                              every_n_epochs=10)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    print(f"\nGetting trainer, GPU status: {'gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu'}\n")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      accelerator='gpu' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu',
                      devices=cfg.general.gpus if torch.cuda.is_available() and cfg.general.gpus > 0 else None,
                      limit_train_batches=20 if name == 'test' else None,
                      limit_val_batches=20 if name == 'test' else None,
                      limit_test_batches=20 if name == 'test' else None,
                      val_check_interval=cfg.general.val_check_interval,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,
                      enable_progress_bar=cfg.train.progress_bar,
                      callbacks=callbacks,
                      num_sanity_val_steps=0,
                      log_every_n_steps=cfg.general.log_every_steps,
                      logger=lightning_loggers)

    if not cfg.general.test_only:
        print("\nEntering trainer\n")
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        print("\nEntering tester\n")
        # Start by evaluating test_only_path
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)

        # Testing for only one component model of HiGGs is implemented only for h2 currently
        # Next piece of code generates and saves graphs
        graphs = []
        print(f"Running for {n_batches} batches of size {bs}, remainder {remainder}, to produce {n_to_generate} graphs")
        for ib in tqdm(range(n_batches + 1)):
            # Testing here assumes that you're generating graphs only for h2
            batch_size = bs if ib <= n_batches else remainder
            print(f"Batch size: {batch_size}")
            predicted_graph_list, networkx_graphs = model.sampling_batch_h2(ib, batch_size, 0, 100, batch_size, figures=True)

            graphs += networkx_graphs

        # Dump pickle of resulting networkx graphs
        for graph_number in range(len(graphs)):
            with open(f"sampled_h1_graphs_{graph_number}.pkl", "wb") as handle:
                pickle.dump(graphs[graph_number], handle, protocol=pickle.HIGHEST_PROTOCOL)

        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
