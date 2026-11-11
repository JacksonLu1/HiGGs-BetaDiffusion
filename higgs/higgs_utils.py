"""
util functions for HiGGs, anonymous author, 2023
"""

import numpy as np
import networkx as nx
import wandb
import os
import sys
import warnings
import pandas as pd

import omegaconf

import torch
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import pickle

from datetime import datetime

from torch_geometric.io import read_npz
from torch_geometric.utils import to_networkx

pwd = os.getcwd()
dgd_dir = os.path.join(pwd, "dgd")
sys.path.insert(0, dgd_dir)

import utils
# 旧的连续扩散模型导入，当前仓库中没有对应文件，且 HiGGs 采样未使用
# from diffusion_model import LiftedDenoisingDiffusion

from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion_model_beta_h2 import ConditionalGraphBetaDiffusionH2
from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1
from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics, EGOSamplingMetrics, FBSamplingMetrics, GITSamplingMetrics, FullSampleMetrics, ByClassSampleMetrics

class TooManyNodesError(Exception):
    pass


_TRAINING_LOGGER_CLASS_NAMES = (
    "ConfigCSVLogger",
    "ConfigTensorBoardLogger",
    "ConfigWandbLogger",
)
_checkpoint_pickle_compat_registered = False


def register_checkpoint_pickle_compatibility():
    """Expose training-only logger classes for checkpoints pickled by dgd/main.py."""
    global _checkpoint_pickle_compat_registered
    if _checkpoint_pickle_compat_registered:
        return

    main_module = sys.modules["__main__"]
    try:
        from main import (
            ConfigCSVLogger,
            ConfigTensorBoardLogger,
            ConfigWandbLogger,
        )
        logger_classes = {
            "ConfigCSVLogger": ConfigCSVLogger,
            "ConfigTensorBoardLogger": ConfigTensorBoardLogger,
            "ConfigWandbLogger": ConfigWandbLogger,
        }
    except Exception as exc:
        warnings.warn(
            "[HiGGs] Could not safely import the training logger classes "
            f"from dgd/main.py ({exc!r}); using pickle-only compatibility "
            "placeholders.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger_classes = {
            name: type(name, (), {"__module__": "__main__"})
            for name in _TRAINING_LOGGER_CLASS_NAMES
        }

    for name, logger_class in logger_classes.items():
        if not hasattr(main_module, name):
            setattr(main_module, name, logger_class)
    _checkpoint_pickle_compat_registered = True


def _load_checkpoint_cpu(checkpoint_path):
    """Load checkpoint metadata after registering dgd/main.py pickle globals."""
    register_checkpoint_pickle_compatibility()
    try:
        return torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _checkpoint_resolution(checkpoint_path, stage_name):
    """Read a hierarchy resolution from checkpoint metadata without guessing."""
    checkpoint = _load_checkpoint_cpu(checkpoint_path)

    hyper_parameters = checkpoint.get("hyper_parameters", {})
    checkpoint_cfg = (hyper_parameters.get("cfg")
                      if hasattr(hyper_parameters, "get") else None)
    resolution = None
    if checkpoint_cfg is not None:
        try:
            resolution = omegaconf.OmegaConf.select(
                checkpoint_cfg, "dataset.resolution", default=None)
        except (AttributeError, TypeError, ValueError):
            try:
                resolution = checkpoint_cfg["dataset"]["resolution"]
            except (KeyError, TypeError):
                resolution = None

    del checkpoint
    print(f"[HiGGs] {stage_name} checkpoint resolution: {resolution}")
    if resolution is None:
        warnings.warn(
            f"[HiGGs] {stage_name} checkpoint has no "
            "hyper_parameters.cfg.dataset.resolution; resolution "
            "compatibility cannot be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
    return resolution


def audit_checkpoint_resolutions(h2_checkpoint, h1_checkpoint, glue_checkpoint):
    """Fail before sampling when known hierarchy resolutions are inconsistent."""
    resolutions = {
        "H2": _checkpoint_resolution(h2_checkpoint, "H2"),
        "H1": _checkpoint_resolution(h1_checkpoint, "H1"),
        "Glue": _checkpoint_resolution(glue_checkpoint, "Glue"),
    }

    normalized = set()
    for resolution in resolutions.values():
        if resolution is None:
            continue
        try:
            normalized.add(float(resolution))
        except (TypeError, ValueError):
            normalized.add(str(resolution))

    if len(normalized) > 1:
        details = "\n".join(
            f"{stage} checkpoint resolution={resolution}"
            for stage, resolution in resolutions.items())
        raise ValueError(
            "Resolution mismatch:\n"
            f"{details}\n\n"
            "Do not mix checkpoints trained with different hierarchy "
            "resolutions."
        )
    return resolutions


def _checkpoint_identity(checkpoint_path):
    """Read family/stage metadata without inferring either from a filename."""
    checkpoint = _load_checkpoint_cpu(checkpoint_path)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    checkpoint_cfg = (hyper_parameters.get("cfg")
                      if hasattr(hyper_parameters, "get") else None)
    family = checkpoint.get("model_family")
    stage = checkpoint.get("stage")
    if family is None and hasattr(hyper_parameters, "get"):
        family = hyper_parameters.get("model_family")
    if stage is None and hasattr(hyper_parameters, "get"):
        stage = hyper_parameters.get("stage")
    if checkpoint_cfg is not None:
        if family is None:
            family = omegaconf.OmegaConf.select(
                checkpoint_cfg, "model.family", default=None)
        if stage is None:
            stage = omegaconf.OmegaConf.select(
                checkpoint_cfg, "dataset.h", default=None)
        if family is None and omegaconf.OmegaConf.select(
                checkpoint_cfg, "model.type", default=None) == "discrete":
            family = "digress"
    del checkpoint
    return family, stage


def _require_checkpoint_identity(checkpoint_path, expected_family,
                                 expected_stage, label, allowed_families=None):
    family, stage = _checkpoint_identity(checkpoint_path)
    allowed = ({expected_family} if allowed_families is None
               else set(allowed_families))
    print(f"[HiGGs] {label} checkpoint family: {family}")
    print(f"[HiGGs] {label} checkpoint stage: {stage}")
    if family not in allowed:
        raise ValueError(
            f"{label} checkpoint family mismatch: expected one of "
            f"{sorted(allowed)}, got {family}")
    stage_aliases = {"h1": 1.0, "h2": 2.0, "glue": 1.5, "h1.5": 1.5}
    normalized_stage = stage_aliases.get(str(stage).lower(), stage)
    try:
        stage_matches = float(normalized_stage) == float(expected_stage)
    except (TypeError, ValueError):
        stage_matches = False
    if not stage_matches:
        raise ValueError(
            f"{label} checkpoint stage mismatch: expected "
            f"{expected_stage}, got {stage}")
    return family, stage


def get_samplers(cfg):
    """
    Loads a model for sampling using a given config.

    params:
    param: cfg: omegaconf.DictConfig config keys

    returns:
    param: cfg: the same config with updated keys from the loaded model
    param: model: the loaded sampling model
    """
    register_checkpoint_pickle_compatibility()
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'

    stage = cfg["dataset"]["h"]
    print(f"Model stage: {stage}")
    if stage == 1:
        resume = cfg.general.h1_model
        h1_family = omegaconf.OmegaConf.select(
            cfg, "general.h1_model_family", default="digress")
        if h1_family == "beta_h1":
            _require_checkpoint_identity(resume, "beta_h1", 1.0, "H1")
            model_class = ConditionalGraphBetaDiffusionH1
        else:
            _require_checkpoint_identity(
                resume, "digress", 1.0, "H1",
                allowed_families={"digress", "original_h1"})
            model_class = DiscreteDenoisingDiffusion
            print("[HiGGs] Stage h1 model: DiscreteDenoisingDiffusion")
    elif stage == 2:
        resume = cfg.general.h2_model
        h2_family = omegaconf.OmegaConf.select(
            cfg, "general.h2_model_family",
            default=omegaconf.OmegaConf.select(cfg, "model.family", default="digress"))
        model_class = (ConditionalGraphBetaDiffusionH2
                       if h2_family == "beta_h2"
                       else DiscreteDenoisingDiffusion)
        _require_checkpoint_identity(
            resume, h2_family, 2.0, "H2",
            allowed_families={h2_family})
        if model_class is DiscreteDenoisingDiffusion:
            print("[HiGGs] Stage h2 model: DiscreteDenoisingDiffusion")
    elif stage == 1.5:
        resume = cfg.general.glue_model
        _require_checkpoint_identity(
            resume, "original_glue", 1.5, "Glue",
            allowed_families={"digress", "original_glue"})
        model_class = DiscreteDenoisingDiffusion
        print("[HiGGs] Stage glue model: edge-DiGress")
    elif stage not in (1, 2):
        raise ValueError(f"Unknown HiGGs stage: {stage}")

    model = model_class.load_from_checkpoint(resume)

    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys_harsh(cfg, saved_cfg)
    # SAMPLER_INFERENCE_MODE_FIX
    # Checkpoints are used only for sampling. Disable Dropout and gradients.
    model.eval()

    if hasattr(model, "freeze"):
        model.freeze()
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    dropout_modules = [
        module
        for module in model.modules()
        if isinstance(
            module,
            (
                torch.nn.Dropout,
                torch.nn.Dropout1d,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
            ),
        )
    ]
    dropout_training = sum(
        int(module.training) for module in dropout_modules
    )

    print(
        "[HiGGs inference] "
        f"class={model.__class__.__name__} "
        f"model.training={model.training} "
        f"dropout_modules={len(dropout_modules)} "
        f"dropout_training={dropout_training}"
    )

    if model.training or dropout_training != 0:
        raise RuntimeError(
            "Sampling model did not enter evaluation mode correctly"
        )

    return cfg, model

def setup_wandb(cfg):
    """
    Uses a config dictionary to initialise wandb to track sampling.
    Requires a wandb account, https://wandb.ai/

    params: cfg: omegaconf.DictConfig config dictionary

    returns:
    param: cfg: same config
    """
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if cfg.dataset.subsample:
        extra_label = "_subsampled"
    else:
        extra_label = ""

    kwargs = {'name': f"Sampling-" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), 'project': f'hidiff_{cfg.general.name}{extra_label}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    wandb.log({"Type":"Sampling"})

    return cfg

def sampling_quality(original_G, cg):
    """
    Uses sampling metrics from DGD to assess the quality of the sampled graph
    compared to a baseline real version

    params:
    param: original_G: networkx.Graph of the original graph
    param: cg: networkx.Graph, the sampled version. cg refers to the largest connected component
    """
    utils.general_graph_metrics(cg)
    sampling_metrics = FullSampleMetrics([original_G])
    sampling_metrics([cg], "Full Sampling Metrics", 0, 0)

    sampling_metrics = ByClassSampleMetrics(original_G, cg)




def load_original(root_path="", name = None, all_graphs = False):
    """
    Loads original version of the graphs in our experiments

    params:
    param: root_path: string, optional, path of data directory for sbm graphs (all others are downloaded in-code), default=""
    param: name: name of dataset being loaded, ie cora, sbm, default=fb_hierarchies
    param: all_graphs: Boolean, default=False. Whether to load all the graphs in the dataset. Only relevant for the sbm dataset.

    returns:
    param: G: networkx.Graph, the loaded original graph
    """

    print(f"Loading original {name} graph with data directory {root_path}")
    if name == "cora":
        cora_url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz'
        resp = urlopen(cora_url)
        out = read_npz(BytesIO(resp.read()))

        G = to_networkx(out, to_undirected=True)

        node_classes = {n: out.y[i].item() for i,n in enumerate(list(G.nodes()))}

        nx.set_node_attributes(G, node_classes, name = "target")
        G = G.to_undirected()
        CGs = [G.subgraph(c) for c in nx.connected_components(G)]
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
        G = CGs[0]

        G = nx.convert_node_labels_to_integers(G)

        return G

    elif name == "sbm":
        file_path = os.path.join(root_path, "sbm/raw/sbm_200.pt")

        adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(
            file_path)

        if not all_graphs:
            random_graph_selection = np.random.randint(len(adjs))

            adj = adjs[random_graph_selection]

            g = nx.from_numpy_array(adj.numpy())

            these_atom_types = np.ones(g.order())

            these_edges, these_types = [], []
            for edge in g.edges():
                start, end = edge[0], edge[1]
                these_edges += [[start, end]]
                these_types += [1.]

            these_nodes = [(ind, {"target": v}) for ind, v in enumerate(these_atom_types)]
            these_edges = [(edge[0], edge[1], {"type": these_types[ind]}) for ind, edge in enumerate(these_edges)]

            G = nx.Graph()
            G.add_nodes_from(these_nodes)
            G.add_edges_from(these_edges)

            return G
        else:
            graphs = []
            for adj in adjs:
                random_graph_selection = np.random.randint(len(adjs))

                adj = adjs[random_graph_selection]

                g = nx.from_numpy_array(adj.numpy())

                these_atom_types = np.ones(g.order())

                these_edges, these_types = [], []
                for edge in g.edges():
                    start, end = edge[0], edge[1]
                    these_edges += [[start, end]]
                    these_types += [1.]

                these_nodes = [(ind, {"target": v}) for ind, v in enumerate(these_atom_types)]
                these_edges = [(edge[0], edge[1], {"type": these_types[ind]}) for ind, edge in enumerate(these_edges)]

                G = nx.Graph()
                G.add_nodes_from(these_nodes)
                G.add_edges_from(these_edges)

                graphs.append(G)
            return graphs

    else:
        resp = urlopen("https://snap.stanford.edu/data/facebook_large.zip")
        myzip = ZipFile(BytesIO(resp.read()))

        edgelist = pd.read_csv(myzip.open("facebook_large/musae_facebook_edges.csv"))
        G = nx.from_pandas_edgelist(df=edgelist, source="id_1", target="id_2")

        class_df = pd.read_csv(myzip.open("facebook_large/musae_facebook_target.csv"))

        unique_types = np.unique(class_df["page_type"])
        types = {target: i for i, target in enumerate(unique_types.tolist())}

        node_classes = {n: types[class_df.at[n, "page_type"]] for n in list(G.nodes())}

        nx.set_node_attributes(G, node_classes, name="target")
        return G
