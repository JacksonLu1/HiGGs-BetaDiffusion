"""Runtime-only H1 k-core adapter for the frozen mean-two-hop + HOG arms.

This file does not patch the experiment sources or replace their partitioner.
Only the two appended node columns and one appended edge column use tau=.20;
all existing structure/HOG features, output decisions and Beta transitions keep
their checkpoint configuration.  The threshold is a transferred SBM candidate,
not a claim that it has been tuned or validated on another dataset.

The caller must put the isolated experiment's ``dgd`` directory on sys.path
before calling get_h1_class.  No source or data imports occur merely on import.
"""
from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping

import torch
from omegaconf import OmegaConf, open_dict


PROTOCOL = "h1-noisy-kcore-multidataset-transferred-v1"
FEATURE_THRESHOLD = 0.20
K = 3
MAX_T = 0.50
ALIASES = {"facebook": "fb_hierarchies", "fb": "fb_hierarchies", "msrc_21": "msrc21"}
FAMILIES = {
    "base": ("cora", "fb_hierarchies", "planar", "dd", "community_small", "enzymes", "tree", "grid"),
    "four": ("ego_small", "proteins", "msrc21", "ego"),
    "tu": ("imdb_binary", "nci1", "collab"),
}
_CLASS_NAMES = {"base": "KcoreH1", "four": "KcoreFourBenchmarkH1", "tu": "KcoreTUBenchmarkH1"}
_BASE_CLASSES = {
    "base": ("diffusion_model_beta_h1", "ConditionalGraphBetaDiffusionH1"),
    "four": ("four_benchmark_models", "FourBenchmarkH1"),
    "tu": ("tu_benchmark_models", "TUBenchmarkH1"),
}
_CLASS_CACHE = {}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_dataset(dataset):
    name = str(dataset).strip().lower()
    name = ALIASES.get(name, name)
    _require(any(name in values for values in FAMILIES.values()), "Unsupported dataset: " + name)
    return name


def _validate_candidate(tau, k, max_t):
    _require(type(tau) in (int, float) and not isinstance(tau, bool) and float(tau) == FEATURE_THRESHOLD,
             "This transferred-candidate package fixes the feature threshold at 0.20")
    _require(type(k) is int and k == K, "This transferred-candidate package fixes k=3")
    _require(type(max_t) in (int, float) and not isinstance(max_t, bool) and float(max_t) == MAX_T,
             "This transferred-candidate package fixes max_t=0.50")


def _validate_old_feature_contract(cfg):
    _require(float(OmegaConf.select(cfg, "dataset.h", default=-1)) == 1.0, "Expected an H1 configuration")
    canonical_dataset(OmegaConf.select(cfg, "dataset.name", default=""))
    expected = {
        "model.two_hop.enabled": True,
        "model.two_hop.normalization": "mean",
        "model.two_hop.channels": 16,
        "model.two_hop.warm_start_checkpoint": None,
        "model.hog_rw.enabled": True,
        "model.hog_rw.rw_depth": 4,
        "model.eta_strategy": "fixed",
        "model.edge_threshold": 0.5,
    }
    for key, value in expected.items():
        actual = OmegaConf.select(cfg, key)
        _require(actual == value and (type(value) is not bool or type(actual) is bool),
                 "Existing mean/HOG contract differs: " + key)


def enable_kcore(cfg, tau=FEATURE_THRESHOLD, k=K, max_t=MAX_T):
    """Return a new configuration; never mutate the old checkpoint config.

    No seeds, batch sizes, data paths, partitions, diffusion schedules, sampling
    thresholds, optimizers or training budgets are changed by this function.
    """
    _validate_candidate(tau, k, max_t)
    result = copy.deepcopy(cfg)
    if not OmegaConf.is_config(result):
        result = OmegaConf.create(result)
    _validate_old_feature_contract(result)
    with open_dict(result):
        result.model.h1_kcore = {"enabled": True, "k": K, "max_t": MAX_T}
        result.model.kcore_feature_threshold = FEATURE_THRESHOLD
        result.model.kcore_transfer_protocol = PROTOCOL
    return result


def _zero_added_columns(model):
    """Initialize only the newly appended inputs to zero; no weights are loaded."""
    with torch.no_grad():
        for name, count in (("X", 2), ("E", 1)):
            layer = getattr(model.model, "mlp_in_" + name)[0]
            _require(layer.weight.ndim == 2 and layer.weight.shape[1] > count,
                     "Unexpected first input linear layer for " + name)
            layer.weight[:, -count:].zero_()


class _KcoreTransferMixin:
    """Keep family-specific singleton compatibility through cooperative super."""

    _dataset_family = ()

    def __init__(self, cfg, dataset_infos, train_metrics=None, sampling_metrics=None,
                 visualization_tools=None, extra_features=None, domain_features=None):
        _validate_old_feature_contract(cfg)
        name = canonical_dataset(cfg.dataset.name)
        _require(name in self._dataset_family, "Wrong dataset-aware runtime H1 class")
        _require(OmegaConf.select(cfg, "model.kcore_transfer_protocol") == PROTOCOL,
                 "Missing or incompatible k-core transfer protocol")
        raw = OmegaConf.select(cfg, "model.kcore_feature_threshold")
        _validate_candidate(raw, OmegaConf.select(cfg, "model.h1_kcore.k"),
                            OmegaConf.select(cfg, "model.h1_kcore.max_t"))
        _require(OmegaConf.select(cfg, "model.h1_kcore.enabled") is True,
                 "The transferred-candidate H1 requires k-core enabled")
        self._transfer_dataset = name
        self._transfer_tau = float(raw)
        super().__init__(cfg, dataset_infos, train_metrics, sampling_metrics,
                         visualization_tools, extra_features, domain_features)
        _require(self.kcore_options == {"enabled": True, "k": K, "max_t": MAX_T},
                 "Installed H1 did not retain the k-core options")
        _require(float(self.edge_threshold) == .5, "Output threshold must remain 0.5")
        # Strict checkpoint restoration runs after __init__ and restores these
        # learned columns normally. This operation only defines scratch init.
        _zero_added_columns(self)

    def _extra_data(self, X_t, E_t, condition_y, t, node_mask):
        # Parent uses the ORIGINAL .5 threshold for all old feature channels.
        # Replace only the tail added by the opt-in installed k-core adapter.
        X, E, y = super()._extra_data(X_t, E_t, condition_y, t, node_mask)
        _require(node_mask.dtype == torch.bool, "node_mask must remain Boolean")
        score = self._state_to_unit(E_t, self.edge_scale, self.edge_shift)[..., 0]
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        selected = (score > self._transfer_tau) & torch.triu(valid, diagonal=1)
        adjacency = (selected | selected.transpose(1, 2)).to(score)
        feature_module = importlib.import_module("diffusion.h1_kcore_features")
        cx, ce = feature_module.masked_h1_kcore_features(adjacency, node_mask, t, k=K, max_t=MAX_T)
        _require(X.shape[-1] >= 2 and E.shape[-1] >= 1, "Missing appended k-core columns")
        return torch.cat((X[..., :-2], cx.to(X)), -1), torch.cat((E[..., :-1], ce.to(E)), -1), y

    def _kcore_checkpoint_metadata(self):
        metadata = super()._kcore_checkpoint_metadata()
        # An unadapted .5-k-core loader will reject this metadata, preventing a
        # checkpoint from silently being interpreted with the wrong feature.
        metadata["feature_edge_threshold"] = self._transfer_tau
        return metadata

    def _transfer_metadata(self):
        return {
            "version": 1,
            "protocol": PROTOCOL,
            "dataset": self._transfer_dataset,
            "feature_threshold": self._transfer_tau,
            "threshold_origin": "sbm_train_candidate_transfer_not_dataset_tuned",
            "output_threshold": 0.5,
            "hog_structural_threshold": 0.5,
            "initialization": "scratch_zero_added_columns",
        }

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["h1_kcore_transfer"] = self._transfer_metadata()

    def on_load_checkpoint(self, checkpoint):
        _require(checkpoint.get("h1_kcore_transfer") == self._transfer_metadata(),
                 "K-core transfer checkpoint metadata mismatch")
        hp = checkpoint.get("hyper_parameters")
        _require(isinstance(hp, Mapping) and "cfg" in hp, "Missing saved H1 configuration")
        saved = hp["cfg"]
        _require(OmegaConf.select(saved, "model.kcore_feature_threshold") == self._transfer_tau,
                 "Saved k-core feature threshold mismatch")
        _require(OmegaConf.select(saved, "model.kcore_transfer_protocol") == PROTOCOL,
                 "Saved k-core protocol mismatch")
        _require(canonical_dataset(OmegaConf.select(saved, "dataset.name", default="")) == self._transfer_dataset,
                 "Saved k-core dataset mismatch")
        _validate_old_feature_contract(saved)
        super().on_load_checkpoint(checkpoint)


def get_h1_class(dataset):
    """Get the strict runtime adapter while retaining dataset-specific fixes."""
    name = canonical_dataset(dataset)
    family = next(key for key, values in FAMILIES.items() if name in values)
    if family not in _CLASS_CACHE:
        module_name, symbol = _BASE_CLASSES[family]
        base = getattr(importlib.import_module(module_name), symbol)
        class_name = _CLASS_NAMES[family]
        adapter = type(class_name, (_KcoreTransferMixin, base), {
            "__module__": __name__, "__qualname__": class_name,
            "_dataset_family": FAMILIES[family],
        })
        globals()[class_name] = adapter
        _CLASS_CACHE[family] = adapter
    return _CLASS_CACHE[family]


def __getattr__(name):
    # Make explicit class-name imports and Python object reconstruction robust
    # without importing all three model families on ordinary module import.
    for family, symbol in _CLASS_NAMES.items():
        if name == symbol:
            return get_h1_class(FAMILIES[family][0])
    raise AttributeError(name)
