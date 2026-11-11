"""Dataset-scoped singleton compatibility; existing diffusion sources are untouched.

No parameters, graph vertices, transition equations or sampling thresholds are
added. For a width-one dense batch, the original transformer's unbiased std is
undefined. We pad its *neural forward* with one masked vertex and crop it away.
This is the same compatibility policy used by the original benchmark arm.
"""
import torch
from torch.nn import functional as F

import utils
from diffusion_model_beta_h1 import ConditionalGraphBetaDiffusionH1
from diffusion_model_beta_h2 import ConditionalGraphBetaDiffusionH2
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from metrics.abstract_metrics import CrossEntropyMetric
from models.transformer_model import GraphTransformer
from models.h1_twohop_transformer import H1TwoHopGraphTransformer


DATASETS = ("ego_small", "proteins", "msrc21", "ego")


class SingletonForward:
    def forward(self, X, E, y, node_mask):
        if X.shape[1] != 1:
            return super().forward(X, E, y, node_mask)
        if not torch.all(node_mask):
            raise ValueError("Singleton batch must contain one real node per graph")
        result = super().forward(
            F.pad(X, (0, 0, 0, 1)),
            F.pad(E, (0, 0, 0, 1, 0, 1)), y,
            F.pad(node_mask, (0, 1), value=False))
        return utils.PlaceHolder(X=result.X[:, :1], E=result.E[:, :1, :1], y=result.y)


class SingletonGraphTransformer(SingletonForward, GraphTransformer):
    pass


class SingletonH1TwoHopTransformer(SingletonForward, H1TwoHopGraphTransformer):
    pass


class EmptyEdgeCrossEntropy(CrossEntropyMetric):
    def compute(self):
        # Empty edge targets carry zero total CE, rather than a 0/0 mean.
        if self.total_samples == 0:
            return self.total_ce
        return super().compute()


def adapt_backbone(model):
    """Change only Python forward dispatch, preserving parameter keys and RNG."""
    if type(model) is H1TwoHopGraphTransformer:
        model.__class__ = SingletonH1TwoHopTransformer
    elif type(model) is GraphTransformer:
        model.__class__ = SingletonGraphTransformer
    else:
        raise TypeError("Unrecognized backbone; singleton compatibility not applied")


class FourBenchmarkH2(ConditionalGraphBetaDiffusionH2):
    def __init__(self, cfg, dataset_infos, train_metrics=None, sampling_metrics=None,
                 visualization_tools=None, extra_features=None, domain_features=None):
        super().__init__(cfg, dataset_infos, train_metrics, sampling_metrics,
                         visualization_tools, extra_features, domain_features)
        if self.cfg.dataset.name not in DATASETS:
            raise ValueError("Four-benchmark model cannot alter an existing experiment")
        adapt_backbone(self.model)


class FourBenchmarkH1(ConditionalGraphBetaDiffusionH1):
    def __init__(self, cfg, dataset_infos, train_metrics=None, sampling_metrics=None,
                 visualization_tools=None, extra_features=None, domain_features=None):
        super().__init__(cfg, dataset_infos, train_metrics, sampling_metrics,
                         visualization_tools, extra_features, domain_features)
        if self.cfg.dataset.name not in DATASETS:
            raise ValueError("Four-benchmark model cannot alter an existing experiment")
        adapt_backbone(self.model)


class FourBenchmarkGlue(DiscreteDenoisingDiffusion):
    def __init__(self, cfg, dataset_infos, train_metrics=None, sampling_metrics=None,
                 visualization_tools=None, extra_features=None, domain_features=None):
        super().__init__(cfg, dataset_infos, train_metrics, sampling_metrics,
                         visualization_tools, extra_features, domain_features)
        if self.cfg.dataset.name not in DATASETS:
            raise ValueError("Four-benchmark model cannot alter an existing experiment")
        adapt_backbone(self.model)
        self.train_loss.edge_loss = EmptyEdgeCrossEntropy()


def model_class(stage):
    return {"h2": FourBenchmarkH2, "h1": FourBenchmarkH1,
            "glue": FourBenchmarkGlue}[stage]
