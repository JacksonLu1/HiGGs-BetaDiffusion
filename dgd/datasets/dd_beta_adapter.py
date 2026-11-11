"""DD-only compatibility; existing datasets and Beta transitions are unchanged."""
import torch
from diffusion.distributions import DistributionNodes
from datasets.graph_collection import GraphCollectionInfos


class DDDistributionNodes(DistributionNodes):
    def log_prob(self, batch_n_nodes):
        if batch_n_nodes.ndim != 1:
            raise ValueError("Expected a one-dimensional node-count tensor")
        p = self.prob.to(device=batch_n_nodes.device)
        valid = (batch_n_nodes >= 0) & (batch_n_nodes < p.numel())
        probas = torch.zeros_like(batch_n_nodes, dtype=p.dtype)
        probas[valid] = p[batch_n_nodes[valid]]
        return torch.log(probas + 1e-30)


class DDBetaDatasetInfos(GraphCollectionInfos):
    def __init__(self, datamodule, dataset_config):
        if dataset_config.name != "dd":
            raise ValueError("This adapter is scoped to DD only")
        super().__init__(datamodule, dataset_config)
        self.nodes_dist = DDDistributionNodes(self.n_nodes)
