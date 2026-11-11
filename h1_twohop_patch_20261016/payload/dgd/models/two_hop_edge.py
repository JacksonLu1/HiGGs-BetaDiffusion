"""H1-only dense two-hop edge residual, inspired by Local PPGN.

This is an adaptation of edge-wise two-hop messages, not the paper's sparse
local-expansion generator. The forward/reverse diffusion kernels are untouched.
"""

from collections.abc import Mapping

import torch
from torch import nn


def two_hop_options(value=None):
    defaults = dict(enabled=False, channels=16, warm_start_checkpoint=None)
    if value is not None:
        if not isinstance(value, Mapping):
            raise TypeError("model.two_hop must be a mapping")
        unknown = set(value) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown model.two_hop options: {sorted(unknown)}")
        defaults.update(value)
    if type(defaults['enabled']) is not bool:
        raise TypeError("model.two_hop.enabled must be a bool")
    channels = defaults['channels']
    if type(channels) is not int or not 1 <= channels <= 128:
        raise ValueError("model.two_hop.channels must be an integer in [1, 128]")
    path = defaults['warm_start_checkpoint']
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise TypeError("model.two_hop.warm_start_checkpoint must be a path or null")
    if path is not None and not defaults['enabled']:
        raise ValueError("two-hop warm start requires model.two_hop.enabled=true")
    return defaults


class TwoHopEdgeResidual(nn.Module):
    """Add a masked, symmetric residual to hidden edge features.

    Products are reduced over valid intermediate nodes without allocating an
    [N,N,N,C] tensor. Dense matmul still costs O(B*C*N^3); use H1 communities.
    The zero output initialization makes the initial residual exactly zero.
    """

    def __init__(self, edge_dim, global_dim, channels=16):
        super().__init__()
        if min(edge_dim, global_dim, channels) <= 0:
            raise ValueError("Two-hop dimensions must be positive")
        self.channels = channels
        self.left = nn.Linear(edge_dim, channels)
        self.right = nn.Linear(edge_dim, channels)
        self.output = nn.Sequential(
            nn.Linear(edge_dim + channels, edge_dim), nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.gate = nn.Linear(global_dim, 1)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, edges, global_features, node_mask):
        if edges.ndim != 4 or edges.size(1) != edges.size(2):
            raise ValueError("Expected edge tensor [batch, nodes, nodes, channels]")
        batch, n, _, _ = edges.shape
        if node_mask.shape != (batch, n) or node_mask.dtype != torch.bool:
            raise ValueError("Expected boolean node_mask [batch, nodes]")
        if global_features.ndim != 2 or global_features.size(0) != batch:
            raise ValueError("Expected global features [batch, channels]")
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        valid = valid & ~torch.eye(n, device=edges.device, dtype=torch.bool)[None]
        clean = edges.masked_fill(~valid[..., None], 0)
        clean = (clean + clean.transpose(1, 2)) * 0.5
        left = self.left(clean).tanh().masked_fill(~valid[..., None], 0)
        right = self.right(clean).tanh().masked_fill(~valid[..., None], 0)
        left = left.permute(0, 3, 1, 2).reshape(batch * self.channels, n, n)
        right = right.permute(0, 3, 1, 2).reshape(batch * self.channels, n, n)
        # Keep reductions finite with mixed precision; float64 remains float64.
        with torch.autocast(device_type=edges.device.type, enabled=False):
            dtype = torch.float64 if edges.dtype == torch.float64 else torch.float32
            messages = torch.bmm(left.to(dtype), right.to(dtype))
        messages = messages.reshape(batch, self.channels, n, n).permute(0, 2, 3, 1)
        normalizer = (node_mask.sum(-1).to(messages.dtype) - 2).clamp_min(1).sqrt()
        messages = messages / normalizer[:, None, None, None]
        messages = ((messages + messages.transpose(1, 2)) * 0.5).to(edges.dtype)
        residual = self.output(torch.cat((clean, messages), dim=-1))
        residual = residual * self.gate(global_features).sigmoid()[:, None, None, :]
        residual = (residual + residual.transpose(1, 2)) * 0.5
        return edges + residual.masked_fill(~valid[..., None], 0)
