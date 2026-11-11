"""HOG-Diff RW/SPD input subcomponent, NOT the complete HOG-Diff model.

get_rw_feat is adapted from Yiming Huang and Tolga Birdal, HOG-Diff (2026):
https://github.com/Yiminghh/HOG-Diff/blob/main/models/utils.py
Upstream Git blob: 5d72f438b1122d6f05b6ca9e95b135691f5b1eff.
Copyright (c) 2026 Yiming Huang and Tolga Birdal.
SPDX-License-Identifier: CC-BY-NC-4.0
https://creativecommons.org/licenses/by-nc/4.0/
See THIRD_PARTY_NOTICES_HOG_RW.md for provenance and licensing.

Changes: comments/style, strict opt-in configuration, and an external HiGGs
masking wrapper. The original powers P^2,...,P^(k+1) and zero-count SPD encoding
are retained. This is NOT a replacement with exact shortest-path distances.
No claim of a new encoding algorithm, complete ScoreNet or HOG-Diff replication.
"""

from collections.abc import Mapping
import torch

HOG_RW_ALGORITHM = "HOG-Diff:utils.py:5d72f438:get_rw_feat"


def hog_rw_options(config=None):
    defaults = {"enabled": False, "rw_depth": 4}
    if config is None:
        return defaults
    if not isinstance(config, Mapping):
        raise TypeError("model.hog_rw must be a mapping")
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown hog_rw options: {sorted(unknown)}")
    result = dict(defaults, **dict(config))
    if type(result["enabled"]) is not bool:
        raise TypeError("hog_rw.enabled must be bool")
    if type(result["rw_depth"]) is not int or not 1 <= result["rw_depth"] <= 32:
        raise ValueError("hog_rw.rw_depth must be an integer in [1, 32]")
    return result


@torch.no_grad()
def get_rw_feat(k_step, dense_adj):
    """Original HOG-Diff computation; returns RW landing and SPD channels."""
    rw_list = []
    deg = dense_adj.sum(-1, keepdims=True)
    AD = dense_adj / (deg + 1e-8)
    rw_list.append(AD)
    for _ in range(k_step):
        rw = torch.bmm(rw_list[-1], AD)
        rw_list.append(rw)
    rw_map = torch.stack(rw_list[1:], dim=1)
    rw_landing = torch.diagonal(rw_map, offset=0, dim1=2, dim2=3)
    rw_landing = rw_landing.permute(0, 2, 1)
    tmp_rw = rw_map.sort(dim=1)[0]
    spd_ind = (tmp_rw <= 0).sum(dim=1)
    spd_onehot = torch.nn.functional.one_hot(
        spd_ind, num_classes=k_step + 1).to(torch.float)
    spd_onehot = spd_onehot.permute(0, 3, 1, 2)
    return rw_landing, spd_onehot


@torch.no_grad()
def masked_hog_rw_features(binary_adj, node_mask, rw_depth):
    """Append-only HiGGs adapter; output X=[B,N,k], E=[B,N,N,k+1]."""
    if binary_adj.ndim != 3 or binary_adj.shape[1] != binary_adj.shape[2]:
        raise ValueError("binary_adj must have shape [B,N,N]")
    if node_mask.dtype != torch.bool or node_mask.shape != binary_adj.shape[:2]:
        raise ValueError("node_mask must be bool [B,N]")
    hog_rw_options({"enabled": True, "rw_depth": rw_depth})
    if not torch.isfinite(binary_adj).all():
        raise ValueError("Nonfinite adjacency passed to HOG RW/SPD")
    if not ((binary_adj == 0) | (binary_adj == 1)).all():
        raise ValueError("HOG RW/SPD adapter expects a binary adjacency")
    n = binary_adj.shape[1]
    valid = node_mask[:, :, None] & node_mask[:, None, :]
    diagonal = torch.eye(n, device=binary_adj.device, dtype=torch.bool)[None]
    valid_edge = valid & ~diagonal
    adj = binary_adj.float() * valid_edge
    if not torch.equal(adj, adj.transpose(1, 2)):
        raise ValueError("HOG RW/SPD requires an undirected adjacency")
    rw, spd = get_rw_feat(rw_depth, adj)
    rw = rw * node_mask.unsqueeze(-1)
    spd = spd.permute(0, 2, 3, 1) * valid_edge.unsqueeze(-1)
    return rw, spd
