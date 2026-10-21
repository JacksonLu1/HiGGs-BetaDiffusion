"""Opt-in H1 backbone; the shared HiGGs GraphTransformer is not edited."""

import torch

import utils
from models.transformer_model import GraphTransformer
from models.two_hop_edge import TwoHopEdgeResidual


class H1TwoHopGraphTransformer(GraphTransformer):
    def __init__(self, *args, two_hop_channels=16, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = kwargs['hidden_dims']
        self.two_hop_edge = TwoHopEdgeResidual(
            int(hidden['de']), int(hidden['dy']), two_hop_channels)

    def forward(self, X, E, y, node_mask):
        # Same forward as the server backbone, plus one hidden-edge residual.
        bs, n = X.shape[:2]
        diag_mask = ~torch.eye(n, device=E.device, dtype=torch.bool)
        diag_mask = diag_mask[None, :, :, None].expand(bs, -1, -1, -1)
        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = utils.PlaceHolder(
            X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y
        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)
        E = self.two_hop_edge(E, y, node_mask)
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)
        X = X + X_to_out
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out
        E = 1 / 2 * (E + E.transpose(1, 2))
        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
