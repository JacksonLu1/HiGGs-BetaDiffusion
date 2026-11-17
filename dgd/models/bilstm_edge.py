"""BiLSTM aggregation of symmetric two-hop paths for the H1 edge update."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.two_hop_edge import TwoHopEdgeResidual


class BiLSTMEdgeUpdate(TwoHopEdgeResidual):
    """Aggregate all two-hop path tokens in an order defined by their features."""

    PAIR_CHUNK = 256

    def __init__(self, edge_dim, global_dim, channels=16):
        if type(channels) is not int or channels <= 0 or channels % 2:
            raise ValueError("BiLSTM path channels must be a positive even integer")
        super().__init__(edge_dim, global_dim, channels, normalization="mean", gate_mode="gru")
        self.normalization = "bilstm"
        self.gate_mode = "bilstm_gru"
        self.sequence = nn.LSTM(channels, channels // 2, batch_first=True, bidirectional=True)

    @staticmethod
    def _lexicographic_tokens(tokens):
        pair_count, length, channels = tokens.shape
        with torch.no_grad():
            order = torch.arange(length, device=tokens.device).expand(pair_count, length)
            # Stable sorting from the last feature to the first uses all features
            # to break ties. Equal complete tokens can appear in either order.
            for feature in reversed(range(channels)):
                values = tokens[..., feature].gather(1, order)
                order = order.gather(1, values.argsort(dim=1, stable=True))
        return tokens.gather(1, order[..., None].expand(-1, -1, channels))

    def _summarize_pairs(self, left, right, nodes, pair_positions):
        start, end = pair_positions
        # Enumerate compact node positions other than these endpoints. Because
        # start < end, each skip adjusts the index for the skips that precede it.
        positions = torch.arange(nodes.numel() - 2, device=nodes.device)
        positions = positions + (positions >= start[:, None])
        positions = positions + (positions >= end[:, None])
        middle = nodes[positions]
        start = nodes[start, None]
        end = nodes[end, None]
        # Keep path products and recurrence in the LSTM's parameter dtype.
        # In particular CPU bf16 autocast does not always support BiLSTM.
        dtype = self.sequence.weight_ih_l0.dtype
        with torch.autocast(device_type=left.device.type, enabled=False):
            first = left[start, middle].to(dtype) * right[middle, end].to(dtype)
            second = left[end, middle].to(dtype) * right[middle, start].to(dtype)
            tokens = self._lexicographic_tokens((first + second) * 0.5)
            _, (hidden, _) = self.sequence(tokens)
        return hidden.transpose(0, 1).reshape(-1, self.channels)

    def forward(self, edges, global_features, node_mask):
        if edges.ndim != 4 or edges.size(1) != edges.size(2):
            raise ValueError("Expected edge tensor [batch, nodes, nodes, channels]")
        batch, n, _, edge_dim = edges.shape
        if node_mask.shape != (batch, n) or node_mask.dtype != torch.bool:
            raise ValueError("Expected boolean node_mask [batch, nodes]")
        if global_features.ndim != 2 or global_features.size(0) != batch:
            raise ValueError("Expected global features [batch, channels]")
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        valid = valid & ~torch.eye(n, device=edges.device, dtype=torch.bool)[None]
        clean = edges.masked_fill(~valid[..., None], 0)
        clean = (clean + clean.transpose(1, 2)) * 0.5
        if batch == 0:
            return clean

        left = self.left(clean).tanh()
        right = self.right(clean).tanh()
        results = []
        for index, mask in enumerate(node_mask):
            nodes = mask.nonzero().flatten()
            count = nodes.numel()
            if count < 2:
                results.append(clean[index])
                continue

            pairs = torch.triu_indices(count, count, offset=1, device=nodes.device)
            start, end = nodes[pairs]
            edge_vectors = clean[index, start, end]
            if count == 2:
                message = clean.new_zeros((1, self.channels))
            else:
                chunks = []
                for pair_chunk in pairs.split(self.PAIR_CHUNK, dim=1):
                    inputs = (left[index], right[index], nodes, pair_chunk)
                    if self.training and torch.is_grad_enabled():
                        # Recompute path products and recurrence in backward.
                        # Checkpoint inputs reference the quadratic projections.
                        chunk = checkpoint(self._summarize_pairs, *inputs, use_reentrant=False)
                    else:
                        chunk = self._summarize_pairs(*inputs)
                    chunks.append(chunk)
                message = torch.cat(chunks, dim=0).to(clean.dtype)

            update = self.gru(self.output(torch.cat((edge_vectors, message), dim=-1)), edge_vectors)
            # One calculation per unordered pair; the reverse edge is identical.
            locations = torch.cat((start * n + end, end * n + start))
            values = torch.cat((update, update), dim=0).to(clean.dtype)
            dense = clean.new_zeros((n * n, edge_dim)).index_copy(0, locations, values)
            results.append(dense.reshape(n, n, edge_dim))
        return torch.stack(results)
