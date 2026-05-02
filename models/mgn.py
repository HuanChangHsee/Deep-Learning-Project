
"""
models/mgn.py

MeshGraphNets variant for cylinder flow next-step prediction.
Reference: Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021.

Architecture:
    Node Encoder  : MLP(7  -> hidden_dim)
    Edge Encoder  : MLP(3  -> hidden_dim)
    Processor     : stack of message-passing blocks, each updating
                    edge repr then node repr (both with residuals)
    Decoder       : MLP(hidden_dim -> 3)

Key difference from GCN:
    - Edge features (dx, dy, distance) are explicitly encoded and updated
      during message passing, giving the model directional geometric context.
    - Predicts the residual (delta) rather than absolute next state.

Input  : Data.x        (N, 7)  -- [u, v, pressure, node_type_onehot x4]
         Data.edge_attr (E, 3)  -- [dx, dy, distance]
Output : (N, 3)                 -- predicted [u, v, pressure] at t+1
"""

import torch
import torch.nn as nn


def build_mlp(in_dim, hidden_dim, out_dim, num_layers=2, activate_final=False):
    """
    Utility: build a small MLP with LayerNorm on the output.
    Used for all encoders, edge/node update functions, and decoder.
    """
    layers = []
    dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
        elif activate_final:
            layers.append(nn.ReLU())
    layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class MGNBlock(nn.Module):
    """
    One message-passing block:
        1. Compute edge messages from (src_node, dst_node, edge) features
        2. Aggregate messages at each node
        3. Update node features from (node, aggregated_messages)
    Both edge and node updates use residual connections.
    """

    def __init__(self, hidden_dim):
        super().__init__()

        # Edge update: takes [src, dst, edge] -> new edge repr
        self.edge_fn = build_mlp(
            in_dim=hidden_dim * 3,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim
        )

        # Node update: takes [node, aggregated_edges] -> new node repr
        self.node_fn = build_mlp(
            in_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim
        )

    def forward(self, node_feat, edge_feat, edge_index):
        src, dst = edge_index[0], edge_index[1]

        # Edge update
        edge_input = torch.cat([node_feat[src], node_feat[dst], edge_feat], dim=-1)
        edge_feat = edge_feat + self.edge_fn(edge_input)   # residual

        # Aggregate messages at each node
        num_nodes = node_feat.shape[0]
        agg = torch.zeros(num_nodes, edge_feat.shape[-1],
                          device=node_feat.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(edge_feat), edge_feat)

        # Node update
        node_input = torch.cat([node_feat, agg], dim=-1)
        node_feat = node_feat + self.node_fn(node_input)   # residual

        return node_feat, edge_feat


class MeshGraphNets(nn.Module):
    def __init__(self, in_node_dim=7, in_edge_dim=3,
                 hidden_dim=128, out_dim=3, num_layers=3):
        """
        Args:
            in_node_dim : input node feature size (7)
            in_edge_dim : input edge feature size (3 = dx, dy, dist)
            hidden_dim  : hidden representation size
            out_dim     : output size (3 = u, v, pressure)
            num_layers  : number of message-passing blocks
        """
        super().__init__()

        # Encoders
        self.node_encoder = build_mlp(in_node_dim, hidden_dim, hidden_dim)
        self.edge_encoder = build_mlp(in_edge_dim, hidden_dim, hidden_dim)

        # Processor: stack of MGNBlocks
        self.blocks = nn.ModuleList([
            MGNBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Decoder: predict residual delta
        self.decoder = build_mlp(hidden_dim, hidden_dim, out_dim,
                                 activate_final=False)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode nodes and edges into hidden space
        node_feat = self.node_encoder(x)            # (N, hidden_dim)
        edge_feat = self.edge_encoder(edge_attr)    # (E, hidden_dim)

        # Message passing
        for block in self.blocks:
            node_feat, edge_feat = block(node_feat, edge_feat, edge_index)

        # Decode: predict residual (delta state)
        delta = self.decoder(node_feat)             # (N, 3)

        # Add residual to current normalized state (first 3 features of x are u,v,p)
        out = data.x[:, :3] + delta                 # (N, 3)

        return out
