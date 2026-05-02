"""
models/gcn.py

GCN baseline for cylinder flow next-step prediction.

Architecture:
    Encoder  : Linear(7 -> hidden_dim)
    Processor: stack of GCNConv layers with ReLU + residual connections
    Decoder  : Linear(hidden_dim -> 3)

Input  : Data.x  (N, 7)  -- [u, v, pressure, node_type_onehot x4]
Output : (N, 3)           -- predicted [u, v, pressure] at t+1
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=64, out_dim=3, num_layers=3):
        """
        Args:
            in_dim     : number of input node features (7)
            hidden_dim : width of hidden layers
            out_dim    : number of output features (3 = u, v, pressure)
            num_layers : number of GCNConv message-passing layers
        """
        super().__init__()

        # Encoder: project raw features into hidden space
        self.encoder = nn.Linear(in_dim, hidden_dim)

        # Processor: stack of GCNConv layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Decoder: project back to fluid state
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Encode
        x = self.encoder(x)         # (N, hidden_dim)
        x = torch.relu(x)

        # Message passing with residual connections
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)  # (N, hidden_dim)
            x = norm(x)
            x = torch.relu(x)
            x = x + residual         # residual connection

        # Decode
        out = self.decoder(x)        # (N, 3)
        return out
