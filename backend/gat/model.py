"""3-layer Graph Attention Network for nodal pressure head prediction."""
import sys
import os
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, LayerNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class PressureGAT(nn.Module):
    """
    3-layer GAT: 8 node features, 6 edge features → (N,) pressure heads.
    Node feature 7 is 1/(1+hop_distance) from the failed pipe's endpoints —
    an explicit proximity-to-failure signal, since the failure itself only
    changes 2 of ~234 directed edges (a ~0.85% perturbation) and 3 rounds of
    message passing alone struggled to propagate that sparse edge bit to the
    rest of the graph (see backend/wntr_engine.py:extract_graph_features).
    ~180K parameters.
    """

    def __init__(self, node_in: int = 8, edge_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        self.node_in = node_in
        self.conv1 = GATConv(node_in, 64, heads=4, edge_dim=edge_dim, concat=True, dropout=dropout)
        self.norm1 = LayerNorm(256)
        self.conv2 = GATConv(256, 64, heads=4, edge_dim=edge_dim, concat=True, dropout=dropout)
        self.norm2 = LayerNorm(256)
        self.conv3 = GATConv(256, 32, heads=2, edge_dim=edge_dim, concat=True, dropout=dropout)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.norm1(h)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.conv3(h, edge_index, edge_attr)
        h = self.relu(h)

        return self.out(h).squeeze(-1)

    def physics_loss_term(
        self,
        pred_heads: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        demands: torch.Tensor,
        lambda_phys: float = 0.01,
    ) -> torch.Tensor:
        """
        Soft physics regularization: penalize flow balance violations.
        Uses Hazen-Williams approximation to estimate inter-node flows.
        """
        src, dst = edge_index[0], edge_index[1]
        L = edge_attr[:, 0].clamp(min=1e-3)      # normalized log-length
        D = edge_attr[:, 1].clamp(min=1e-3)      # normalized diameter
        C = edge_attr[:, 2].clamp(min=1e-3)      # roughness

        dH = (pred_heads[src] - pred_heads[dst]).clamp(min=0)
        # Approximate flow using inverse HW: Q ∝ (dH * C^1.852 * D^4.87 / L)^(1/1.852)
        # Using simplified normalized version
        resistance = (L.abs() / (C.pow(1.852) * D.pow(4.87) + 1e-9)).clamp(min=1e-9)
        # clamp min=1e-4 to avoid near-zero base causing infinite gradient of pow(1/1.852)
        Q_est = (dH / (10.67 * resistance + 1e-9)).clamp(min=1e-4).pow(1.0 / 1.852)

        # Flow balance at each non-reservoir node
        N = pred_heads.shape[0]
        in_flow = torch.zeros(N, device=pred_heads.device)
        out_flow = torch.zeros(N, device=pred_heads.device)
        in_flow.scatter_add_(0, dst, Q_est)
        out_flow.scatter_add_(0, src, Q_est)

        balance = (in_flow - out_flow - demands.clamp(min=0)).abs()
        # Only penalize demand nodes (non-reservoir = demand > 0 or not marked reservoir)
        demand_mask = demands > 0
        if demand_mask.sum() > 0:
            return lambda_phys * balance[demand_mask].mean()
        return torch.tensor(0.0, device=pred_heads.device)
