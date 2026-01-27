"""
Baseline models for FC/SC inputs with dual-stream encoders.
Each baseline encodes FC and SC separately, concatenates embeddings,
then feeds them to a shared prediction head.
Updated to include GIN, GraphSAGE and fixes for GCN/GAT.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops

from dial.model import GraphormerNodeEncoder


def build_edge_index_from_adj(adj: torch.Tensor, force_self_loops: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge index and weights from an adjacency matrix.
    Explicitly handles self-loops to prevent NaN in GCN normalization.
    """
    device = adj.device
    N = adj.shape[0]

    # Ensure no NaNs in input
    adj = torch.nan_to_num(adj, nan=0.0)

    # 1. Extract edges (excluding diagonal first to handle self-loops cleanly later)
    # Use torch.where to get indices
    edge_index = adj.nonzero().t()  # [2, E_all]
    row, col = edge_index[0], edge_index[1]
    edge_weight = adj[row, col]

    # 2. Filter out explicit zeros if any remain (usually nonzero handles this, but float precision...)
    mask = torch.abs(edge_weight) > 1e-6
    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask]

    # 3. Handle Self-loops
    # First remove existing self-loops to avoid duplicates/inconsistencies
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if force_self_loops:
        # Add self-loops with weight 1.0 (typical for GCN/GraphSAGE on connectivity graphs)
        # Note: For FC (correlation), self-loop is 1.0. For SC, it might vary, but fixing to 1.0 avoids 0-degree NaN.
        edge_index, edge_weight = add_self_loops(
            edge_index,
            edge_weight,
            fill_value=1.0,
            num_nodes=N
        )

    return edge_index, edge_weight


class MLPEncoder(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 128, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        in_dim = num_nodes * num_nodes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, adj: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # adj: [B, N, N]
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        B = adj.shape[0]
        x = adj.reshape(B, -1)
        return self.net(x)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        # 【关键修改 1】normalize=False
        # GCN默认的谱归一化在处理带负权重的图(如FC)时会产生NaN，必须关闭。
        # 【关键修改 2】add_self_loops=False
        # 我们在数据预处理阶段手动添加自环，避免重复添加。
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=False, normalize=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 【关键修改 3】添加 BN 层替代谱归一化

        self.conv2 = GCNConv(hidden_dim, embed_dim, add_self_loops=False, normalize=False)
        self.bn2 = nn.BatchNorm1d(embed_dim)  # 添加 BN 层

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                batch_vec: torch.Tensor) -> torch.Tensor:
        # 第一层卷积 -> BN -> 激活 -> Dropout
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        # 第二层卷积 -> BN -> 激活
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.bn2(x)
        x = self.act(x)

        # 池化
        x = global_mean_pool(x, batch_vec)
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, embed_dim: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # GAT usually doesn't take edge_weights in the standard PyG implementation (it computes attention)
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True, add_self_loops=True)
        # Input to conv2 is hidden_dim * heads
        self.conv2 = GATConv(hidden_dim * heads, embed_dim, heads=1, dropout=dropout, concat=False, add_self_loops=True)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        # Ignore edge_weight for GAT to avoid shape mismatch issues and rely on attention
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        x = global_mean_pool(x, batch_vec)
        return x


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embed_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        # GraphSAGE standard implementation also doesn't typically use edge weights,
        # but PyG implementation supports it? SAGEConv docs say no edge_weight.
        # So we ignore edge_weight.
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        x = global_mean_pool(x, batch_vec)
        return x


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        # GIN requires an MLP for the aggregation
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.conv1 = GINConv(self.mlp1, train_eps=True)

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.conv2 = GINConv(self.mlp2, train_eps=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch_vec: torch.Tensor) -> torch.Tensor:
        # GIN ignores edge_weight
        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch_vec)
        return x


class BaselineHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBaseline(nn.Module):
    def __init__(self, num_nodes: int, embed_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fc_encoder = MLPEncoder(num_nodes, hidden_dim, embed_dim, dropout)
        self.sc_encoder = MLPEncoder(num_nodes, hidden_dim, embed_dim, dropout)
        self.head = BaselineHead(embed_dim * 2, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, fc_adj: torch.Tensor, sc_adj: torch.Tensor) -> torch.Tensor:
        fc_emb = self.fc_encoder(fc_adj)
        sc_emb = self.sc_encoder(sc_adj)
        feat = torch.cat([fc_emb, sc_emb], dim=-1)
        return self.head(feat)


class _GraphConvDualStream(nn.Module):
    """
    Shared dual-stream wrapper for GNN encoders.
    """
    def __init__(self, encoder_ctor, num_nodes: int, hidden_dim: int, embed_dim: int, num_classes: int, dropout: float, **kwargs):
        super().__init__()
        # We pass num_nodes as in_dim because we use adjacency rows as features (or Identity)
        self.fc_encoder = encoder_ctor(in_dim=num_nodes, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout, **kwargs)
        self.sc_encoder = encoder_ctor(in_dim=num_nodes, hidden_dim=hidden_dim, embed_dim=embed_dim, dropout=dropout, **kwargs)

        self.head = BaselineHead(embed_dim * 2, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

    def _encode_single(self, adj: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """
        adj: [N, N]
        """
        N = adj.shape[0]
        # Robustly build edge index with self-loops to prevent GCN NaN
        edge_index, edge_weight = build_edge_index_from_adj(adj, force_self_loops=True)

        # Use adjacency matrix rows as node features.
        # Ensure it's on the right device and requires_grad is handled if needed (usually data isn't leaf variable)
        x = adj.clone()

        batch_vec = torch.zeros(N, dtype=torch.long, device=adj.device)

        return encoder(x, edge_index, edge_weight, batch_vec).squeeze(0)

    def forward(self, fc_adj: torch.Tensor, sc_adj: torch.Tensor) -> torch.Tensor:
        if fc_adj.dim() == 2: fc_adj = fc_adj.unsqueeze(0)
        if sc_adj.dim() == 2: sc_adj = sc_adj.unsqueeze(0)

        fc_emb = []
        sc_emb = []
        # Iterate over batch (Graph Baselines usually handle variable graphs, but here N is fixed mostly)
        for b in range(fc_adj.shape[0]):
            fc_emb.append(self._encode_single(fc_adj[b], self.fc_encoder))
            sc_emb.append(self._encode_single(sc_adj[b], self.sc_encoder))

        feat = torch.cat([torch.stack(fc_emb, dim=0), torch.stack(sc_emb, dim=0)], dim=-1)
        return self.head(feat)


class GCNBaseline(_GraphConvDualStream):
    def __init__(self, num_nodes: int, hidden_dim: int = 64, embed_dim: int = 128, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(GCNEncoder, num_nodes, hidden_dim, embed_dim, num_classes, dropout)

class GATBaseline(_GraphConvDualStream):
    def __init__(self, num_nodes: int, hidden_dim: int = 64, embed_dim: int = 128, heads: int = 4, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(GATEncoder, num_nodes, hidden_dim, embed_dim, num_classes, dropout, heads=heads)

class GraphSAGEBaseline(_GraphConvDualStream):
    def __init__(self, num_nodes: int, hidden_dim: int = 64, embed_dim: int = 128, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(SAGEEncoder, num_nodes, hidden_dim, embed_dim, num_classes, dropout)

class GINBaseline(_GraphConvDualStream):
    def __init__(self, num_nodes: int, hidden_dim: int = 64, embed_dim: int = 128, num_classes: int = 2, dropout: float = 0.1):
        super().__init__(GINEncoder, num_nodes, hidden_dim, embed_dim, num_classes, dropout)


class GraphormerBaseline(nn.Module):
    expects_graphormer_inputs = True

    def __init__(
            self,
            num_nodes: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            num_classes: int = 2,
            dropout: float = 0.1,
            max_degree: int = 511,
            max_path_len: int = 5,
    ):
        super().__init__()
        encoder_kwargs = dict(
            N=num_nodes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_degree=max_degree,
            max_path_len=max_path_len,
        )
        self.fc_encoder = GraphormerNodeEncoder(**encoder_kwargs)
        self.sc_encoder = GraphormerNodeEncoder(**encoder_kwargs)
        self.head = BaselineHead(
            in_dim=d_model * 2,
            hidden_dim=dim_feedforward // 2,
            num_classes=num_classes,
            dropout=dropout
        )
        self.graph_dropout = nn.Dropout(dropout * 0.5)

    @staticmethod
    def _pool_nodes(node_repr: torch.Tensor) -> torch.Tensor:
        return node_repr.mean(dim=1)

    def forward(
            self,
            fc_node_feat: torch.Tensor,
            fc_in_degree: torch.Tensor,
            fc_out_degree: torch.Tensor,
            fc_path_data: torch.Tensor,
            fc_dist: torch.Tensor,
            fc_attn_mask: torch.Tensor,
            sc_node_feat: torch.Tensor,
            sc_in_degree: torch.Tensor,
            sc_out_degree: torch.Tensor,
            sc_path_data: torch.Tensor,
            sc_dist: torch.Tensor,
            sc_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        fc_repr = self.fc_encoder(
            fc_node_feat, fc_in_degree, fc_out_degree, fc_path_data, fc_dist, fc_attn_mask
        )
        sc_repr = self.sc_encoder(
            sc_node_feat, sc_in_degree, sc_out_degree, sc_path_data, sc_dist, sc_attn_mask
        )

        fc_graph = self.graph_dropout(self._pool_nodes(fc_repr))
        sc_graph = self.graph_dropout(self._pool_nodes(sc_repr))
        feat = torch.cat([fc_graph, sc_graph], dim=-1)
        return self.head(feat)