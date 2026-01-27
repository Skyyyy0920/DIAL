"""
Model of DIAL
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder

from .routing import compute_edge_energy, mask_from_energy
from .loss import compute_losses
from .utils import build_edge_index_from_S


# ============================================================================
# Node Encoder - Graphormer
# ============================================================================

class GraphormerNodeEncoder(nn.Module):
    """Graphormer node encoder implemented with DGL GraphormerLayer."""

    def __init__(
            self,
            N: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            num_spatial: int = 510,
            max_degree: int = 511,  # +1 = 512
            max_path_len: int = 5,
            edge_feat_dim: int = 1,
    ):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.nhead = nhead

        self.fc_feature_proj = nn.Sequential(
            nn.Linear(N, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        self.degree_encoder = DegreeEncoder(max_degree=max_degree, embedding_dim=d_model, direction='in')
        self.spatial_encoder = SpatialEncoder(max_dist=num_spatial, num_heads=nhead)
        self.path_encoder = PathEncoder(max_len=max_path_len, feat_dim=edge_feat_dim, num_heads=nhead)

        # Virtual node (graph token)
        self.graph_token = nn.Embedding(1, d_model)
        self.graph_token_virtual_distance = nn.Embedding(1, nhead)

        self.layers = nn.ModuleList([
            GraphormerLayer(
                feat_size=d_model,
                hidden_size=dim_feedforward,
                num_heads=nhead,
                dropout=dropout,
                activation=nn.GELU(),
                norm_first=False
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self,
            node_feat: torch.Tensor,
            in_degree: torch.Tensor,
            out_degree: torch.Tensor,
            path_data: torch.Tensor,
            dist: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = node_feat.shape

        # Project node features
        x = self.fc_feature_proj(node_feat)  # [batch, N, d_model]

        # Add degree encoding
        degree_emb = self.degree_encoder(in_degree)
        H = x + degree_emb

        # Add virtual node
        graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # [batch, 1, d_model]
        H = torch.cat([graph_token_feat, H], dim=1)  # [batch, N+1, d_model]

        # Prepare attention bias
        attn_bias = torch.zeros(
            batch_size,
            num_nodes + 1,
            num_nodes + 1,
            self.nhead,
            device=dist.device,
        )  # [batch_size, N+1, N+1, attention_head]

        dist_long = dist.long()
        path_encoding = self.path_encoder(dist_long, path_data)
        spatial_encoding = self.spatial_encoder(dist_long)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        # Virtual node spatial encoding
        t = self.graph_token_virtual_distance.weight.reshape(1, 1, self.nhead)
        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        for layer in self.layers:
            H = layer(H, attn_mask=attn_mask, attn_bias=attn_bias)

        return self.norm(H[:, 1:, :])  # Remove virtual node


class EdgeGate(nn.Module):
    """Edge gating network that produces per-edge gate values a_e."""

    def __init__(self, d_model: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        edge_feature_dim = 2 * d_model + 2

        self.mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
            self,
            H: torch.Tensor,
            edge_index: torch.Tensor,
            S_e: torch.Tensor,
            F_e: torch.Tensor
    ) -> torch.Tensor:
        H_u = H[edge_index[0]]
        H_v = H[edge_index[1]]
        H_diff = torch.abs(H_u - H_v)
        H_prod = H_u * H_v

        edge_features = torch.cat([
            H_diff,  # [E, d]
            H_prod,  # [E, d]
            S_e.unsqueeze(1),  # [E, 1]
            F_e.unsqueeze(1)  # [E, 1]
        ], dim=1)

        a_e = self.mlp(edge_features).squeeze(1)  # [E, 1]
        return a_e


# ============================================================================
# Transformer with mask to get final representation
# ============================================================================

class MaskedGraphTransformer(nn.Module):
    """Masked graph Transformer that learns representations on softly-masked subgraphs."""

    def __init__(
            self,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout: float = 0.3
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
            self,
            H: torch.Tensor,  # List of [B, N, d_model]
            edge_index_list: List[torch.Tensor],  # List of [2, E_i]
            m_list: List[torch.Tensor],  # List of [E_i]
            eps: float = 1e-9
    ) -> torch.Tensor:
        device = H[0].device
        B, N, _ = H.shape

        attn_bias_list = []
        for i in range(B):
            edge_index = edge_index_list[i]
            m = m_list[i]
            attn_bias = self._build_attention_bias(N, edge_index, m, eps, device)
            attn_bias_list.append(attn_bias)

        attn_bias_batch = torch.stack(attn_bias_list, dim=0)  # [B, N, N]

        attn_bias_batch = attn_bias_batch.repeat_interleave(self.nhead, dim=0)  # [B*nhead, N, N]

        Z = self.transformer(H, mask=attn_bias_batch)
        Z = self.norm(Z)

        return Z

    def _build_attention_bias(
            self,
            N: int,
            edge_index: torch.Tensor,
            m: torch.Tensor,
            eps: float,
            device: torch.device
    ) -> torch.Tensor:
        """Construct attention bias for an individual graph using the soft mask."""
        # Initialize non-edges with -inf so they are ignored during attention.
        attn_bias = torch.full((N, N), float('-inf'), device=device)
        attn_bias.fill_diagonal_(0.0)  # Preserve self-loops

        # Assign bias for every valid edge: log(m_e + eps)
        row, col = edge_index[0], edge_index[1]
        bias_values = torch.log(m + eps)
        attn_bias[row, col] = bias_values
        attn_bias[col, row] = bias_values  # Mirror to keep the graph undirected

        return attn_bias


# ============================================================================
# Prediction Head
# ============================================================================


class PredictionHead(nn.Module):
    def __init__(
            self,
            d_model: int = 64,
            num_classes: int = 2,
            task: str = 'classification',
            hidden_dim: int = 32,
            dropout: float = 0.2
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.task = task

        if task == 'classification':
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif task == 'regression':
            self.mlp = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

        self.graph_repr_dropout = nn.Dropout(dropout * 0.5)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        graph_repr = Z.mean(dim=1)
        graph_repr = self.graph_repr_dropout(graph_repr)

        y_pred = self.mlp(graph_repr)

        if self.task == 'regression':
            y_pred = y_pred.squeeze(-1)

        return y_pred


# ============================================================================
# Top-level model
# ============================================================================

class DIALModel(nn.Module):
    """End-to-end energy-driven masking model."""

    def __init__(
            self,
            N: int,
            d_model: int = 64,
            nhead: int = 4,
            num_node_layers: int = 2,
            num_graph_layers: int = 2,
            dim_feedforward: int = 128,
            num_classes: int = 2,
            task: str = 'classification',
            dropout: float = 0.1,
            tau: float = 8.0,
            lambda_cost: float = 1.0,
            delta: float = 1e-6
    ):
        super().__init__()

        self.N = N
        self.d_model = d_model
        self.task = task
        self.tau = tau
        self.lambda_cost = lambda_cost
        self.delta = delta

        self.node_encoder = GraphormerNodeEncoder(
            N=N,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_node_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.edge_gate = EdgeGate(
            d_model=d_model,
            hidden_dim=dim_feedforward // 2,
            dropout=dropout
        )

        self.threshold = nn.Parameter(torch.tensor(0.0))

        self.graph_transformer = MaskedGraphTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_graph_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.prediction_head = PredictionHead(
            d_model=d_model,
            num_classes=num_classes,
            task=task,
            hidden_dim=dim_feedforward // 2
        )

    def forward(
            self,
            node_feat: torch.Tensor,
            in_degree: torch.Tensor,
            out_degree: torch.Tensor,
            path_data: torch.Tensor,
            dist: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            S: torch.Tensor,
            F: torch.Tensor,
            y: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch_size = node_feat.shape[0]
        H = self.node_encoder(node_feat, in_degree, out_degree, path_data, dist, attn_mask)

        energy_list: List[torch.Tensor] = []
        edge_index_list: List[torch.Tensor] = []
        m_list: List[torch.Tensor] = []

        for idx in range(batch_size):
            energies, edge_index, S_e, _ = compute_edge_energy(
                S[idx], F[idx], H[idx],
                edge_gate=self.edge_gate,
                delta=self.delta
            )

            m = mask_from_energy(
                energies,
                S_e=S_e,
                tau=self.tau,
                lambda_cost=self.lambda_cost,
                threshold=self.threshold
            )

            energy_list.append(energies)
            edge_index_list.append(edge_index)
            m_list.append(m)

        Z = self.graph_transformer(H, edge_index_list, m_list)
        y_pred = self.prediction_head(Z)

        loss_dict = compute_losses(
            y_pred=y_pred,
            y=y,
            task=self.task,
        )
        avg_loss = loss_dict['loss']

        return y_pred, avg_loss

    def inference(
            self,
            node_feat: torch.Tensor,
            in_degree: torch.Tensor,
            out_degree: torch.Tensor,
            path_data: torch.Tensor,
            dist: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            S: torch.Tensor,
            F: torch.Tensor,
            budget_lambda: Optional[float] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Batched inference routine that mirrors training-time soft routing.
        """
        self.eval()
        batch_size = node_feat.shape[0]

        with torch.no_grad():
            H = self.node_encoder(node_feat, in_degree, out_degree, path_data, dist, attn_mask)

            edge_index_list: List[torch.Tensor] = []
            m_list: List[torch.Tensor] = []

            for idx in range(batch_size):
                energies, edge_index, S_e, _ = compute_edge_energy(
                    S[idx], F[idx], H[idx],
                    edge_gate=self.edge_gate,
                    delta=self.delta
                )

                m_soft = mask_from_energy(
                    energies,
                    S_e=S_e,
                    tau=self.tau,
                    lambda_cost=self.lambda_cost if budget_lambda is None else budget_lambda,
                    threshold=self.threshold
                )

                edge_index_list.append(edge_index)
                m_list.append(m_soft)

            Z = self.graph_transformer(H, edge_index_list, m_list)
            y_pred = self.prediction_head(Z)

        # return y_pred, edge_index_list
        return y_pred, edge_index_list, m_list