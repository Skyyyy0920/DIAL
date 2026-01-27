"""
Utility helpers for building graph structures, masks, and normalization.
"""

import torch
from typing import Tuple, Optional


def build_edge_index_from_S(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge indices and weights from a structural connectivity matrix.

    Args:
        S: [N, N] symmetric adjacency matrix with non-negative entries.

    Returns:
        edge_index: [2, E] edge indices (upper triangle only).
        edge_weight: [E] edge weights.
    """
    triu_mask = torch.triu(torch.ones_like(S, dtype=torch.bool), diagonal=1)
    edges = (S > 0) & triu_mask

    row, col = torch.where(edges)
    edge_index = torch.stack([row, col], dim=0)  # [2, E]

    edge_weight = S[row, col]  # [E]

    return edge_index, edge_weight


def build_incidence_matrix(edge_index: torch.Tensor, N: int) -> torch.Tensor:
    """
    Build the edge-node incidence matrix for an undirected graph.

    Args:
        edge_index: [2, E] edge indices.
        N: Number of nodes.

    Returns:
        Bmat: [E, N] incidence matrix with +1 at the source and -1 at the target.
    """
    E = edge_index.shape[1]
    device = edge_index.device

    Bmat = torch.zeros(E, N, device=device, dtype=torch.float32)
    edge_idx = torch.arange(E, device=device)
    Bmat[edge_idx, edge_index[0]] = 1.0
    Bmat[edge_idx, edge_index[1]] = -1.0

    return Bmat


def laplacian_from_conductance(
        Bmat: torch.Tensor,
        g_e: torch.Tensor,
        delta: float = 1e-6
) -> torch.Tensor:
    """
    Build the graph Laplacian from edge conductances with ridge regularization.

    Args:
        Bmat: [E, N] incidence matrix.
        g_e: [E] edge conductances.
        delta: Ridge coefficient added to the diagonal.

    Returns:
        Lg: [N, N] Laplacian matrix (regularized to stay invertible).
    """
    G_diag = torch.diag(g_e)
    Lg = Bmat.t() @ G_diag @ Bmat

    N = Lg.shape[0]
    Lg = Lg + delta * torch.eye(N, device=Lg.device, dtype=Lg.dtype)

    return Lg


def standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Z-score normalize a tensor.

    Args:
        x: Input tensor.
        eps: Numerical stability constant.

    Returns:
        Standardized tensor.
    """
    if x.numel() == 0:
        return x

    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + eps)


def build_task_laplacian(T: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """
    Construct a Laplacian from a task-related affinity matrix.

    Args:
        T: [N, N] affinity/similarity matrix (not necessarily symmetric).
        ridge: Small diagonal term for numerical stability.

    Returns:
        L_T: [N, N] task Laplacian.
    """
    # Symmetrize and ensure non-negative weights
    T_sym = 0.5 * (T + T.transpose(0, 1))
    T_sym = torch.abs(T_sym)
    T_sym = T_sym.clone()
    T_sym.fill_diagonal_(0.0)

    degree = T_sym.sum(dim=1)
    L_T = torch.diag(degree) - T_sym

    N = T.shape[0]
    L_T = L_T + ridge * torch.eye(N, device=T.device, dtype=T.dtype)
    return L_T


def create_attention_mask_from_adjacency(
        S: torch.Tensor,
        add_self_loops: bool = True
) -> torch.Tensor:
    """
    Create an attention mask from an adjacency matrix.

    Args:
        S: [N, N] adjacency matrix.
        add_self_loops: Whether to include self-loops.

    Returns:
        mask: [N, N] attention mask (-inf for non-adjacent pairs).
    """
    N = S.shape[0]
    device = S.device

    adj_mask = (S > 0).float()

    if add_self_loops:
        adj_mask = adj_mask + torch.eye(N, device=device, dtype=adj_mask.dtype)

    attention_mask = torch.where(
        adj_mask > 0,
        torch.zeros_like(adj_mask),
        torch.full_like(adj_mask, float('-inf'))
    )

    return attention_mask
