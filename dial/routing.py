"""
Routing and energy-computation utilities using closed-form energy (Route A).
"""

import torch
import torch.nn as nn
from typing import Tuple

from .utils import (
    build_incidence_matrix,
    laplacian_from_conductance,
    standardize,
    build_edge_index_from_S,
    build_task_laplacian,
)


def _safe_solve(mat: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """
    Solve a linear system with a fallback to least squares for stability.
    """
    try:
        return torch.linalg.solve(mat, rhs)
    except RuntimeError:
        return torch.linalg.lstsq(mat, rhs).solution


def compute_edge_energy(
        S: torch.Tensor,
        F: torch.Tensor,
        H: torch.Tensor,
        edge_gate: nn.Module,
        delta: float = 1e-6,
        scaling_factor: float = 100.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-edge closed-form energy E_e following Route A.

    Args:
        S: [N, N] structural connectivity.
        F: [N, N] functional/feature similarity (used as task matrix T).
        H: [N, d_model] node embeddings.
        edge_gate: Module producing edge conductances.
        delta: Ridge for Laplacian.

    Returns:
        E_e: [E] energy per edge.
        edge_index: [2, E] edge indices.
        S_e: [E] structural weights per edge.
        a_e: [E] pre-activation edge scores.
    """
    N = S.shape[0]
    device = S.device

    edge_index, S_e = build_edge_index_from_S(S)  # [2, E], [E]
    E = edge_index.shape[1]

    if E == 0:
        empty = torch.zeros(0, device=device, dtype=H.dtype)
        return empty, edge_index, S_e, empty

    F_e = F[edge_index[0], edge_index[1]]  # [E]

    a_e = edge_gate(H, edge_index, S_e, F_e)  # [E]
    g_e = torch.exp(a_e)  # conductance, positive

    Bmat = build_incidence_matrix(edge_index, N)  # [E, N]
    Lg = laplacian_from_conductance(Bmat, g_e, delta=delta)  # [N, N]
    L_T = build_task_laplacian(F)  # [N, N]

    # Matrix-vector Route A (batched over edges)
    d_mat = Bmat.transpose(0, 1)  # [N, E], each column is d_e
    x = _safe_solve(Lg, d_mat)  # [N, E]
    y = L_T @ x  # [N, E]
    z = _safe_solve(Lg, y)  # [N, E]

    edge_dot = (Bmat * z.transpose(0, 1)).sum(dim=1)  # [E], d_e^T z_e
    E_e = 2.0 * g_e * edge_dot  # [E]

    E_e = E_e * scaling_factor

    return E_e, edge_index, S_e, a_e


def mask_from_energy(
        E: torch.Tensor,
        S_e: torch.Tensor,
        tau: float,
        lambda_cost: float,
        threshold: nn.Parameter,
        eps: float = 1e-6
) -> torch.Tensor:
    """
    Build a differentiable edge mask using Log-Min-Max normalization.
    This handles long-tail energy distributions better than Z-Score.
    """
    if E.numel() == 0:
        return E

    # 1. 能量处理：Log + MinMax
    # Log 压缩动态范围，MinMax 保证在 [0, 1] 之间，无负数
    log_E = torch.log(E + eps)
    E_min, E_max = log_E.min(), log_E.max()

    if E_max > E_min:
        E_norm = (log_E - E_min) / (E_max - E_min + eps)
    else:
        # 如果所有能量都一样，给一个中间值，或者0
        E_norm = torch.zeros_like(log_E)

    # 2. 成本处理：Log + MinMax
    # 结构越强(S大) -> Cost越小。
    # S 接近 0 时，log(S) 是大负数，-log(S) 是大正数 (高成本)
    log_C = -torch.log(S_e + eps)
    C_min, C_max = log_C.min(), log_C.max()

    if C_max > C_min:
        C_norm = (log_C - C_min) / (C_max - C_min + eps)
    else:
        C_norm = torch.zeros_like(log_C)

    # 3. 计算 Logits
    # E_norm [0, 1], C_norm [0, 1]
    # 结果更加可控
    logits = tau * (E_norm - lambda_cost * C_norm - threshold)

    return torch.sigmoid(logits)
