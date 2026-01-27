"""
Loss functions for the DIAL model, covering both task loss and regularizers.
"""

import torch
import torch.nn.functional as F
from typing import Dict


def compute_losses(
        y_pred: torch.Tensor,
        y: torch.Tensor,
        task: str,
) -> Dict[str, torch.Tensor]:
    if task == 'classification':
        L_task = F.cross_entropy(y_pred, y)
    elif task == 'regression':
        L_task = F.mse_loss(y_pred, y)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    return {
        'loss': L_task,
        'task': L_task.item(),
    }
