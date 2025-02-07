
from typing import Tuple
import numpy as np
import torch

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    # Mixup Augmentation (applied only to inputs; speaker labels are kept intact for ArcFace)
    lam: float = np.random.beta(alpha, alpha)
    batch_size: int = x.size(0)
    index: torch.Tensor = torch.randperm(batch_size)
    mixed_x: torch.Tensor = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y  # Note: y remains unchanged


def label_smoothing(y: torch.Tensor, classes: int, smoothing: float = 0.1) -> torch.Tensor:
    # Label Smoothing (not used with ArcFace, kept here for completeness)
    y_smoothed: torch.Tensor = torch.full((y.size(0), classes), smoothing / (classes - 1))
    y_smoothed.scatter_(1, y.unsqueeze(1), 1.0 - smoothing)
    return y_smoothed
