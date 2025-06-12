# Metrics/gpu/vi_index.py
import torch

from ..base import BaseMetric


class VIIndex(BaseMetric):
    """
    Variation of Information computed on GPU.
    Returns a similarity score: 1 / (1 + VI).
    """
    type = "gpu"

    def __init__(self, num_classes: int = None, eps: float = 1e-8):
        super().__init__()
        # If num_classes not provided, will infer later
        self.num_classes = num_classes
        self.eps = eps

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Flatten predictions and masks
        pred_flat = prediction.view(-1).long()
        mask_flat = mask.view(-1).long()

        # Determine number of classes
        if self.num_classes is None:
            num_classes = int(torch.max(pred_flat.max(), mask_flat.max())) + 1
        else:
            num_classes = self.num_classes

        # Joint histogram via bincount
        joint = pred_flat * num_classes + mask_flat
        counts = torch.bincount(joint, minlength=num_classes * num_classes).float()
        counts = counts.view(num_classes, num_classes)

        # Joint and marginal probabilities
        p_ij = counts / counts.sum()
        p_i = p_ij.sum(dim=1)
        p_j = p_ij.sum(dim=0)

        # Entropies
        Hx = - (p_i * torch.log(p_i + self.eps)).sum()
        Hy = - (p_j * torch.log(p_j + self.eps)).sum()

        # Mutual Information
        MI = (p_ij * (torch.log(p_ij + self.eps)
                      - torch.log(p_i.unsqueeze(1) + self.eps)
                      - torch.log(p_j.unsqueeze(0) + self.eps))).sum()

        VI = Hx + Hy - 2 * MI
        # Convert to similarity (1 / (1 + VI)) for consistency with Metric API
        return 1.0 / (1.0 + VI + self.eps)
