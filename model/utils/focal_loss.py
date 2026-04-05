import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

def soft_focal_loss(inputs, target_probs, gamma=2.0):
    """Focal Loss for soft targets (Mix-up)"""
    log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
    probs = torch.exp(log_probs)

    # Standard Focal Loss weighting
    focal_weight = (1 - probs) ** gamma
    loss = -(target_probs * focal_weight * log_probs).sum(dim=1)
    return loss.mean()
