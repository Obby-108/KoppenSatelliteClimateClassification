import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Calculate unweighted Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate pt (the probability of the correct class)
        pt = torch.exp(-ce_loss)

        # Apply the Focal factor: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Apply Class Weights (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        return loss.mean()


def soft_focal_loss(inputs, target_probs, alpha=None, gamma=2.0):
    """
    Weighted Focal Loss for soft targets (Mix-up).
    target_probs: [batch_size, num_classes] one-hot/soft labels
    alpha: [num_classes] tensor of weights
    """
    log_probs = F.log_softmax(inputs, dim=1)
    probs = F.softmax(inputs, dim=1)

    # Calculate focal weights for every class
    p_t = (target_probs * probs).sum(dim=1, keepdim=True)  # [B, 1]
    focal_weight = (1 - p_t) ** gamma  # [B, 1]

    if alpha is not None:
        loss = -(target_probs * focal_weight * log_probs * alpha).sum(dim=1)
    else:
        loss = -(target_probs * focal_weight * log_probs).sum(dim=1)

    return loss.mean()
