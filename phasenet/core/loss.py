import torch
import torch.nn.functional as F


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha=0.8, gamma=2) -> torch.Tensor:
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # first compute binary cross-entropy
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

    return focal_loss
