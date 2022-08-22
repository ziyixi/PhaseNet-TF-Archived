import torch


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha=0.8, gamma=2) -> torch.Tensor:
    # first compute binary cross-entropy
    x = inputs.view(-1)
    y = targets.view(-1)
    # refer to https://github.com/pytorch/pytorch/blob/ba98c0e38ca76a8e03d27e20a1d2bfc324d502ca/aten/src/ATen/native/Loss.cpp#L127-L161
    # as amp binary_cross_entropy is not supported
    BCE = (y-1)*torch.max(torch.log(1-x), torch.tensor(-100.).type_as(x)) - \
        y*torch.max(torch.log(x), torch.tensor(-100.).type_as(x))
    BCE = BCE.mean()*inputs.numel()/len(inputs)  # batch mean
    # BCE = F.binary_cross_entropy(
    #     inputs.view(-1), targets.view(-1), reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

    return focal_loss
