from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test_one_epoch(model: nn.Module,
                   criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                   test_loader: DataLoader,
                   device: torch.device) -> Optional[dict]:
    model.eval()
    loss_log = []
    for meta in test_loader:
        # * forward
        sgram, target = meta['sgram'].to(device), meta['label'].to(device)
        output = model(sgram)
        predict = output['predict']
        loss = criterion(predict, target)
        loss_log.append(loss.detach().item())

    res = {
        "loss": loss_log,
        "loss_mean": np.mean(loss_log),
    }
    return res
