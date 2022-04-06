from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def test_one_epoch(model: nn.Module,
                   criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                   test_loader: DataLoader,
                   device: torch.device,
                   enable_log: bool = False) -> Optional[dict]:
    model.eval()
    loss_log = []
    if enable_log:
        predict_log = []
    with torch.inference_mode():
        for meta in test_loader:
            # * forward
            sgram, target = meta['sgram'].to(device), meta['label'].to(device)
            output = model(sgram)
            predict = output['predict']
            loss = criterion(predict, target)
            loss_log.append(loss.detach().item())
            if enable_log:
                predict_log.append(
                    torch.nn.functional.softmax(predict.detach(), dim=1))

    res = {
        "loss": loss_log,
        "loss_mean": np.mean(loss_log),
    }
    if enable_log:
        res['predict'] = predict_log
    return res
