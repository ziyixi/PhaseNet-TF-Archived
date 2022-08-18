import torch
import torch.nn as nn
from torchmetrics import Metric


class KlDiv(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, predict: torch.Tensor, label: torch.Tensor):
        self.loss += nn.functional.kl_div(
            torch.nn.functional.log_softmax(predict, dim=1), label, reduction='batchmean',
        )
        self.total += 1

    def compute(self):
        return self.loss / self.total
