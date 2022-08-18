from typing import List

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


class MetricBase(Metric):
    def __init__(self, threshold: int):
        super().__init__()
        self.threshold = threshold
        self.add_state("tp", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, predict: torch.Tensor, label: torch.Tensor):
        # predict,label has the shape bt, ch, nt
        # the first ch is the noise, will ignore in calculation
        pred = (predict[:, 1:, :] > self.threshold).int()
        targ = (label[:, 1:, :] > self.threshold).int()

        t = (pred == targ)
        f = (pred != targ)
        p = (pred == 1)
        n = (pred == 0)

        self.tp += torch.sum(torch.logical_and(t, p))
        self.fp += torch.sum(torch.logical_and(f, p))
        self.fn += torch.sum(torch.logical_and(f, n))
        self.tn += torch.sum(torch.logical_and(t, n))


class Accuracy(MetricBase):
    def __init__(self, threshold: float):
        super().__init__(threshold)

    def compute(self):
        return (self.tp.float()+self.tn.float())/(self.tp+self.tn+self.fp+self.fn)


class Precision(MetricBase):
    def __init__(self, threshold: float):
        super().__init__(threshold)

    def compute(self):
        if self.tp+self.fp == 0:
            return torch.tensor(0.0)  # tp-fp curve, here recall will be 0
        return (self.tp.float())/(self.tp+self.fp)


class Recall(MetricBase):
    def __init__(self, threshold: float):
        super().__init__(threshold)

    def compute(self):
        if self.tp+self.fn == 0:
            return torch.tensor(0.0)
        return (self.tp.float())/(self.tp+self.fn)


class F1(MetricBase):
    def __init__(self, threshold: float):
        super().__init__(threshold)

    def compute(self):
        precision = (self.tp.float())/(self.tp+self.fp) if self.tp + \
            self.fp != 0 else torch.tensor(1.0)
        recall = (self.tp.float())/(self.tp+self.fn) if self.tp + \
            self.fn != 0 else torch.tensor(1.0)
        return 2*precision*recall/(precision+recall)
