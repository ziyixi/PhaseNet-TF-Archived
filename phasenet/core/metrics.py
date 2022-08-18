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
            return torch.tensor(1.0)  # tp-fp curve, here recall will be 0
        return (self.tp.float())/(self.tp+self.fp)


class Recall(MetricBase):
    def __init__(self, threshold: float):
        super().__init__(threshold)

    def compute(self):
        if self.tp+self.fn == 0:
            return torch.tensor(1.0)
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


class TPR(MetricBase):
    def __init__(self, threshold: float):
        super().__init__(threshold)

    def compute(self):
        return (self.tp.float())/(self.tp+self.fn)


class FPR(MetricBase):
    def __init__(self, threshold: float):
        super().__init__(threshold)

    def compute(self):
        return (self.fp.float())/(self.fp+self.tn)


class AUC(Metric):
    def __init__(self, dt: float = 0.05):
        # dt: threshold step
        super().__init__()
        self.dt = dt
        self.thresholds = torch.arange(0, 1+dt, dt)
        self.thresholds[0] -= 0.00001
        self.thresholds[-1] += 0.00001

        self.add_state("tp", default=torch.zeros(len(self.thresholds)),
                       dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(len(self.thresholds)),
                       dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(len(self.thresholds)),
                       dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(len(self.thresholds)),
                       dist_reduce_fx="sum")

    def update(self, predict: torch.Tensor, label: torch.Tensor):
        for idx, threshold in enumerate(self.thresholds):
            pred = (predict[:, 1:, :] > threshold).int()
            targ = (label[:, 1:, :] > threshold).int()

            t = (pred == targ)
            f = (pred != targ)
            p = (pred == 1)
            n = (pred == 0)

            self.tp[idx] += torch.sum(torch.logical_and(t, p))
            self.fp[idx] += torch.sum(torch.logical_and(f, p))
            self.fn[idx] += torch.sum(torch.logical_and(f, n))
            self.tn[idx] += torch.sum(torch.logical_and(t, n))

    def compute(self):
        tpr = []
        fpr = []
        for idx, _ in enumerate(self.thresholds):
            tpr.append((self.tp[idx].float())/(self.tp[idx]+self.fn[idx])
                       if self.tp[idx]+self.fn[idx] != 0 else torch.tensor(0.0))
            fpr.append((self.fp[idx].float())/(self.fp[idx]+self.tn[idx])
                       if self.fp[idx]+self.tn[idx] != 0 else torch.tensor(0.0))
        tpr = [each.item() for each in tpr]
        fpr = [each.item() for each in fpr]
        return torch.trapezoid(torch.tensor(tpr[::-1]), torch.tensor(fpr[::-1]))
