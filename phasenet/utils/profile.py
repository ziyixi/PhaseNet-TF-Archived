from typing import Callable, Optional

import torch
import torch.nn as nn
from phasenet.conf.load_conf import ProfileConfig
from torch.utils.data import DataLoader


def main_profile(pcfg: ProfileConfig,
                 model: nn.Module,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizer: torch.optim.Optimizer,
                 data_loader: DataLoader,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
                 device: torch.device) -> Optional[dict]:
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=pcfg.wait, warmup=pcfg.warmup, active=pcfg.active, repeat=pcfg.repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("profile"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step, meta in enumerate(data_loader):
            if step >= (pcfg.warmup+pcfg.active)*pcfg.repeat:
                break
            # * forward
            model.train()
            sgram, target = meta['sgram'].to(device), meta['label'].to(device)
            output = model(sgram)
            predict = output['predict']
            loss = criterion(predict, target)
            # * backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            prof.step()
