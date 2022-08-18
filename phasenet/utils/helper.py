import subprocess

import torch
from phasenet.conf import Config
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.tensorboard import SummaryWriter


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
