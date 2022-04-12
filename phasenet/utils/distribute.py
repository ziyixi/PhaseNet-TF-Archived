import os
import torch.distributed as dist


def setup_distribute(rank: int, world_size: int, master_port: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distribute():
    dist.destroy_process_group()
