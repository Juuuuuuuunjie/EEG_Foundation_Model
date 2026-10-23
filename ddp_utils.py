import os
import torch
import torch.distributed as dist


def setup_ddp():
    """
    支持两种启动方式：

    1. 普通单卡/CPU:
       python train.py

    2. DDP 多卡:
       torchrun --nproc_per_node=8 train.py
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

        device = torch.device("cuda", local_rank)
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distributed = False

    return distributed, rank, world_size, local_rank, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


@torch.no_grad()
def all_reduce_mean_scalar(value, device=None):
    """
    对一个 Python float 或 0-d tensor 做跨进程平均。
    """
    if not is_dist_avail_and_initialized():
        if torch.is_tensor(value):
            return value.item()
        return float(value)

    if not torch.is_tensor(value):
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        value = torch.tensor(value, dtype=torch.float32, device=device)
    else:
        value = value.detach().float()

    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= get_world_size()
    return value.item()
