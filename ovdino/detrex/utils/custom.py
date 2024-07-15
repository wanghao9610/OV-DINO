import torch
import torch.distributed as dist


def rank0_print(*msg):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*msg)
    else:
        print(*msg)


def setup_dist_args(args):
    assert torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    args.num_gpus = num_gpus
    return args
