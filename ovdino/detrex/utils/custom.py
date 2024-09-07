import os

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
    dist_args = f"--nproc_per_node={num_gpus}"

    # Our distributed args are designed for MeiTuan's cluster,
    # the dist_url may be different in your environment.
    # Please refer to PyTorch's official documents for more details.
    num_nodes = os.getenv("NNODES", None)
    try:
        num_nodes = int(num_nodes)
    except:
        num_nodes = None
    if num_nodes is not None and num_nodes > 1:
        node_rank = int(os.getenv("NODE_RANK", None))
        master_addr = os.getenv("MASTER_ADDR", "localhost")
        master_port = os.getenv("MASTER_PORT", "29500")

        args.num_machines = num_nodes
        args.machine_rank = node_rank
        args.dist_url = f"tcp://{master_addr}:{master_port}"

        dist_args += f" --nnodes={num_nodes} --node_rank={node_rank} --master_addr={master_addr} --master_port={master_port}"

    # print dist args for debug.
    print(f"dist_args: {dist_args}")
    return args
