import builtins
import os
import random
import re
import sys

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from .dist_helper import get_rank, get_world_size, init_process_group
from .typing import List

print_debug_info = False


def _suppress_print(gpu=None):
    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    if (gpu is not None and gpu != 0) or (gpu is None and get_rank() != 0):
        builtins.print = print_pass


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _parse_slurm_node_list(s: str) -> List[str]:
    nodes = []
    # Extract "hostname", "hostname[1-2,3,4-5]," substrings
    p = re.compile(r"(([^\[]+)(?:\[([^\]]+)\])?),?")
    for m in p.finditer(s):
        prefix, suffixes = s[m.start(2) : m.end(2)], s[m.start(3) : m.end(3)]
        for suffix in suffixes.split(","):
            span = suffix.split("-")
            if len(span) == 1:
                nodes.append(prefix + suffix)
            else:
                width = len(span[0])
                start, end = int(span[0]), int(span[1]) + 1
                nodes.extend([prefix + f"{i:0{width}}" for i in range(start, end)])
    return nodes


def _get_master_port(seed: int = 0) -> int:
    MIN_MASTER_PORT, MAX_MASTER_PORT = (20_000, 60_000)

    master_port_str = os.environ.get("MASTER_PORT")
    if master_port_str is None:
        rng = random.Random(seed)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

    return int(master_port_str)


class _TorchDistributedEnvironment:
    def __init__(self):
        self.master_addr = "localhost"
        self.master_port = os.environ.get("MASTER_PORT")
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.local_world_size = 1

    def _set_from_slurm_env(self):
        # logger.info("Initialization from Slurm environment")
        job_id = int(os.environ["SLURM_JOB_ID"])

        node_count = int(os.environ["SLURM_JOB_NUM_NODES"])
        nodes = _parse_slurm_node_list(os.environ["SLURM_JOB_NODELIST"])
        assert len(nodes) == node_count
        if print_debug_info:
            print(f"jobid: {job_id}, node_count: {node_count}")

        self.master_addr = nodes[0]
        self.master_port = _get_master_port(seed=job_id)
        self.rank = int(os.environ["SLURM_PROCID"])
        self.world_size = int(
            os.environ["SLURM_NTASKS"]
        )  # SLURM_GPUS #int(os.environ["WORLD_SIZE"])
        if print_debug_info:
            print(f"master_addr: {self.master_addr}, master_port: {self.master_port}")
            print(f"rank: {self.rank}, world_size: {self.world_size}")

        assert self.rank < self.world_size
        self.local_rank = int(os.environ["SLURM_LOCALID"])

        self.local_world_size = self.world_size // node_count
        if print_debug_info:
            print(
                f"local_rank: {self.local_rank}, local_world_size: {self.local_world_size}",
                flush=True,
            )
        assert self.local_rank < self.local_world_size

    def export(self, *, overwrite: bool) -> "_TorchDistributedEnvironment":
        # See the "Environment variable initialization" section from
        # https://pytorch.org/docs/stable/distributed.html for the complete list of
        # environment variables required for the env:// initialization method.
        env_vars = {
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }
        if not overwrite:
            for k, v in env_vars.items():
                assert k in os.environ, "{} is missing in environ"
                assert (
                    v == os.environ[k]
                ), "Environment variables inconsistent {} != {}".format(
                    os.environ[k], v
                )

        os.environ.update(env_vars)
        if print_debug_info:
            print("export env finished", flush=True)
        return self


def init_distributed_mode(args):

    cudnn.benchmark = True

    torch_env = _TorchDistributedEnvironment()
    if args.interactive_session:
        torch_env.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch_env.rank = int(os.environ.get("RANK", 0))
        torch_env.world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch_env.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch_env.world_size))
    else:
        torch_env._set_from_slurm_env()
    torch_env.export(overwrite=True)

    args.gpu = torch_env.local_rank
    args.global_rank = torch_env.rank
    args.world_size = torch_env.world_size

    print(
        f"Initialize ddp. rank:{args.global_rank}, world size:{args.world_size}, local rank:{args.gpu}"
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.global_rank,
        device_id=torch.device("cuda", index=args.gpu),
    )

    torch.cuda.set_device(args.gpu)
    dist.barrier(device_ids=[args.gpu])
    setup_for_distributed(args.global_rank == 0)


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
