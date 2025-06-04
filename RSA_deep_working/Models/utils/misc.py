import random
import numpy as np
import torch
import os

def set_seed(seed=42, deterministic=True):
    """
    Set random seed for python, numpy, torch and CUDA.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id, base_seed=42):
    """
    Worker initialization function for DataLoader to ensure reproducibility.
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(preferred="cuda"):
    """
    Returns the torch device, "cuda" if available.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
