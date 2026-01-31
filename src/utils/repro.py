from __future__ import annotations

import os 
import random
from dataclasses import dataclass 

import numpy as np 
import torch 

@dataclass(frozen=True)
class ReproConfig:
    seed: int 
    deterministic: bool

def seed_everything(cfg: ReproConfig) -> None:
    # Python
    random.seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    # numpy
    np.random.seed(cfg.seed)

    # torch
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if cfg.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
