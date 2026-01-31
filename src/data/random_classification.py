from __future__ import annotations

from dataclasses import dataclass 

import torch
from torch.utils.data import Dataset 

@dataclass(frozen=True)
class RandomDataConfig:
    num_samples: int = 2048
    num_classes: int = 10
    image_size: int = 32

class RandomImageClassificationDataset(Dataset):
    # random images and random labels

    def __init__(self, cfg: RandomDataConfig) -> None:
        self.cfg = cfg

    def __len__(self) -> int:
        return self.cfg.num_samples 

    def __getitem__(self, idx: int):
        # torch RNG with global seed 
        x = torch.rand(3, self.cfg.image_size, self.cfg.image_size, dtype=torch.float32)
        y = torch.randint(low=0, high=self.cfg.num_classes, size=(1,), dtype=torch.long).item()
        return x,y 
