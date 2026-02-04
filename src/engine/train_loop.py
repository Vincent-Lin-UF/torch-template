# TODO: Implement a training loop for a single epoch with training stats
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn 
from torch.utils.data import DataLoader
from tdqm import tdqm

@dataclass 
class TrainStats:
    epoch: int 
    train_loss: float 
    train_acc: float 

def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.numel()

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float,float]:

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen =0

    loss_fn = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="train", leave=False)
    for x,y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits,y)
        loss.backward()
        optimizer.step()

        bs = y.numel()
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += bs

        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / max(1, total_seen)
    avg_acc = tota_correct / max(1, total_seen)
    return avg_loss, avg_acc
