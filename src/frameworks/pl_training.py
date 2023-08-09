from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd


def _replace_module(network: nn.Module, to_replace: type, replace_with: type):
    """
    recursively checks if submodule of <network> is of class <to_replace>
    and in case replaces it with instance of class <replace_with>.
    inplace operation.
    """
    for module_name, module in network.named_children():
        if isinstance(module, to_replace):
            setattr(network, module_name, replace_with())
        else:
            _replace_module(module, to_replace, replace_with)  # recursive call


class ExUBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        # xavier init
        upper = math.sqrt(6 / (2 * in_features))
        lower = - upper
        weight_data = (lower - upper) * torch.rand(size=[in_features]) + upper
        self.weight = nn.Parameter(data=weight_data)
        self.bias = nn.Parameter(data=torch.zeros(size=[in_features]))

    def forward(self, x: torch.Tensor):
        """
        returns relu_n(exp(w)(x-b))
        relu_n(x) = max(1, min(0, x))
        """
        exu = torch.exp(self.weight) * (x - self.bias)
        return torch.clamp(input=exu, min=0, max=1)


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.activation(self.conv(x))


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_features=10, out_features=5), ExUBlock(in_features=5))
        self.head = nn.Sequential(nn.Linear(in_features=5, out_features=1), ExUBlock(in_features=1))
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        logits = self.head(self.body(x))
        return self.ce(input=logits, target=y), logits


class Data(Dataset):
    def __init__(self):
        super().__init__()
        self.features = torch.ones(size=[128, 3, 32, 32])
        self.targets = torch.randint(size=[128], low=0, high=1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return item

    def collate_fn(self, batch_ids: List[int]):
        """
        speeds up data augmentation by performing it per batch.

        :param batch_ids: ids of instances used in current batch
        :return:
        """
        features_batch = self.features[min(batch_ids):max(batch_ids)]
        targets_batch = self.targets[min(batch_ids):max(batch_ids)]
        return features_batch, targets_batch