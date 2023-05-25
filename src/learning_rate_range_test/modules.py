from abc import ABC, abstractmethod
import torch
from torchvision.models import ResNet18_Weights, resnet18
import torch.nn as nn


class Module(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def freeze_pretrained_layers(self):
        pass

    @abstractmethod
    def unfreeze_pretrained_layers(self):
        pass


class ClsHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_p: float):
        super().__init__()
        self.rect = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.bn = nn.BatchNorm1d(num_features=in_features)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        return self.linear(self.bn(self.dropout(self.rect(x))))


class CNN(Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.head = ClsHead(in_features=1000, out_features=10, dropout_p=0.2)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        scores = self.head(self.resnet(x))
        loss = self.loss(scores, y)
        return {"scores": scores, "loss": loss}

    def freeze_pretrained_layers(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_pretrained_layers(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
