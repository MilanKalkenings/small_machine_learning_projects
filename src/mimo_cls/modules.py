import torch
from milankalkenings.deep_learning import Module
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class Mimo(Module):
    def __init__(self, n_subnets: int, n_classes: int):
        super(Mimo, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.conv1 = nn.Conv2d(in_channels=resnet.conv1.in_channels*n_subnets,
                                 out_channels=resnet.conv1.out_channels,
                                 kernel_size=resnet.conv1.kernel_size,
                                 stride=resnet.conv1.stride,
                                 padding=resnet.conv1.padding,
                                 bias=resnet.conv1.bias)
        resnet.fc = nn.Linear(in_features=resnet.fc.in_features,
                              out_features=n_classes*n_subnets,
                              bias=resnet.conv1.bias)
        self.n_subnets = n_subnets
        self.n_classes = n_classes
        self.resnet = resnet
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y):
        loss = 0
        scores_all_subnets = self.resnet(x.float())
        scores = torch.split(scores_all_subnets, split_size_or_sections=self.n_classes, dim=1)
        for subnet in range(self.n_subnets):
            loss += self.loss(scores[subnet], y[:, subnet])
        loss = loss / self.n_subnets
        return {"loss": loss, "scores": scores}

    def freeze_pretrained_layers(self):
        # todo: freeze resnet params that are not in first or last layer
        pass

    def unfreeze_pretrained_layers(self):
        # todo unfreeze resnet params that are not in first or last layer
        pass




