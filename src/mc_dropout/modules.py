import torch
import torch.nn as nn
from milankalkenings.deep_learning import Module


class MCDropout(nn.Module):
    def __init__(self, p: float):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor):
        mask = (torch.rand(size=x.shape) > self.p).int()
        return x * mask


class CNNBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 mc_dropout: bool,
                 kernel_size: int = 3,
                 padding: int = 1,
                 dropout_p: float = 0.2,
                 kernel_size_pool: int = 2):
        super(CNNBlock, self).__init__()
        self.rect = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        if mc_dropout:
            self.dropout = MCDropout(p=dropout_p)
        else:
            self.dropout = nn.Dropout(p=dropout_p)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pool)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.pool(self.rect(self.conv(x))))


class CNN(Module):
    def __init__(self, depth: int, example_x: torch.Tensor, n_classes: int, mc_dropout: bool, in_channels: int = 1):
        super(CNN, self).__init__()
        sequential = nn.Sequential()
        sequential.append(CNNBlock(in_channels=in_channels,
                                   out_channels=4,
                                   mc_dropout=mc_dropout,
                                   kernel_size=5,
                                   padding=2,
                                   dropout_p=0.8))
        last_out_channels = 4
        for i in range(depth - 1):
            sequential.append(CNNBlock(in_channels=last_out_channels,
                                       out_channels=last_out_channels * 2,
                                       mc_dropout=mc_dropout))
            last_out_channels = last_out_channels * 2
        sequential.append(nn.Flatten())
        self.sequential = sequential
        self.head_cls = nn.Linear(in_features=self.head_width(example_x=example_x), out_features=n_classes)
        self.loss_cls = nn.CrossEntropyLoss()

    def head_width(self, example_x: torch.Tensor):
        return self.sequential(example_x).shape[1]

    def forward(self, x, y):
        scores = self.head_cls(self.sequential(x))
        loss = self.loss_cls(scores, y)
        return {"loss": loss, "scores": scores}

    def freeze_pretrained_layers(self):
        # no pretrained modules used
        pass

    def unfreeze_pretrained_layers(self):
        # no pretrained modules used
        pass
