import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class CarliniWagnerResNet(nn.Module):
    def __init__(self, x: torch.Tensor, kappa: float, t: int, lamb: float):
        super(CarliniWagnerResNet, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.eval()
        self.x = nn.Parameter(x)
        self.criterion = CarliniWagnerCriterion(kappa=kappa, lamb=lamb)
        self.t = t
        self.x0 = torch.clone(x)

    def forward(self):
        scores = self.resnet(self.x)
        loss = self.criterion(scores=scores, t=self.t, x=self.x, x0=self.x0)
        return loss, self.x.data, scores


class CarliniWagnerCriterion(nn.Module):
    def __init__(self, kappa: float, lamb: float):
        super(CarliniWagnerCriterion, self).__init__()
        self.kappa = torch.tensor(kappa)
        self.softmax = nn.Softmax(dim=0)
        self.lamb = lamb

    def forward(self, scores: torch.Tensor, t: int, x: torch.Tensor, x0: torch.Tensor):
        probs = self.softmax(scores[0])
        prob_t = probs[t]
        prob_j = torch.max(torch.cat([probs[:t], probs[t + 1:]]))
        diff = torch.reshape(prob_j-prob_t, [1])
        neg_kappa = torch.reshape(-self.kappa, [1])
        norm = torch.norm(x-x0)
        diff = torch.max(torch.cat([diff, neg_kappa]))
        return self.lamb * diff + (1-self.lamb) * norm
