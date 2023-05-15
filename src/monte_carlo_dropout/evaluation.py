import torch
from torch.utils.data import DataLoader
from milankalkenings.deep_learning import Module


def acc_mc_dropout(module: Module, loader_eval: DataLoader, n_forward_passes: int):
    module.eval()
    correct_count = 0
    obs_count = 0
    for batch in loader_eval:
        with torch.no_grad():
            scores_n_forward_passes = []
            x, y = batch
            for forward_pass in range(n_forward_passes):
                module_out = module(x=x, y=y)
                scores = module_out["scores"]
                scores_n_forward_passes.append(scores)
            scores_mean = torch.mean(torch.stack(scores_n_forward_passes), dim=0)
            preds = torch.argmax(scores_mean, dim=1)
            correct_count += float((preds == y).sum())
            obs_count += len(y)
    return correct_count / obs_count
