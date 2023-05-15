import torch
from torch.utils.data import DataLoader
from modules import Mimo


def accuracy_inf(mimo: Mimo, loader_inf_eval: DataLoader):
    mimo.eval()
    correct_count = 0
    obs_count = 0
    for batch in loader_inf_eval:
        x, y = batch
        with torch.no_grad():
            scores = mimo(x=x, y=y)["scores"]
            scores_mean = torch.mean(input=torch.stack(tensors=scores), dim=0)
            preds = torch.argmax(scores_mean, dim=1)
            ground_truth = y[:, 0]
            correct_count += int((preds == ground_truth).sum())
            obs_count += len(ground_truth)
    return correct_count / obs_count
