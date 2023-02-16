import torch
from torch.utils.data import DataLoader
from milankalkenings.deep_learning import Module
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne(embeddings: torch.Tensor):
    # Normalize the data
    embeddings = (embeddings - embeddings.mean(dim=0)) / embeddings.std(dim=0)
    tsne = TSNE(n_components=2)
    return tsne.fit_transform(embeddings)



def evaluate(module: Module, loader_eval: DataLoader, name_fig: str):
    module.eval()
    embeddings = []
    ys = []
    preds = []
    with torch.no_grad():
        for batch in loader_eval:
            x, y = batch
            out = module(x=x, y=y)
            scores = out["scores"]
            ys.append(y)
            embeddings.append(out["embedding"])
            preds.append(torch.argmax(scores, dim=1))
    embeddings = torch.cat(embeddings, dim=0)
    embeddings_reduced = torch.tensor(tsne(embeddings=embeddings))
    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys, dim=0)
    acc = (preds == ys).float().mean().item()
    plot_embeddings(embeddings=embeddings_reduced, ys=ys, name_fig=name_fig, acc=acc)


def plot_embeddings(embeddings: torch.Tensor, ys: torch.Tensor, name_fig: str, acc: float, n_classes: int = 6):
    separated_embeddings = [torch.empty((0, 2)) for i in range(n_classes)]
    for i in range(len(embeddings)):
        separated_embeddings[ys[i]] = torch.cat((separated_embeddings[ys[i]], embeddings[i].unsqueeze(0)))
    plt.figure()
    plt.title(f"embeddings leading to {round(acc * 100, 2)}% accuracy")
    for i in range(n_classes):
        plt.scatter(separated_embeddings[i][:, 0], separated_embeddings[i][:, 1], label=f"class {i}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../monitoring/{name_fig}.png")
