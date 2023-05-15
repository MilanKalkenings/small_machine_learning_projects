import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class DataInference(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, n_subnets: int):
        super(DataInference, self).__init__()
        self.x = torch.repeat_interleave(input=x, repeats=n_subnets, dim=1)
        y = y.reshape([len(y), 1])
        self.y = torch.repeat_interleave(input=y, repeats=n_subnets, dim=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item: int):
        return self.x[item], self.y[item]


class DataTraining(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, n_subnets: int):
        super(DataTraining, self).__init__()
        x_parts = torch.split(tensor=x, split_size_or_sections=int(len(x) / n_subnets), dim=0)
        self.x = torch.cat(x_parts, dim=1)
        self.y = y.reshape([len(self.x), n_subnets])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item: int):
        return self.x[item], self.y[item]


def cifar1_loaders(n_subnets: int, batch_size: int, image_size: int, n_train_obs: int, n_val_obs: int):
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resizer = Resize([image_size, image_size])
    trans = Compose([resizer, ToTensor(), normalizer])
    cifar10 = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=True, transform=trans)
    x = torch.tensor(cifar10.data).permute([0, 3, 1, 2])
    y = torch.tensor(cifar10.targets)

    # equally shuffle x, y
    indices = torch.randperm(50_000)
    x = x[indices]
    y = y[indices]

    x_train, x_val = x[:n_train_obs], x[n_train_obs:n_train_obs+n_val_obs]
    y_train, y_val = y[:n_train_obs], y[n_train_obs:n_train_obs+n_val_obs]

    data_train = DataTraining(x=x_train, y=y_train, n_subnets=n_subnets)
    data_inf_train = DataInference(x=x_train, y=y_train, n_subnets=n_subnets)
    data_inf_val = DataInference(x=x_val, y=y_val, n_subnets=n_subnets)
    loader_train = DataLoader(dataset=data_train, batch_size=batch_size)
    loader_inf_train = DataLoader(dataset=data_inf_train, batch_size=batch_size)
    loader_inf_val = DataLoader(dataset=data_inf_val, batch_size=batch_size)
    print("batches in loaders:")
    print("train", len(loader_train), "inference_train", len(loader_inf_train), "inference_val", len(loader_inf_val))
    return loader_train, loader_inf_train, loader_inf_val
