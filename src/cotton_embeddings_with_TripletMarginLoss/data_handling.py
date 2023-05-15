import torch
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset


class DebugData(Dataset):
    def __init__(self, batch_debug: torch.Tensor):
        """

        :param x: one batch of inputs
        :param y: one batch of labels
        """
        super(DebugData, self).__init__()
        x, y = batch_debug
        self.x = x
        self.y = y
        self.len = len(x)

    def __len__(self):
        return self.len

    def __getitem__(self, item: int):
        return self.x[item], self.y[item]


def create_loaders(batch_size: int):
    transform_train = transforms.Compose([
        transforms.Resize(size=80),
        transforms.RandomCrop(size=64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=360),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_eval = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    data = datasets.ImageFolder(root="../data/raw")
    len_train, len_val, len_test = [e.int() for e in torch.Tensor([0.8, 0.1, 0.1]) * len(data)]
    data_train, data_val, data_test = torch.utils.data.random_split(data, [len_train, len_val, len_test])
    data_train.dataset.transform = transform_train
    data_val.dataset.transform = transform_eval
    data_test.dataset.transform = transform_eval

    loader_train = DataLoader(dataset=data_train, batch_size=batch_size)
    loader_val = DataLoader(dataset=data_val, batch_size=batch_size)
    loader_test = DataLoader(dataset=data_test, batch_size=batch_size)
    return loader_train, loader_val, loader_test
