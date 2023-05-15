import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def create_loaders_mnist(batch_size: int):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_val_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    # Split the train set into train and validation sets
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

    loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("batch size:", batch_size)
    print("train batches:", len(loader_train), "val batches:", len(loader_val), "test batches:", len(loader_test))
    return loader_train, loader_val, loader_test
