from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class DataHandlerSetup:
    def __init__(self):
        self.root_cifar10_train = "../data/cifar10/train"
        self.root_cifar10_test = "../data/cifar10/test"
        normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # standard image normalization
        self.transforms = Compose([Resize([64, 64]), ToTensor(), normalizer])
        self.batch_size = 64
        self.few_data = False  # True for performing initial experiments with few data


class DataHandler:
    def __init__(self, setup: DataHandlerSetup):
        self.root_cifar10_train = setup.root_cifar10_train
        self.root_cifar10_test = setup.root_cifar10_test
        self.transforms = setup.transforms
        self.batch_size = setup.batch_size

        self.cifar10_train = torchvision.datasets.CIFAR10(root=self.root_cifar10_train, train=True, download=True,
                                                          transform=self.transforms)
        self.cifar10_test = torchvision.datasets.CIFAR10(root=self.root_cifar10_test, train=False, download=True,
                                                         transform=self.transforms)

        if setup.few_data:
            data_train_few, _ = random_split(dataset=self.cifar10_train, lengths=[1_000, 49_000])
            data_train, data_val = random_split(dataset=data_train_few, lengths=[900, 100])
        else:
            data_train, data_val = random_split(dataset=self.cifar10_train, lengths=[40000, 10000])
        self.loader_train = DataLoader(dataset=data_train, batch_size=self.batch_size)
        self.loader_val = DataLoader(dataset=data_val, batch_size=self.batch_size)
        self.loader_test = DataLoader(dataset=self.cifar10_test, batch_size=self.batch_size)
