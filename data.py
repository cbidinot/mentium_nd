from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms


def get_dataloaders(
    task: str = "mnist",
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    image_size: int = 32,
    data_root: str | Path = "./data",
    download: bool = True,
    num_workers: int = 2,
    normalization: str = "cifar",
):
    data_root = Path(data_root).expanduser().resolve()

    if task == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(data_root, train=True, download=download, transform=transform)
        test_dataset = datasets.MNIST(data_root, train=False, download=download, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        return train_loader, test_loader

    if task == "cifar10":
        if normalization == "imagenet":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

        transform_test = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_train = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train_dataset = datasets.CIFAR10(data_root, train=True, download=download, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_root, train=False, download=download, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers
        )
        return train_loader, test_loader

    if task == "cifar100":
        if normalization == "imagenet":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

        transform_test = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_train = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train_dataset = datasets.CIFAR100(data_root, train=True, download=download, transform=transform_train)
        test_dataset = datasets.CIFAR100(data_root, train=False, download=download, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False, pin_memory=False, num_workers=num_workers
        )
        return train_loader, test_loader

    raise ValueError(f"Unsupported task: {task}. Choose from 'mnist', 'cifar10', 'cifar100'.")