import torch

from torch.utils.data import DataLoader
from data.dataset import CIFAR10Dataset
from data.dataset import set_transforms


def set_dataloader(args, phase):
    phase = args.
    root_path = args.
    transform = set_transforms(args.)
    train_class_counts = args.
    val_class_counts = args.

    dataset = CIFAR10Dataset(phase, root_path, transform,
                             train_class_counts, val_class_counts)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            num_workers=num_workers, sampler=None)
    return dataloader
