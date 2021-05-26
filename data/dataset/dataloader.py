import torch

from torch.utils.data import DataLoader
from data.dataset.dataset_cifar10 import CIFAR10Dataset
from data.dataset.transforms import set_transforms


def set_dataloader(args, phase):
    batch_size = args.TRAIN.BATCH_SIZE
    num_workers = args.DATA.NUM_WORKERS
    root_path = args.DATA.ROOT_PATH
    transform = set_transforms(args.DATA.TRANSFORM_LIST, args.DATA.IMG_SIZE)
    train_class_counts = args.DATA.TRAIN_CLASS_COUNTS
    val_class_counts = args.DATA.VAL_CLASS_COUNTS

    dataset = CIFAR10Dataset(phase, root_path, transform,
                            train_class_counts, val_class_counts).dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            num_workers=num_workers, sampler=None, shuffle=True)
    return dataloader
