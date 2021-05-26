import numpy as np
import torch

from torchvision.datasets import CIFAR10

class CIFAR10Dataset:
    def __init__(self, phase, root_path, transform, train_class_counts, val_class_counts):
        self.prepare(phase, root_path, transform, train_class_counts, val_class_counts)

    def prepare(self, phase, root_path, transform, train_class_counts, val_class_counts):
        train_dataset = CIFAR10(download=True, root=root_path, train=True, transform=transform)
        val_dataset = CIFAR10(download=True, root=root_path, train=True, transform=transform)
        test_dataset = CIFAR10(download=True, root=root_path, train=False, transform=transform)

        train_class_indices, val_class_indices = self.set_num_data(train_dataset, train_class_counts, val_class_counts)
        targets = np.array(train_dataset.targets)
        train_dataset.targets = targets[train_class_indices]
        train_dataset.data = train_dataset.data[train_class_indices]
        val_dataset.targets = targets[val_class_indices]
        val_dataset.data = val_dataset.data[val_class_indices]
        train_num = len(train_dataset)
        val_num = len(val_dataset)
        # to confirm train and val datasets
        # print("Training data is {}, {} in toatal".format(train_class_counts, train_num))
        # print("val data is {}, {} in toatal".format(val_class_counts, val_num))
        if phase == "train":
            self.dataset = train_dataset
        elif phase == "val":
            self.dataset = val_dataset
        else:
            self.dataset = test_dataset
            
            
    def set_num_data(self, train_dataset, train_class_counts, val_class_counts):
        targets = np.array(train_dataset.targets)
        classes, class_counts = np.unique(targets, return_counts=True)
        num_classes = len(classes)
        class_indices = [np.where(targets == i)[0] for i in range(num_classes)]

        # sampling_class_counts: class counts for train and val
        sampling_class_counts = [train + val for (train, val) in zip(train_class_counts, val_class_counts)]
        sampling_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, sampling_class_counts)]

        train_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(sampling_class_indices, train_class_counts)]
        val_class_indices = [class_idx[class_count:] for class_idx, class_count in zip(sampling_class_indices, train_class_counts)]
        train_class_indices = np.hstack(train_class_indices)
        val_class_indices = np.hstack(val_class_indices)
        return train_class_indices, val_class_indices
