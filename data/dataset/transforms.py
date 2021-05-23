import numpy as np
import torch

def set_transforms(transforms_list):
    compose_list = []

    for t in transform_list:
        if t == "normalize":
            compose_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        if t == "resize":
            compose_list.append(transforms.Resize(img_size)
        if t == "to_tensor":
            compose_list.append(transforms.ToTensor()))
    
    return transforms.Compose(comose_list)

