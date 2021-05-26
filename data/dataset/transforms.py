import numpy as np
import torchvision.transforms as transforms

def set_transforms(transforms_list, img_size):
    compose_list = []

    for t in transforms_list:
        if t == "normalize":
            compose_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        if t == "resize":
            compose_list.append(transforms.Resize(img_size))
        if t == "to_tensor":
            compose_list.append(transforms.ToTensor())
    
    return transforms.Compose(compose_list)

