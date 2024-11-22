from torchvision.transforms import (
    Compose,
    RandomVerticalFlip,
    RandomHorizontalFlip,
    GaussianBlur,
    RandomCrop,
    RandomErasing,
)
from typing import Type, Dict

augmentation_dict: dict[str, Type] = {
    "RandomVerticalFlip": RandomVerticalFlip,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "GaussianBlur": GaussianBlur,
    "RandomCrop": RandomCrop,
    "RandomErasing": RandomErasing,
}


def create_augmentations(augmentation_str_dict: Dict):
    augmentations_list = [
        augmentation_dict[augmentation_class](**augmentation_config)
        for augmentation_class, augmentation_config in augmentation_str_dict.items()
    ]
    return Compose(augmentations_list)
