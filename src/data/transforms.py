# src/data/transforms.py

import torchvision.transforms as transforms



def build_shape_transform(train):
    if not train:
        return None

    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
    ])


def build_color_transform(train):
    if not train:
        return None

    return transforms.Compose([
    ])


def build_transforms(train):
    return build_shape_transform(train), build_color_transform(train)