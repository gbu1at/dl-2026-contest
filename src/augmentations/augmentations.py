import torch
from torchvision import transforms


def get_train_transform(cfg):
    return transforms.Compose([
                transforms.Resize(cfg["image_size"]),
                transforms.ToTensor()
    ])


def get_val_transform(cfg):
    return transforms.Compose([
                transforms.Resize(cfg["image_size"]),
                transforms.ToTensor()
    ])