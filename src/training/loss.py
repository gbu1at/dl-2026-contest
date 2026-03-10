import torch.nn as nn


def get_criterion(cfg):
    return nn.CrossEntropyLoss()