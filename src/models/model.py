import torch
import torch.nn as nn
import timm

def get_model(cfg):

    model_name = cfg["model_name"]
    num_classes = cfg["num_classes"]
    pretrained = cfg["pretrained"]
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    return model
