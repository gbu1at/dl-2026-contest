from .augmentations import get_train_transform, get_val_transform
from .dataset import CarDataset
from torch.utils.data import DataLoader


def build_dataloader(cfg):
    cfg_dataloader = cfg["dataloader"]
    cfg_augmentations = cfg["augmentations"]

    train_transform = get_train_transform(cfg_augmentations["train_transform"])
    val_transform = get_train_transform(cfg_augmentations["val_transform"])



    dataset_train = CarDataset(cfg_dataloader["dataset"]["train"], train_transform)
    dataset_val = CarDataset(cfg_dataloader["dataset"]["val"], val_transform)


    dataloader_train = DataLoader(
                                  dataset=dataset_train, 
                                  batch_size=cfg_dataloader["batch_size"],
                                  shuffle=True,
                                  num_workers=cfg_dataloader["num_workers"]
                                 )
    dataloader_val = DataLoader(
                                dataset=dataset_val, 
                                batch_size=cfg_dataloader["batch_size"],
                                shuffle=False,
                                num_workers=cfg_dataloader["num_workers"]
                                )


    return dataloader_train, dataloader_val
