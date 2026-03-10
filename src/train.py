import sys
import os

PROJECT_ROOT = "/content/drive/MyDrive/dl-2026-contest/dl-2026-contest/src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch

from utils.seed import set_seed
from utils.logger import get_logger
from utils.config import load_cfg

from datasets.dataloader import build_dataloader

from models.model import get_model

from training.trainer import Trainer
from training.optimizer import get_optimizer
from training.loss import get_criterion


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = load_cfg("configs/train_config.yaml")

    set_seed(cfg["seed"])

    train_dataloader, val_dataloader = build_dataloader(cfg)
    model = get_model(cfg["model"])
    optimizer = get_optimizer(cfg["optimizer"], model)
    criterion = get_criterion(cfg)
    logger = get_logger(cfg["logger"])

    num_epoch = cfg["num_epoch"]

    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, criterion, device, cfg, logger)

    for i in range(num_epoch):
        trainer.train_epoch(i)


if __name__ == "__main__":
    main()