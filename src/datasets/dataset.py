from torch.utils.data import Dataset
from torchvision.io import decode_image

import pandas as pd
import os



class CarDataset(Dataset):
    def __init__(self, cfg, transform=None) -> None:
        super().__init__()

        annotation_file = cfg["annotation_file"]
        img_dir = cfg["img_dir"]

        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


