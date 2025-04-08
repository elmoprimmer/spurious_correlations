import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch

from sklearn.model_selection import train_test_split


class ISICDataset(Dataset):
    def __init__(self, basedir, csv_file, transform=None, split="train", test_size=0.2, val_size=0.1, seed=42,
                 pre_split=True):
        self.basedir = basedir
        self.transform = transform
        self.metadata = pd.read_csv(csv_file)

        split_mapping = {"train": 0, "val": 1, "test": 2}
        split_value = split_mapping.get(split)

        self.metadata['combined_group'] = self.metadata['benign_malignant'] * 2 + self.metadata['patches']

        if pre_split:
            self.metadata = self.metadata[self.metadata['split'] == split_value]
        else:
            train_data, test_data = train_test_split(
                self.metadata,
                test_size=test_size,
                random_state=seed,
                stratify=self.metadata['combined_group']
            )
            train_data, val_data = train_test_split(
                train_data,
                test_size=val_size / (1 - test_size),
                random_state=seed,
                stratify=train_data['combined_group']
            )

            if split == "train":
                self.metadata = train_data
            elif split == "test":
                self.metadata = test_data
            elif split == "val":
                self.metadata = val_data
            else:
                raise ValueError(f"Invalid split {split}")

        self.y_array = self.metadata['benign_malignant'].values
        self.p_array = self.metadata['patches'].values
        self.filename_array = self.metadata['isic_id'].values

        self.n_classes = np.unique(self.y_array).size
        self.n_secondary_classes = np.unique(self.p_array).size
        self.n_places = self.n_secondary_classes

        self.group_array = (self.y_array * self.n_secondary_classes + self.p_array).astype('int')
        self.n_groups = self.n_classes * self.n_secondary_classes

        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_secondary_classes).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        print(self.group_counts)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        p = self.p_array[idx]
        g = self.group_array[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, y, g, p
