import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class WaterBirdDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        Args:
                root (str): Path to the root directory containing the dataset folders.
                split (str, optional): Dataset split to use ('train', 'val', 'test'). Defaults to 'train'.
                transform (callable, optional): Transformations to apply to the images. Defaults to None.
        """

        self.root = root
        self.split = split
        self.transform = transform
        self.metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))

        split_dict = {'train': 0, 'val': 1, 'test': 2}
        self.metadata = self.metadata[self.metadata['split'] == split_dict[split]]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # img path
        image_path = os.path.join(self.root, row["img_filename"])

        # load image
        image = Image.open(image_path).convert('RGB')

        # transformations
        if self.transform is not None:
            image = self.transform(image)

        label = row["y"]
        place = row["place"]
        return {
            "image": image,
            "label": label,
            "place": place,
        }


class WaterBirds:
    def __init__(self, root, seed=None):
        """
        Constructor for wrapper class.
        Args:
            root (str): Path to the root directory containing the dataset folders.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """

        self.root = root
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)

    def get_train_set(self):
        """
        Returns:
        torch.utils.data.dataset.Subset: train set
        """
        return WaterBirdDataset(
            root=self.root,
            split='train',
            transform=self.train_transformation
        )

    def get_val_set(self):
        """
        Returns:
        torch.utils.data.dataset.Subset: val set
        """
        return WaterBirdDataset(
            root=self.root,
            split='val',
            transform=self.val_transformation
        )

    def get_test_set(self):
        """
        Returns:
        torch.utils.data.dataset.Subset: train set
        """
        return WaterBirdDataset(
            root=self.root,
            split='test',
            transform=self.val_transformation #same transformation as for val
        )

    @staticmethod
    def train_transformation():
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def val_transformation():
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])