import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class ISICSubset(Dataset):
    """
    Generate a subset of the dataset given the indices of the samples
    """

    def __init__(self, dataset, indices):
        """
        Constructor of the class

        Args:
            dataset (torch.utils.data.Dataset): a dataset
            indices (list): indices of the samples from the original dataset
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class ISICDataset(Dataset):
    def __init__(self, root, metadata_path, split='train', transform=None):
        """
        Args:
                root (str): Path to the root directory containing the dataset folders.
                split (str, optional): Dataset split to use ('train', 'val', 'test'). Defaults to 'train'.
                transform (callable, optional): Transformations to apply to the images. Defaults to None.
        """

        self.root = root
        self.split = split
        self.transform = transform
        self.metadata = pd.read_csv(metadata_path)

        split_dict = {'train': 0, 'val': 1, 'test': 2}
        self.metadata = self.metadata[self.metadata['split'] == split_dict[split]]

        self.y_array = self.metadata['benign_malignant'].values
        self.p_array = self.metadata['patches'].values
        self.n_y = np.unique(self.y_array).size
        self.n_p = np.unique(self.p_array).size
        self.group_array = (self.y_array * self.n_p + self.p_array).astype('int')
        self.n_groups = self.n_y*self.n_p
        print(np.unique(self.group_array))
        print(
            f"Number of unique labels: {self.n_y}, Number of unique places: {self.n_p}, Total groups: {self.n_groups}")
        for group in np.unique(self.group_array):
            count = len(np.where(self.group_array == group)[0])
            print(f"group {group}: {count}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # img path
        image_path = os.path.join(self.root, row["isic_id"])

        # load image
        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Error loading image at {image_path}: {e}")

        # transformations
        if self.transform is not None:
            image = self.transform(image)

        label = row["benign_malignant"]
        place = row["patches"]
        group = label*self.n_p+place
        return image, label, group



class ISIC:
    def __init__(self, root, metadata_path, seed=None, num_workers=8):
        """
        Constructor for wrapper class.
        Args:
            root (str): Path to the root directory containing the dataset folders.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            num_workers (int, optional): Number of wokrers
        """

        self.root = root
        self.seed = seed
        self.metadata_path = metadata_path
        self.num_workers = num_workers

        if seed is not None:
            torch.manual_seed(seed)

    def get_train_set(self):
        """
        Returns:
        torch.utils.data.dataset.Subset: train set
        """
        return ISICDataset(
            root=self.root,
            split='train',
            metadata_path=self.metadata_path,
            transform=self.train_transformation()
        )

    def get_valid_set(self):
        """
        Returns:
        torch.utils.data.dataset.Subset: val set
        """
        return ISICDataset(
            root=self.root,
            split='val',
            metadata_path=self.metadata_path,
            transform=self.val_transformation()
        )

    def get_test_set(self):
        """
        Returns:
        torch.utils.data.dataset.Subset: test set
        """
        return ISICDataset(
            root=self.root,
            split='test',
            metadata_path=self.metadata_path,
            transform=self.val_transformation()  # same transformation as for val
        )

    @staticmethod
    def train_transformation():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def val_transformation():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


