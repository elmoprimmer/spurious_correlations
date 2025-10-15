import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class WaterBirdSubset(Dataset):
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

        self.y_array = self.metadata['y'].values
        self.p_array = self.metadata['place'].values
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
        image_path = os.path.join(self.root, row["img_filename"])

        # load image
        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"Error loading image at {image_path}: {e}")

        # transformations
        if self.transform is not None:
            image = self.transform(image)

        label = row["y"]
        place = row["place"]
        group = label*self.n_p+place
        return image, label, group



class WaterBirds:
    def __init__(self, root, seed=None, num_workers=8):
        """
        Constructor for wrapper class.
        Args:
            root (str): Path to the root directory containing the dataset folders.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            num_workers (int, optional): Number of wokrers
        """

        self.root = root
        self.seed = seed
        self.num_workers = num_workers

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
            transform=self.train_transformation()
        )

    def get_valid_set(self):
        """
        Returns:
        torch.utils.data.dataset.Subset: val set
        """
        return WaterBirdDataset(
            root=self.root,
            split='val',
            transform=self.val_transformation()
        )

    def get_test_set(self):
        """
        Returns:
        torch.utils.data.dataset.Subset: train set
        """
        return WaterBirdDataset(
            root=self.root,
            split='test',
            transform=self.val_transformation()  # same transformation as for val
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


def get_sample_indices_for_group(
        dataset, num_samples_per_group=5, device="cpu", groups=None
):
    """
    Get indices of samples based on the chosen classes

    Args:
        dataset (torch.utils.data.Dataset): dataset
        num_samples_per_group (int, optional): The number of samples
                            to choose from each class. Defaults to 5. "all" for all.
        device (str, optional): device. Defaults to "cpu".

    Returns:
        list: list of indices from the given dataset based on the chosen classes
    """
    if groups is None:
        total = len(dataset)
        if num_samples_per_group == "all":
            indices = torch.randperm(total, device=device)
        else:
            n = min(num_samples_per_group, total)
            indices = torch.randperm(total, device=device)[:num_samples_per_group]
        indices = indices.long().cpu().tolist()
        return indices
            
    elif isinstance(groups, int):
        target_groups = [groups]
    else:
        target_groups = list(groups)
    indices = torch.Tensor([]).to(device)
    print("target groups:", target_groups)
    for group in target_groups:

        # Loop thru groups and
        # Get all indices for the specific group
        class_indices = torch.where(
            torch.tensor(dataset.group_array).to(device) == torch.tensor(group).to(device)
        )[0].to(device)
        if num_samples_per_group != "all":
            if num_samples_per_group <= np.sum(dataset.group_array == group):
                # Choose a random subset of the indices
                random_indices_for_group = torch.randperm(len(class_indices))[
                                           :num_samples_per_group
                                           ].to(device)
                # Get the indices for the class chosen in a random order
                class_indices = class_indices[random_indices_for_group].to(device)
            else:
                print('too few samples')
                # Append the indices to the list
        indices = torch.cat((indices, class_indices), dim=0)

    indices = indices.long()
    # change indices to list of integers
    indices = indices.cpu().tolist()
    return indices
