from pathlib import Path
from torch.utils.data import Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
import numpy as np


def get_all_datasets(
    pretrain_path: Path,
    val_path: Path,
    test_path: Path,
    transforms_dict: dict
):
    """
    Creates pretrain, train, validation, and test datasets.

    1. Uses ImageFolder on `pretrain_path` for 'pretrain_dataset'.
    2. Uses ImageFolder on `val_path` for splitting into 'train_dataset' and 'val_dataset'.
    3. Uses ImageFolder on `test_path` for 'test_dataset'.

    Args:
        pretrain_path (Path): Path to pretrain folder.
        val_path (Path): Path to valid folder.
        test_path (Path): Path to test folder.
        transforms_dict (dict): Dictionary containing transforms for 'pretrain', 'valid', and 'test'.

    Returns:
        (Dataset, Dataset, Dataset, Dataset):
            Tuple of pretrain_dataset, train_dataset, val_dataset, test_dataset.
    """

    pretrain_dataset = datasets.ImageFolder(pretrain_path, transform=transforms_dict['pretrain'])
    val_dataset_full = datasets.ImageFolder(val_path, transform=transforms_dict['valid'])
    test_dataset = datasets.ImageFolder(test_path, transform=transforms_dict['test'])

    # Split validation dataset into train/val
    train_idx, validation_idx = train_test_split(
        np.arange(len(val_dataset_full)),
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=val_dataset_full.targets
    )

    train_dataset = Subset(val_dataset_full, train_idx)
    val_dataset = Subset(val_dataset_full, validation_idx)

    return pretrain_dataset, train_dataset, val_dataset, test_dataset