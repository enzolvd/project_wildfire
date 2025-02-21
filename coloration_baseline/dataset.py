import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, Subset

class Coloring_Dataset(Dataset):

    def __init__(self, dataset_dir: Path, colors_dict = {0 : [0,0,0], 1 : [0, 100, 0], 2 : [0, 0, 100], 3 : [100, 0, 0]}):
        self.dir = dataset_dir
        self.colors_dict = colors_dict
        self.image_names = []
        for d in os.listdir(self.dir):
            for file in os.listdir(self.dir/d):
                self.image_names.append(os.path.join(d,file))

    def add_color(self, X):
        label = torch.zeros((len(self.colors_dict)))
        color_id = torch.multinomial(torch.Tensor(list(self.colors_dict.keys())), 1, replacement=True)
        X = torch.remainder(X + torch.Tensor(self.colors_dict[color_id.item()]).view(3,1,1), 255)
        label[color_id] = 1
        return X, label
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.dir,
                                self.image_names[idx])
        image = transforms.ToTensor()(Image.open(img_name))
        image, label = self.add_color(image)

        return image, label
    
def get_all_datasets(
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

    return train_dataset, val_dataset, test_dataset

def add_color(X, colors_dict = {0 : [0,0,0], 1 : [0, 100, 0], 2 : [0, 0, 100], 3 : [100, 0, 0]}, label=0):
    # label = torch.zeros((len(colors_dict)))
    color_id = torch.multinomial(torch.Tensor(list(colors_dict.keys())), 1, replacement=True)
    color_id = label
    X = torch.remainder(X + torch.Tensor(colors_dict[color_id]).view(3,1,1), 255)
    # label[color_id] = 1
    return X, label

if __name__ == '__main__':
    image_name = os.path.join("..","wildfire_datasets","train","wildfire","-60.1853,50.2269.jpg")

    image = transforms.ToTensor()(Image.open(image_name))
    f = plt.figure()
    
    X_colored, id = add_color(image, label=0)
    ax1 = f.add_subplot(2,2,1)
    ax1.imshow(X_colored.transpose(0,2))
    plt.axis('off')

    X_colored, id = add_color(image, label=1)
    ax2 = f.add_subplot(2,2,2)
    ax2.imshow(X_colored.transpose(0,2))
    plt.axis('off')

    X_colored, id = add_color(image, label=2)
    ax3 = f.add_subplot(2,2,3)
    ax3.imshow(X_colored.transpose(0,2))
    plt.axis('off')

    X_colored, id = add_color(image, label=3)
    ax4 = f.add_subplot(2,2,4)
    ax4.imshow(X_colored.transpose(0,2))
    plt.axis('off')



    plt.savefig("colored_images.jpg")

