from pathlib import Path
from torch.utils.data import Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


def train_one_epoch(model, optimizer, data_loader, loss_fn, device):
    model.train()
    losses = []


    for x, y in tqdm(data_loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


def validate(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Validating", leave=False):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            losses.append(loss.item())
            correct_predictions += (y_hat.argmax(dim=1) == y).sum().item()

    return losses, correct_predictions


def plot_curves(train_losses, val_losses, val_accuracies, save_path=None, title_suffix=""):
    epochs_range = range(len(val_accuracies))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title(f'Training and Validation Loss {title_suffix}')
    axes[0].legend()

    # Plot Accuracy
    axes[1].plot(epochs_range, val_accuracies, label='Val Accuracy')
    axes[1].set_title(f'Validation Accuracy {title_suffix}')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_checkpoint(
    save_path,
    model,
    optimizer,
    epoch,
    val_accuracy,
):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
    }

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
):

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint