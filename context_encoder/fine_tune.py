import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


from classifier import create_wildfire_classifier
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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
    import os
    
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



def main():
    parser = argparse.ArgumentParser(description="Train a wildfire prediction model.")
    parser.add_argument("--model", type=str, default="run_4", help="Which model to train.")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints", help="Path to Swin checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory for dataset.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save checkpoints and plots.")
    parser.add_argument("--type", type=str, default="best", help="Directory to save checkpoints and plots.")
    args = parser.parse_args()

    # -------------------
    # Prepare directories
    # -------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # -------------------
    # Prepare Data
    # -------------------
    dataset_path = Path(args.data_dir)
    pretrain_path = dataset_path / 'train'
    val_path = dataset_path / 'valid'
    test_path = dataset_path / 'test'

    data_transforms = {
        'pretrain': transforms.Compose([transforms.ToTensor()]),
        'valid': transforms.Compose([transforms.ToTensor()]),
        'test': transforms.Compose([transforms.ToTensor()]),
    }

    _, train_dataset, val_dataset, test_dataset = get_all_datasets(
        pretrain_path=pretrain_path,
        val_path=val_path,
        test_path=test_path,
        transforms_dict=data_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=8)

    path = args.checkpoints+'/'+args.model+f'_{args.type}_model.pth'
    best_paths = []
    train_losses_model = []
    val_losses_model = []
    val_accuracies_model = []
    freeze = [False, True]
    for freeze_backbone in freeze:
        print(f'Training with freeze backbone={freeze_backbone}')
        model = create_wildfire_classifier(path, freeze_backbone=freeze_backbone)

        best_model_path = output_dir / f'{freeze_backbone}_classifier.pth'
        best_paths.append(best_model_path)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        # For plotting
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_accuracy = 0.0

        for epoch in range(args.epochs):
            print(f"\nEpoch [{epoch + 1}/{args.epochs}]")

            # Train
            train_loss = train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                loss_fn=criterion,
                device=device,
            )

            # Validate
            val_loss, correct_predictions = validate(
                model=model,
                data_loader=val_loader,
                loss_fn=criterion,
                device=device,
            )

            epoch_train_loss = np.mean(train_loss)
            epoch_val_loss = np.mean(val_loss)
            val_accuracy = correct_predictions / len(val_dataset)

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint(
                    save_path=str(best_model_path),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_accuracy=val_accuracy
                )
        train_losses_model.append(train_losses)
        val_losses_model.append(val_losses)
        val_accuracies_model.append(val_accuracies)

        checkpoint = load_checkpoint(str(best_model_path), model, optimizer=None)
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation accuracy: {checkpoint['val_accuracy']:.4f}")


        plot_filename = output_dir / f"{args.model}_{freeze_backbone}_training_plot.png"
        plot_curves(train_losses, val_losses, val_accuracies, save_path=str(plot_filename), title_suffix=f"({args.model})")


        test_loss, test_correct = validate(model, test_loader, criterion, device=device)
        test_accuracy = test_correct / len(test_dataset)

        print(f"\nTest Loss: {np.mean(test_loss):.4f} | Test Accuracy: {test_accuracy:.4f}")

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,2,1)
    ax.plot(train_losses_model[0], color='blue', linestyle='-', label='Freezed backbone - train')
    ax.plot(train_losses_model[1], color='red', linestyle='-', label='Unfreezed backbone - train')
    ax.plot(val_losses_model[0], color='blue', linestyle='--', label='Freezed backbone - validation')
    ax.plot(val_losses_model[1], color='red', linestyle='--', label='Unfreezed backbone - validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')

    ax = fig.add_subplot(1,2,2)
    ax.plot(val_accuracies_model[0], color='blue', linestyle='-', label='Freezed backbone')
    ax.plot(val_accuracies_model[1], color='red', linestyle='-', label='Unfreezed backbone')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')

    fig.suptitle('Fine tunning of the Context Encoder')
    path_fig = output_dir / f"{args.model}__training_plot.png"
    plt.savefig(path_fig)

    
if __name__ == "__main__":
    main()
