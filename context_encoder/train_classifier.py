import argparse
from classifier import create_wildfire_classifier
from utils.utils_classifier import *
from utils.load_dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    parser = argparse.ArgumentParser(description="Train a wildfire prediction model.")
    parser.add_argument("--model", type=str, default="context_encoder", help="Which model to train.")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/context_encoder", help="Path to checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory for dataset.")
    parser.add_argument("--output_dir", type=str, default="./outputs/context_encoder", help="Directory to save plots.")
    args = parser.parse_args()
    
    # -------------------
    # Prepare directories
    # -------------------
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoints)
    output_dir.mkdir(exist_ok=True, parents=True)

    # -------------------
    # Prepare Data
    # -------------------
    dataset_path = Path(args.data_dir)
    pretrain_path = dataset_path / 'train'
    val_path = dataset_path / 'valid'
    test_path = dataset_path / 'test'

    print(pretrain_path)
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
    
    path = checkpoint_dir / f'{args.model}.pt'

    train_losses_model = []
    val_losses_model = []
    val_accuracies_model = []
    freeze = [False, True]

    for freeze_backbone in freeze:
        print(f'Training with frozen backbone={freeze_backbone}')
        model = create_wildfire_classifier(path, freeze_backbone=freeze_backbone)

        best_model_path = checkpoint_dir / f'{freeze_backbone}_classifier.pt'
        
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
        # plot_curves(train_losses, val_losses, val_accuracies, save_path=str(plot_filename), title_suffix=f"({args.model})")
        plt.close()

        test_loss, test_correct = validate(model, test_loader, criterion, device=device)
        test_accuracy = test_correct / len(test_dataset)

        print(f"\nTest Loss: {np.mean(test_loss):.4f} | Test Accuracy: {test_accuracy:.4f}")

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1,2,1)
    ax.plot(train_losses_model[0], color='blue', linestyle='-', label='Unfrozen backbone - train')
    ax.plot(train_losses_model[1], color='red', linestyle='-', label='Frozen backbone - train')
    ax.plot(val_losses_model[0], color='blue', linestyle='--', label='Unfrozen backbone - validation')
    ax.plot(val_losses_model[1], color='red', linestyle='--', label='Frozen backbone - validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax = fig.add_subplot(1,2,2)
    ax.plot(val_accuracies_model[0], color='blue', linestyle='-', label='Unfrozen backbone')
    ax.plot(val_accuracies_model[1], color='red', linestyle='-', label='Frozen backbone')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()

    fig.suptitle('Fine tunning of the Context Encoder')
    path_fig = output_dir / f"{args.model}__training_plot.png"
    plt.savefig(path_fig)

    
if __name__ == "__main__":
    main()
