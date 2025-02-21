import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import wandb

from utils.utils_gan import (
    train_one_epoch,
    validation,
    save_checkpoint,
    get_mask,
    plot_comparison
)
from utils.load_dataset import get_all_datasets
from model import ContextEncoder, Discriminator
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def load_checkpoint(checkpoint_path, context_encoder, discriminator, g_optimizer, d_optimizer):
    checkpoint = torch.load(checkpoint_path)
    context_encoder.load_state_dict(checkpoint['context_encoder_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_losses']

def main():
    parser = argparse.ArgumentParser(description="Train a wildfire prediction model.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/context_encoder", help="Path to checkpoint.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory for dataset.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Size of hidden dim.")
    parser.add_argument("--mask_size", type=int, default=50, help="Size of mask")
    parser.add_argument("--run_name", type=str, default='default_run', help="Name of the run")
    parser.add_argument("--bar_load", action="store_true", help="Load from checkpoint")
    parser.add_argument("--load_from_checkpoint", action="store_true", help="Load from checkpoint")
    args = parser.parse_args()

    # -------------------
    # Prepare directories
    # -------------------
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # -------------------
    # Prepare Data
    # -------------------
    dataset_path = Path(args.data_dir)
    pretrain_path = dataset_path / 'train'
    val_path = dataset_path / 'valid'
    test_path = dataset_path / 'test'
    data_transforms = {
        'pretrain': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
        ]),
    }
    pretrain_dataset, train_dataset, _, _ = get_all_datasets(pretrain_path=pretrain_path,
                                                                              val_path=val_path,
                                                                              test_path=test_path,
                                                                              transforms_dict=data_transforms)
    train_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader   = DataLoader(train_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=8)

    # -------------------
    # Training helper with wandb
    # -------------------
    # Check if a run with the given name already exists
    api = wandb.Api()
    project = "wildfire"
    runs = api.runs(project)
    existing_run = None
    for run in runs:
        if run.name == args.run_name:
            existing_run = run
            break

    # Initialize wandb run
    if existing_run:
        wandb.init(
            project=project,
            name=args.run_name,
            id=existing_run.id,
            resume='must'
        )
        print(f"Resuming run {existing_run.id}")
    else:
        wandb.init(
            project=project,
            name=args.run_name,
            config={
                "learning_rate": args.lr,
                "architecture": "ContextEncoder",
                "epochs": args.epochs,
                "hidden_dim": args.hidden_dim
            }
        )
        print("Starting a new run")

    # -------------------
    # Prepare Model
    # -------------------
    best_model_path = checkpoint_dir / f"{args.run_name}_best_model.pth"
    current_model_path = checkpoint_dir / f"{args.run_name}_last_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For plotting
    train_losses = []
    val_losses = []
    best_val_loss = 1
    context_encoder = ContextEncoder().to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = torch.optim.Adam(context_encoder.parameters(), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5)

    if args.load_from_checkpoint:
        checkpoint_path = checkpoint_dir / f"{args.run_name}_last_model.pth"
        if checkpoint_path.exists():
            epoch, val_losses = load_checkpoint(checkpoint_path, context_encoder, discriminator, g_optimizer, d_optimizer)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Checkpoint {checkpoint_path} does not exist. Starting training from scratch.")

    # Training loop
    print(args.bar_load)
    for epoch in range(args.epochs):
        # Train
        train_losses = train_one_epoch(
            context_encoder=context_encoder,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            data_loader=train_loader,
            device=device,
            mask_size=args.mask_size,
            bar_load = args.bar_load
        )

        # Validate
        val_losses = validation(
            context_encoder=context_encoder,
            discriminator=discriminator,
            data_loader=val_loader,
            device=device,
            mask_size=args.mask_size,
            bar_load = args.bar_load
        )

        if not args.bar_load:
            print(f'Epoch {epoch} - Train loss :{train_losses['total_loss']:.6f} - Val loss {val_losses['total_loss']:.6f}')

        epoch_metrics = {}

        # Visualization and metrics logging
        true = train_dataset[0][0][None].to(device)
        mask = get_mask(true.shape, mask_size=(args.mask_size, args.mask_size)).to(device)
        input_masked = true * mask

        # Generate prediction
        context_encoder.eval()
        with torch.no_grad():
            pred = context_encoder(input_masked)

        # Convert to numpy for visualization
        complete = pred*(1-mask) + input_masked
        complete = complete[0].permute(1,2,0).detach().cpu().numpy()
        pred_np = pred[0].permute(1,2,0).detach().cpu().numpy()
        true_np = true[0].permute(1,2,0).detach().cpu().numpy()
        masked_np = input_masked[0].permute(1,2,0).detach().cpu().numpy()

        # Create visualizations
        fig_original = plot_comparison(true_np, complete, 'Original vs Predicted')
        fig_masked = plot_comparison(masked_np, pred_np, 'Masked Input vs Predicted')

        # Log visualizations
        epoch_metrics = {
            'Original_vs_Predicted': wandb.Image(fig_original),
            'Masked_vs_Predicted': wandb.Image(fig_masked)
        }

        # Clean up figures
        plt.close(fig_original)
        plt.close(fig_masked)

        # Log training metrics
        epoch_metrics.update({
            'epoch/train_total_loss': train_losses['total_loss'],
            'epoch/train_reconstruction_loss': train_losses['reconstruction_loss'],
            'epoch/train_adversarial_loss': train_losses['reconstruction_loss'],
            'epoch/train_discriminator_loss': train_losses['discriminator_loss'],
            'epoch/val_total_loss': val_losses['total_loss'],
            'epoch/val_reconstruction_loss': val_losses['reconstruction_loss'],
            'epoch/val_adversarial_loss': val_losses['adversarial_loss']
        })

        wandb.log(epoch_metrics)

        # Save best model based on validation total loss
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            save_checkpoint(
                save_path=str(best_model_path),
                context_encoder=context_encoder,
                discriminator=discriminator,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                epoch=epoch,
                val_losses=val_losses
            )
            print(f'Saved best checkpoint at epoch {epoch} - Val Loss: {best_val_loss:.6f}')

        # Save current model
        save_checkpoint(
            save_path=str(current_model_path),
            context_encoder=context_encoder,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            epoch=epoch,
            val_losses=val_losses
        )

    wandb.finish()

if __name__ == "__main__":
    main()
