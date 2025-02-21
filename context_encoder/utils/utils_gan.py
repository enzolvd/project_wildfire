import numpy as np
import torch
from tqdm import tqdm
import io
import wandb
from PIL import Image
import matplotlib.pyplot as plt
from model import reconstruction_loss, adversarial_loss

def get_mask(input_shape, mask_size=(50,50)):
    batch_size, _, height, width = input_shape
    mask_height, mask_width = mask_size

    mask = torch.ones(input_shape)

    top = np.random.randint(0, height - mask_height + 1, batch_size)
    left = np.random.randint(0, width - mask_width + 1, batch_size)

    for batch in range(batch_size):
        mask[batch,:,top[batch]:top[batch] + mask_height, left[batch]:left[batch] + mask_width] = 0

    return mask

def apply_mask(input, mask):

    input_masked = input*mask
    output_masked_gt = input*(1-mask)
    return input_masked, output_masked_gt

def train_one_epoch(context_encoder, discriminator, g_optimizer, d_optimizer, data_loader, device, mask_size=50, lambda_rec=0.999, lambda_adv=0.001, bar_load=False):
    context_encoder.train()
    discriminator.train()

    losses = {
        'total_loss': [],
        'reconstruction_loss': [],
        'adversarial_loss': [],
        'discriminator_loss': []
    }

    pbar = data_loader
    if bar_load:
        pbar = tqdm(data_loader, desc="Training", leave=False)

    for input, _ in pbar:
        input = input.to(device)

        # Generate mask
        mask = get_mask(input.shape, mask_size=(mask_size, mask_size)).to(device)
        input_masked, _ = apply_mask(input, mask)   # Ground truth for masked region

        ## Train discriminator
        d_optimizer.zero_grad(set_to_none=True)

        # Generate fake image
        with torch.no_grad():
            fake_img = context_encoder(input_masked)

        # Compute discriminator loss
        d_loss, _ = adversarial_loss(discriminator, input, fake_img, mask)
        d_loss.backward()
        d_optimizer.step()

        ## Train generator (context encoder)
        g_optimizer.zero_grad(set_to_none=True)

        # Generate fake image again (need new forward pass for generator training)
        fake_img = context_encoder(input_masked)

        # Compute reconstruction loss
        rec_loss = reconstruction_loss(fake_img, input, mask)

        # Compute adversarial loss for generator
        _, g_loss = adversarial_loss(discriminator, input, fake_img, mask)

        # Combined loss
        total_loss = lambda_rec * rec_loss + lambda_adv * g_loss
        total_loss.backward()
        g_optimizer.step()

        # Log losses
        losses['total_loss'].append(total_loss.item())
        losses['reconstruction_loss'].append(rec_loss.item())
        losses['adversarial_loss'].append(g_loss.item())
        losses['discriminator_loss'].append(d_loss.item())

        # Log to wandb
        wandb.log({
            "batch/total_loss": total_loss.item(),
            "batch/reconstruction_loss": rec_loss.item(),
            "batch/adversarial_loss": g_loss.item(),
            "batch/discriminator_loss": d_loss.item()
        })

        # Clean up
        del input, mask, input_masked, fake_img
        torch.cuda.empty_cache()

    # Return average losses for the epoch
    return {k: sum(v)/len(v) for k, v in losses.items()}


def validation(context_encoder, discriminator, data_loader, device, mask_size=50, lambda_rec=0.999, lambda_adv=0.001, bar_load=False):
    context_encoder.eval()
    discriminator.eval()

    losses = {
        'total_loss': [],
        'reconstruction_loss': [],
        'adversarial_loss': []
    }

    pbar = data_loader
    if bar_load:
        pbar = tqdm(data_loader, desc="Validation", leave=False)

    for input, _ in pbar:
        input = input.to(device)

        # Generate mask
        mask = get_mask(input.shape, mask_size=(mask_size, mask_size)).to(device)
        input_masked, _ = apply_mask(input, mask)

        # Generate fake image
        fake_img = context_encoder(input_masked)

        # Compute losses
        rec_loss = reconstruction_loss(fake_img, input, mask)
        _, g_loss = adversarial_loss(discriminator, input, fake_img, mask)
        total_loss = lambda_rec * rec_loss + lambda_adv * g_loss

        # Log losses
        losses['total_loss'].append(total_loss.item())
        losses['reconstruction_loss'].append(rec_loss.item())
        losses['adversarial_loss'].append(g_loss.item())

        # Clean up
        del input, mask, input_masked, fake_img
        torch.cuda.empty_cache()

    # Return average losses
    return {k: sum(v) / len(v) for k, v in losses.items()}

def plot_comparison(true, pred, title):
    """Create comparison plots for model predictions vs ground truth."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    im1 = axes[0].imshow(true)
    axes[0].set_title('True')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    im2 = axes[1].imshow(pred)
    axes[1].set_title('Predicted')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.suptitle(title)
    return fig

def save_checkpoint(
    save_path,
    context_encoder,
    discriminator,
    g_optimizer,
    d_optimizer,
    epoch,
    val_losses,
):
    checkpoint = {
        'epoch': epoch,
        'context_encoder_state_dict': context_encoder.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'val_losses': val_losses,
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(
    checkpoint_path,
    context_encoder,
    discriminator,
    g_optimizer=None,
    d_optimizer=None,
):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model states
    context_encoder.load_state_dict(checkpoint['context_encoder_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Load optimizer states if provided
    if g_optimizer is not None:
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    if d_optimizer is not None:
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    if 'epoch' in checkpoint.keys():
        return {
            'epoch': checkpoint['epoch'],
            'val_losses': checkpoint['val_losses']
        }

