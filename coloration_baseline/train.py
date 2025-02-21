from model import Baseline
import os
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils import *

def train_one_epoch(model, optimizer, data_loader, loss_fn, device):
    model.train()
    losses = []

    for x, y in tqdm(data_loader, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


def validate(model, data_loader, loss_fn, device, is_classification):
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
            if is_classification:
                correct_predictions += (y_hat.argmax(dim=1) == y).sum().item()
            else:
                correct_predictions += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item()

    return losses, correct_predictions

def train(nb_epochs, model, optimizer, device, train_dataset, val_dataset, criterion, is_classification=False):
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)

    if is_classification:
        save_path = os.path.join("checkpoints", "classification")
    else:
        save_path = os.path.join("checkpoints", "pretrained")
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0

    for epoch in tqdm(range(nb_epochs), desc="Epochs"):
        
        train_loss = train_one_epoch(
            model = model, 
            optimizer=optimizer, 
            data_loader=train_loader, 
            loss_fn=criterion, 
            device=device)
        
        val_loss, correct_predictions = validate(
            model=model,
            data_loader=val_loader,
            loss_fn=criterion,
            device=device,
            is_classification=is_classification
        )

        epoch_train_loss = torch.mean(torch.Tensor(train_loss))
        epoch_val_loss = torch.mean(torch.Tensor(val_loss))
        val_accuracy = correct_predictions / len(val_dataset)

        train_losses.append(epoch_train_loss.item())
        val_losses.append(epoch_val_loss.item())
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                save_path=save_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=val_accuracy
            )

    return  train_losses, val_losses, val_accuracies