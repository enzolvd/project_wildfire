import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


from model import *
from dataset import * 
from utils import *
from train import *

if __name__ == '__main__':
    data_path = Path("../data")
    pretraining_path = data_path / 'train'
    val_path = data_path / 'valid'
    test_path = data_path / 'test'

    data_transforms = {
            'pretrain': transforms.Compose([transforms.ToTensor()]),
            'valid': transforms.Compose([transforms.ToTensor()]),
            'test': transforms.Compose([transforms.ToTensor()]),
        }

    coloring_dataset = Coloring_Dataset(pretraining_path)

    pretrain_idx, preval_idx = train_test_split(
            np.arange(len(coloring_dataset)),
            test_size=0.2,
            random_state=42,
            shuffle=True,
            # stratify=coloring_dataset.label
        )

    pretrain_dataset = Subset(coloring_dataset, pretrain_idx)
    preval_dataset = Subset(coloring_dataset, preval_idx)

    train_dataset, val_dataset, test_dataset = get_all_datasets(
            val_path=val_path,
            test_path=test_path,
            transforms_dict=data_transforms
        )
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=8)

    model = Baseline(4)
    result_path = Path("../outputs/coloration")
    os.makedirs(result_path / "coloring", exist_ok = True)
    os.makedirs(result_path / "frozen_wildfire", exist_ok = True)
    os.makedirs(result_path / "wildfire", exist_ok = True)
    os.makedirs(os.path.join("../outputs/coloration", "pretrained"), exist_ok = True)
    os.makedirs(os.path.join("../outputs/coloration", "classification"), exist_ok = True)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    train_losses, val_losses, val_accuracies = train(1, model, optimizer, device, pretrain_dataset, preval_dataset, criterion)

    print("For the color recognition task:", train_losses[-1], val_losses[-1], val_accuracies[-1])

    plot_curves(train_losses, val_losses, val_accuracies, save_path="../outputs/coloration/results.png", title_suffix="")

    model.swap_to_classification_head(2)
    model = model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    train_losses, val_losses, val_accuracies = train(1, model, optimizer, device, train_dataset, val_dataset, criterion, is_classification=True)

    print("For the wildfire detection task:", train_losses[-1], val_losses[-1], val_accuracies[-1])

    plot_curves(train_losses, val_losses, val_accuracies, save_path="../outputs/coloration/frozen_wildfire_results.png", title_suffix="")

    model.unfreeze_layers()
    model = model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    train_losses, val_losses, val_accuracies = train(1, model, optimizer, device, train_dataset, val_dataset, criterion, is_classification=True)

    print("For the wildfire detection task:", train_losses[-1], val_losses[-1], val_accuracies[-1])

    plot_curves(train_losses, val_losses, val_accuracies, save_path="../outputs/coloration/wildfire_results.png", title_suffix="")

    test_losses, test_acc = validate(model, test_loader, criterion, device, is_classification=True)

    print(f"For the test dataset: final loss = {test_losses[-1]}, test accuracy = {test_acc/len(test_dataset)}")


