import torch
import torch.nn as nn


class Baseline(nn.Module):
    def __init__(self, num_class):
        super(Baseline, self).__init__()

        self.model = nn.Sequential(

        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.Linear(128, num_class),
        )

    def forward(self, X):
        output = self.model(X)
        return output
    
    def load_weights(self, checkpoint_path):

        model_weights_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)["model_state_dict"]
        weights_dict = {}
        for k in model_weights_dict.keys():
            weights_dict[k[6:]] = model_weights_dict[k]
        self.model.load_state_dict(weights_dict)

        return self.model
    
    def swap_to_classification_head(self, output_channels):
        removed = list(self.model.children())[:-1]
        self.model = torch.nn.Sequential(*removed)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = torch.nn.Sequential(self.model, torch.nn.Linear(128,output_channels))

        return self.model
    
    def unfreeze_layers(self):
        for p in self.model.parameters():
            p.requires_grad = True
        return self.model

    
if __name__ == '__main__':
    model = Baseline(4)
    model.load_weights("checkpoints/pretrained/_epoch=4_acc_0.2.pkl")
    print()
    model.swap_to_classification_head(2)
    print(model.parameters())