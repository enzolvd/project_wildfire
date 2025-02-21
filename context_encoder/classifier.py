import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ContextEncoder

class WildfireClassifier(nn.Module):
    def __init__(self, pretrained_encoder, freeze_backbone=True, hidden_dim=64):
        super().__init__()
        
        # Use the pretrained encoder
        self.encoder = pretrained_encoder
        
        # Freeze the encoder weights 
        for param in self.encoder.parameters():
            param.requires_grad = not freeze_backbone
            
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # Get encoder features
        features, _ = self.encoder(x)
        
        # Pass through classifier
        logits = self.classifier(features)
        
        output = F.softmax(logits, dim=-1)
        
        return output.squeeze()

def create_wildfire_classifier(context_encoder_path, freeze_backbone=True,hidden_dim=64):
    """
    Creates a wildfire classifier using a pretrained ContextEncoder
    
    Args:
        context_encoder_path: Path to the pretrained ContextEncoder weights
        hidden_dim: Hidden dimension size (should match pretrained model)
    
    Returns:
        WildfireClassifier model
    """
    # Load pretrained ContextEncoder
    pretrained_model = ContextEncoder(hidden_dim=hidden_dim)    
    context_encoder_weights = torch.load(context_encoder_path)['context_encoder_state_dict']
    pretrained_model.load_state_dict(context_encoder_weights)

    # Create new classifier using pretrained encoder
    classifier = WildfireClassifier(
        pretrained_encoder=pretrained_model.encoder,
        hidden_dim=hidden_dim,
        freeze_backbone=freeze_backbone
    )    
    return classifier