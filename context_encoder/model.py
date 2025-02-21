import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, hidden_dim=64, in_channel=3):
        super().__init__()
        
        # Adjusted convolutional layers for 350x350 input
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=hidden_dim//8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim//8),
            nn.LeakyReLU(0.2)
        )  # Output: 175x175
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim//8, out_channels=hidden_dim//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim//4),
            nn.LeakyReLU(0.2)
        )  # Output: 87x87
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim//4, out_channels=hidden_dim//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.LeakyReLU(0.2)
        )  # Output: 43x43
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim//2, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2)
        )  # Output: 21x21

    def forward(self, x):
        x1 = self.conv1(x)    # 175x175
        x2 = self.conv2(x1)   # 87x87
        x3 = self.conv3(x2)   # 43x43
        x4 = self.conv4(x3)   # 21x21
        return x4, (x1, x2, x3)  

class ChannelWiseMLP(nn.Module):
    def __init__(self, hidden_dim=64, size_latent=21*21):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Enhanced per-channel processing
        self.FC = nn.ModuleList([
            nn.Sequential(
                nn.Linear(size_latent, size_latent*2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(size_latent*2, size_latent),
                nn.ReLU()
            ) for _ in range(hidden_dim)
        ])
        
        # Cross-channel interaction
        self.channel_mixing = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, input):
        batch_size, hidden_dim, height, width = input.size()
        input = input.view(batch_size, hidden_dim, -1)
        
        # Process each channel
        output = torch.stack([self.FC[i](input[:,i,:]) for i in range(hidden_dim)], dim=1)
        output = output.view(batch_size, hidden_dim, height, width)
        
        # Mix channels
        output = self.channel_mixing(output)
        return output

class Decoder(nn.Module):
    def __init__(self, hidden_dim=64, out_channel=3):
        super().__init__()
        
        # 21x21 -> 43x43 (matching x3)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU()
        )  
        
        # 43x43 -> 87x87 (matching x2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim//4, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim//4),
            nn.ReLU()
        )  
        
        # 87x87 -> 175x175 (matching x1)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//8, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim//8),
            nn.ReLU()
        )  
        
        # 175x175 -> 350x350
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim//4, out_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )  

    def forward(self, x, skip_features):
        x1, x2, x3 = skip_features
        
        d1 = self.deconv1(x)
        d1_cat = torch.cat([d1, x3], dim=1)
        
        d2 = self.deconv2(d1_cat)
        d2_cat = torch.cat([d2, x2], dim=1)
        
        d3 = self.deconv3(d2_cat)
        d3_cat = torch.cat([d3, x1], dim=1)
        
        out = self.deconv4(d3_cat)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channel=3, ndf=64, dropout_rate=0.5):
        super().__init__()
        
        # Input layer: 350x350 -> 175x175
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Layer 2: 175x175 -> 87x87
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Layer 3: 87x87 -> 43x43
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate + 0.1)  
        )
        
        # Layer 4: 43x43 -> 21x21
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate + 0.1)  
        )
        
        # Layer 5: 21x21 -> 10x10
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate + 0.1)  
        )
        
        # Output layer: 10x10 -> 1x1
        self.conv6 = nn.Sequential(
            nn.Conv2d(ndf * 16, 1, kernel_size=10, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.conv3(x)  
        x = self.conv4(x)  
        x = self.conv5(x)
        x = self.conv6(x)  
        return x.view(-1, 1).squeeze(1)

class ContextEncoder(nn.Module):
    def __init__(self, hidden_dim=64, in_channel=3):
        super().__init__()
        
        self.encoder = Encoder(hidden_dim=hidden_dim, in_channel=in_channel)
        self.channel_wise_mlp = ChannelWiseMLP(hidden_dim=hidden_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim, out_channel=in_channel)

    def forward(self, x):
        # Encode
        latent, skip_features = self.encoder(x)
        
        # Process features
        processed_latent = self.channel_wise_mlp(latent)
        
        # Decode with skip connections
        output = self.decoder(processed_latent, skip_features)
        return output

def reconstruction_loss(prediction, target, mask):
    """
    Compute the reconstruction loss (L2) only in the masked region
    """
    loss = F.mse_loss(prediction * (1-mask), target * (1-mask), reduction='sum')
    return loss / ((1-mask).sum() + 1e-6)

def adversarial_loss(discriminator, real_img, fake_img, mask):
    # Apply mask to real and fake images
    real_masked = real_img * (1 - mask)
    fake_masked = fake_img * (1 - mask)

    # Compute gradient penalty
    gradient_penalty = compute_gradient_penalty(discriminator, real_masked, fake_masked)

    # Wasserstein loss
    d_real = discriminator(real_masked)
    d_fake = discriminator(fake_masked.detach())

    d_loss = -(torch.mean(d_real) - torch.mean(d_fake)) + 10 * gradient_penalty
    g_loss = -torch.mean(discriminator(fake_masked))

    return d_loss, g_loss

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Compute discriminator output on interpolated samples
    d_interpolates = D(interpolates)

    # Compute gradients 
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute the gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def total_loss(prediction, target, mask, discriminator, lambda_rec=0.999, lambda_adv=0.001):
    # Reconstruction loss in the masked region
    rec_loss = reconstruction_loss(prediction, target, mask)
    
    # Adversarial loss only in the masked region
    _, g_loss = adversarial_loss(discriminator, target, prediction, mask)
    
    # Combined loss with weighting from paper
    total = lambda_rec * rec_loss + lambda_adv * g_loss
    return total, rec_loss, g_loss  # Return individual losses for monitoring