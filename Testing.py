import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, os.path.basename(img_path)  # Also return the filename

# VAE Model
class VAE(nn.Module):
    def __init__(self, img_size=640, latent_dim=256):
        super(VAE, self).__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU()
        )

        # Calculate the flattened size dynamically based on the encoder output
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            encoded_output = self.encoder(dummy_input)
            self.flattened_size = encoded_output.view(1, -1).shape[1]  # Auto-detect correct size

        # Latent space
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder network
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()  # To ensure the output is in the range [0, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)  # Flatten the encoded features
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 40, 40)  # Adjust the reshape size based on the encoder's output
        x = self.decoder(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Loss Function (VAE)
def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x.view(-1, 3 * 640 * 640), x.view(-1, 3 * 640 * 640), reduction='sum')
    # KL divergence loss
    return MSE + 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)

# Data transform
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Inference function
def find_anomalous_regions(model, dataloader):
    model.eval()
    with torch.no_grad():
        for img, filename in dataloader:  # Unpack filename
            img = img.to(device)
            recon, mu, logvar = model(img)

            # Compute reconstruction error
            recon_error = nn.functional.mse_loss(recon.view(-1, 3 * 640 * 640), 
                                                 img.view(-1, 3 * 640 * 640), reduction='none')
            recon_error = recon_error.view(img.shape[0], 3, 640, 640).mean(dim=1)

            # Visualize anomalies using the original filename
            visualize_anomalies(img[0], recon[0], recon_error[0], filename[0])

# Updated visualization function to show anomaly scores with a heatmap
def visualize_anomalies(original_img, reconstructed_img, recon_error, filename):
    # Convert tensors to numpy arrays
    original_img = original_img.cpu().numpy().transpose(1, 2, 0)
    reconstructed_img = reconstructed_img.cpu().numpy().transpose(1, 2, 0)
    recon_error = recon_error.cpu().numpy()

    # Normalize error map
    error_map = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min())

    # Apply heatmap
    heatmap = cv2.applyColorMap((error_map * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    alpha = 0.5
    anomaly_overlay = (1 - alpha) * original_img + alpha * (heatmap / 255.0)
    anomaly_overlay = np.clip(anomaly_overlay, 0, 1)

    # Save the heatmap image with the original filename
    masked_images_path = "masked_v2"
    if not os.path.exists(masked_images_path):
        os.makedirs(masked_images_path)

    masked_image_path = os.path.join(masked_images_path, filename)  # Use original filename
    plt.imsave(masked_image_path, np.clip(anomaly_overlay, 0, 1))

    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original_img)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(reconstructed_img)
    ax[1].set_title("Reconstructed Image")
    ax[1].axis('off')

    ax[2].imshow(anomaly_overlay)
    ax[2].set_title("Anomaly Heatmap")
    ax[2].axis('off')

    plt.show()

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
vae = VAE(img_size=640, latent_dim=256).to(device)

# Load the pre-trained model
vae.load_state_dict(torch.load('vae_model.pth', weights_only=True))
vae.to(device)

# Dataset and DataLoader for testing
test_dataset = ImageDataset("test_v2_cropped", transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Run inference on test set
find_anomalous_regions(vae, test_dataloader)