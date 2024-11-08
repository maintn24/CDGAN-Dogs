import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 64
batch_size = 128
nz = 100  # Size of latent vector (input to generator)
lr = 0.0002
beta1 = 0.5  # Beta1 hyperparam for Adam optimizer
num_epochs = 20


# Define the Generator
class GeneratorNet(nn.Module):
    """Generator with fully connected layers for 64x64 RGB images."""

    def __init__(self, nz=100, img_size=64, img_channels=3):
        super(GeneratorNet, self).__init__()
        output_size = img_size * img_size * img_channels  # Output for 64x64 RGB

        self.hidden0 = nn.Sequential(
            nn.Linear(nz, 256),
            nn.LeakyReLU(0.2),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(
            nn.Linear(1024, output_size),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x.view(-1, 3, image_size, image_size)  # Reshape to (batch, channels, height, width)


# Define the Discriminator
class DiscriminatorNet(nn.Module):
    """Discriminator with fully connected layers for 64x64 RGB images."""

    def __init__(self, img_size=64, img_channels=3):
        super(DiscriminatorNet, self).__init__()
        input_size = img_size * img_size * img_channels  # 64x64 RGB image

        self.hidden0 = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


# Load and Preprocess the Dataset
data_dir = 'datasets/images'  # Set your path here

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models and optimizers
netG = GeneratorNet(nz=nz, img_size=image_size).to(device)
netD = DiscriminatorNet(img_size=image_size).to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Function to display images
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Training loop
fixed_noise = torch.randn(64, nz, device=device)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Train Discriminator
        netD.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), 1, dtype=torch.float, device=device)

        output = netD(real_data).view(-1)
        lossD_real = criterion(output, label)
        lossD_real.backward()

        noise = torch.randn(batch_size, nz, device=device)
        fake_data = netG(noise)
        label.fill_(0)

        output = netD(fake_data.detach()).view(-1)
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake_data).view(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()

        # Print losses periodically
        if i % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss D: {(lossD_real + lossD_fake).item()}, Loss G: {lossG.item()}")

    # Generate images to see progress
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    img_grid = utils.make_grid(fake, padding=2, normalize=True)
    imshow(img_grid)

# Display final generated images
with torch.no_grad():
    fake_images = netG(torch.randn(64, nz, device=device)).detach().cpu()
imshow(utils.make_grid(fake_images, normalize=True))
