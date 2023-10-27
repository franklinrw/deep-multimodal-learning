from torch import nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        # It's good practice to define each layer separately in case you need individual layer references
        # or apply different specific parameters (like different dropout values, batch normalization, etc.)
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)  # Ensure output_dim matches the number of classes

        self.dropout = nn.Dropout(0.5)  # Can adjust the dropout rate if needed

    def forward(self, x):
        # Apply layer by layer transformations
        x = F.relu(self.fc1(x))  # ReLU activation for non-linearity
        x = self.dropout(x)  # Dropout for regularization

        x = F.relu(self.fc2(x))  # ReLU activation for non-linearity
        x = self.dropout(x)  # Dropout for regularization

        # No activation is used before the output layer with CrossEntropyLoss
        x = self.fc3(x)  # The network's output are the class scores

        return x

class SimpleCAE(nn.Module):
    def __init__(self):
        super(SimpleCAE, self).__init__()

        # Simplified Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=4, stride=2, padding=1),  # [batch, 12, 128, 96] assuming input is [batch, 3, 256, 192]
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),  # [batch, 24, 64, 48]
            nn.ReLU(),
            # You can continue to add more layers here if needed
        )

        # Simplified Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),  # [batch, 12, 128, 96]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2, padding=1),  # [batch, 3, 256, 192]
            nn.Sigmoid()  # Sigmoid because we are probably dealing with images (normalized to [0, 1])
        )

    def forward(self, x):
        x = self.encoder(x)  # Encode the input
        x = self.decoder(x)  # Decode the encoded representation
        return x  # Return the reconstructed output


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 128x96
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64x48
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x24
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x12
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x6
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16x12
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32x24
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x48
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 128x96
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 256x192
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Input img -> Hidden dim -> mean, std -> Parameterization trick -> Decoder -> Output img
class VAE(nn.Module):
    def __init__(self, input_dim=24576, h_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        

    def encode(self, x):
        h = F.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma
    
    def decode(self, z):
        h = F.relu(self.z_2hid(z))
        return self.hid_2img(h)  # Removed sigmoid activation because of MSE loss?

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma
