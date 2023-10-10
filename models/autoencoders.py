from torch import nn
import torch.nn.functional as F
import torch

# A plain Autoencoder is not that suitable for the goal at hand
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim=24576, h_dim=200, z_dim=20):
#         super(Autoencoder, self).__init__()

#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, h_dim),
#             nn.ReLU(),
#             nn.Linear(h_dim, z_dim),
#             nn.ReLU()
#         )

#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(z_dim, h_dim),
#             nn.ReLU(),
#             nn.Linear(h_dim, input_dim),
#             nn.Sigmoid()  # Use sigmoid for image reconstruction
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         x_reconstructed = self.decoder(z)
#         return x_reconstructed

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()

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
class VariationalAutoEncoder(nn.Module):
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

if __name__ == "__main__":
    x = torch.randn(1, 24576)
    vae = VariationalAutoEncoder(input_dim=24576)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)