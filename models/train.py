import torch
from tqdm import tqdm
from torch import nn, optim
from torchvision.utils import save_image
    
##### Train functions
def train_ae(model, dataloader, num_epochs=5, learning_rate=1e-3, input_dim=24576, device="cpu"):
    loss_fn = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(dataloader))
        for i, (x, _) in loop:
            x = x.to(device).view(x.shape[0], input_dim)
            x_reconstructed = model(x)
            loss = loss_fn(x_reconstructed, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

def train_vae(model, dataloader, num_epochs=5, learning_rate=1e-3, input_dim=24576, device="cpu"):
    # loss_fn = nn.BCELoss(reduction="sum") #Check out Pytorch Documentation on Loss Functions
    loss_fn = nn.MSELoss(reduction="sum") #More often used for non-binary images
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(dataloader))
        for i, (x, _) in loop:
            # Forward pass
            x = x.to(device).view(x.shape[0], input_dim)
            x_reconstructed, mu, sigma = model(x)

            # Compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x) #reconstruct the input
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) #towards standard gaussian

            # Backprop
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

if __name__ == "__main__":
    # Choose which model to train: Autoencoder or VAE
    model_type = "AE"  # Change to "VAE" for VariationalAutoEncoder

    if model_type == "AE":
        model = Autoencoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
        train_ae(model, train_loader, NUM_EPOCHS, LR_RATE)
    elif model_type == "VAE":
        model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
        train_vae(model, train_loader, NUM_EPOCHS, LR_RATE)

    # Inference (Reconstruction)
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            reconstructed = model(x)
            # Save the reconstructed images (adjust the reshaping to match your data dimensions)
            save_image(reconstructed.view(-1, 3, 64, 128), "reconstructed.png")