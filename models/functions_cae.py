import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

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

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)  # Encode the input
        x = self.decoder(x)  # Decode the encoded representation
        return x  # Return the reconstructed output

import torch.nn as nn

class SimpleCAE_Dropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleCAE_Dropout, self).__init__()

        # Simplified Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=4, stride=2, padding=1),  # [batch, 12, 128, 96]
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout layer
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),  # [batch, 24, 64, 48]
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout layer
            # You can continue to add more layers here if needed
        )

        # Simplified Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),  # [batch, 12, 128, 96]
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout layer
            nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2, padding=1),  # [batch, 3, 256, 192]
            nn.Sigmoid()  # Sigmoid because we are probably dealing with images (normalized to [0, 1])
        )

    def encode(self, x):
        return self.encoder(x)

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

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_cae(cae, loader, lossfunction, optimizer, num_epochs=5, lr=1e-3, device="cuda"):
    """
    This function trains a convolutional autoencoder (CAE).

    Parameters:
    model (nn.Module): The autoencoder model to be trained.
    loader (DataLoader): DataLoader that provides batches of data.
    num_epochs (int): The number of training epochs. Default is 5.
    lr (float): Learning rate for the optimizer. Default is 0.001.

    Returns:
    None: The function trains the model in place and does not return anything.
    """
    # Define the loss function as Mean Squared Error Loss. It's common for reconstruction tasks.

    # Define the optimizer that will be used to minimize the loss. Here, we're using Adam.

    # Set the model to training mode. This activates layers like dropout and batch normalization.
    cae.train()

    # Iterate over the dataset multiple times (each iteration over the entire dataset is called an epoch).
    for epoch in range(num_epochs):
        # Initialize the running loss to zero at the beginning of each epoch.
        running_loss = 0.0

        # Iterate over the DataLoader. Each iteration provides a batch of data.
        for batch in loader:
            # Unpack the batch. We're not interested in labels since autoencoders are unsupervised.
            images, _ = batch
            images = images.to(device)

            #print(images)

            # Prepare the images for input into the model by ensuring the data type and structure are correct.
            images = images.float()
            images = images.squeeze(1)
            images = images.permute(0, 3, 1, 2)

            # Forward pass: pass the images through the model to get the reconstructed images.
            outputs = cae.forward(images)

            #print(outputs)

            # Calculate the loss between the original and the reconstructed images.
            loss = lossfunction(outputs, images)

            # Zero the parameter gradients to prevent accumulation during backpropagation.
            optimizer.zero_grad()

            # Backward pass: compute the gradients of the loss w.r.t. the model parameters.
            loss.backward()

            # Update the model parameters based on the computed gradients.
            optimizer.step()

            # Accumulate the loss for reporting.
            running_loss += loss.item()
            print(f"Batch Loss: {loss:.4f}")

        # Calculate the average loss for this epoch.
        avg_loss = running_loss / len(loader)

        # Print the epoch's summary. The loss is averaged over all batches to get a sense of performance over the entire dataset.
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return cae

def validate_cae(cae, loader, lossfunction, device="cuda"):
    """
    This function evaluates a convolutional autoencoder (CAE) on the validation set.

    Parameters:
    cae (nn.Module): The autoencoder model to be evaluated.
    loader (DataLoader): DataLoader that provides batches of validation data.
    device (str): The device type to be used for evaluation (e.g., "cuda" or "cpu").

    Returns:
    float: The average validation loss over all batches in the validation set.
    """
    # Define the loss function as Mean Squared Error Loss. It's common for reconstruction tasks.

    # Set the model to evaluation mode. This deactivates layers like dropout and batch normalization.
    cae.eval()

    # Initialize the running loss to zero.
    validation_loss = 0.0

    # We do not need to compute gradients for evaluation, so we use torch.no_grad() to prevent PyTorch from using memory to track tensors for autograd.
    with torch.no_grad():
        # Iterate over the DataLoader for the validation data.
        for batch in loader:
            # Unpack the batch. We're not interested in labels since autoencoders are unsupervised.
            images, _ = batch
            images = images.to(device)

            # Prepare the images for input into the model by ensuring the data type and structure are correct.
            images = images.float()
            images = images.squeeze(1)
            images = images.permute(0, 3, 1, 2)

            # Forward pass: pass the images through the model to get the reconstructed images.
            outputs = cae.forward(images)

            # Calculate the loss between the original and the reconstructed images.
            loss = lossfunction(outputs, images)

            # Accumulate the loss for reporting.
            validation_loss += loss.item()

    # Calculate the average loss for the validation set.
    avg_val_loss = validation_loss / len(loader)

    print("Average Validation Loss:", avg_val_loss)

def collect_latent_vectors(model, loader, device="cuda"):
    model.eval()  # Set the model to evaluation mode
    latent_vectors = []
    with torch.no_grad():
        for batch in loader:
            images, _ = batch
            images = images.to(device)
            images = images.float()
            images = images.squeeze(1)
            images = images.permute(0, 3, 1, 2)
            latent_vec = model.encode(images)
            latent_vec = latent_vec.reshape(latent_vec.size(0), -1)  # Flatten the latent vectors using .reshape()
            latent_vectors.append(latent_vec.cpu().numpy())
    return np.concatenate(latent_vectors)
