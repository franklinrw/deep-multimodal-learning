import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

class deepCAE(nn.Module):
    def __init__(self):
        super(deepCAE, self).__init__()

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

def train_cae(cae, loader, lossfunction, optimizer, num_epochs=5, visualize=False, device="cuda"):
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
    # Set the model to training mode. This activates layers like dropout and batch normalization.
    cae.train()
    batch_loss_history = []
    epoch_loss_history = []

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
            batch_loss_history.append(loss.item())

        # Calculate the average loss for this epoch.
        avg_loss = running_loss / len(loader)
        epoch_loss_history.append(avg_loss)

        # Print the epoch's summary. The loss is averaged over all batches to get a sense of performance over the entire dataset.
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if visualize == True:
            visualize_reconstruction(cae, loader, num_samples=2, device='cuda')
            cae.train()

    return cae, batch_loss_history, epoch_loss_history

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
    validation_loss_history = []

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
            validation_loss_history.append(loss.item())

    # Calculate the average loss for the validation set.
    avg_val_loss = sum(validation_loss_history) / len(validation_loss_history)
    print("Average Validation Loss:", avg_val_loss)

    return avg_val_loss, validation_loss_history

def get_latent_dataset(model, loader, label=1, device="cuda"):
    """
    This function extracts features from the data using the encoder part of the autoencoder model.
    
    Parameters:
    model: The trained autoencoder model.
    loader (DataLoader): DataLoader that provides batches of data.
    device (str): The device on which the model is, typically either "cpu" or "cuda".
    
    Returns:
    tuple: A tuple containing two elements: the extracted features and the corresponding labels.
    """

    # Set the model to evaluation mode. This is necessary because, in PyTorch, 
    # certain layers behave differently during training compared to evaluation.
    model.to(device)
    model.eval()

    # These lists will store the features and labels.
    features_list = []
    labels_list = []

    # Disabling gradient calculation is useful for inference, 
    # it reduces memory consumption for computations.
    with torch.no_grad():
        # Iterate over the dataset. 'loader' is an iterable.
        # Each iteration returns a batch of images and corresponding labels.
        for batch in loader:
            images, labels = batch
            #print("Labels type 0:", type(labels[0]),"Labels type 1:", type(labels[1]), "labels length:", len(labels), "labels 0 length:", len(labels[0]))
            images = images.to(device)
            # Convert the images to the appropriate type (float).
            images = images.float()

            # Remove any empty dimensions or dimensions with size one.
            images = images.squeeze(1)

            # Rearrange the dimensions of the image. The model expects the channel dimension to be second.
            images = images.permute(0, 3, 1, 2)

            # Pass the images through the model's encoder to get the features.
            features = model.encoder(images)

            # Reshape the features to a 2D tensor, so that each row corresponds to a set of features from one image.
            # The '-1' tells PyTorch to infer the total number of features automatically.
            features_reshaped = features.reshape(features.size(0), -1)

            # Add the features and labels to our lists.
            features_list.append(features_reshaped)
            labels_list.append(labels[label])  # Assuming 'labels[1]' contains the action labels.

    # Concatenate the list of tensors into a single tensor.
    # 'torch.cat' concatenates tensors along a given dimension, here it's along dimension 0.
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Convert extracted features to TensorDatasets
    latent_dataset = torch.utils.data.TensorDataset(all_features, all_labels)

    return latent_dataset


def visualize_reconstruction(model, test_loader, num_samples=5, device='cuda'):
    """
    Visualize the original and reconstructed images from the test set.
    
    Parameters:
    - model: The trained CAE model.
    - test_loader: DataLoader for the test set.
    - num_samples: Number of samples to visualize.
    - device: The device to use ('cuda' or 'cpu').
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get a batch of test data
        images, _ = next(iter(test_loader))
        
        # Move images to the device
        images = images.float().to(device)  # Model expects float and move to the specified device
        images = images.squeeze(1)  # Remove the dimension with size 1
        images = images.permute(0, 3, 1, 2)   # Move the channels dimension to the correct position
        
        # Get the model's reconstructions
        reconstructions = model(images)
        
        # Move images and reconstructions to CPU for visualization
        images = images.cpu().numpy()
        reconstructions = reconstructions.cpu().numpy()
        
        # Plot the original and reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(15, 5))
        
        for i in range(num_samples):
            # Original images
            ax = axes[0, i]
            ax.imshow(np.transpose(images[i], (1, 2, 0)))
            ax.set_title("Original")
            ax.axis('off')
            
            # Reconstructions
            ax = axes[1, i]
            ax.imshow(np.transpose(reconstructions[i], (1, 2, 0)))
            ax.set_title("Reconstruction")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

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

def visualize_latent_space(model, data_loader, n_components=2, random_state=42):
    """
    Visualize the latent space of an autoencoder using t-SNE.
    
    Parameters:
    - model: The trained autoencoder model.
    - data_loader: DataLoader for the dataset.
    - n_components: Number of components for t-SNE (2 or 3). Default is 2.
    - random_state: Random state for t-SNE. Default is 42.
    """
    # Collect latent vectors
    latent_vectors = collect_latent_vectors(model, data_loader)
    
    # Perform t-SNE
    tsne = TSNE(n_components=n_components, random_state=random_state)
    latent_tsne = tsne.fit_transform(latent_vectors)
    
    # 2D Visualization
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.5)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('2D Visualization of Latent Space')
        plt.show()
    
    # 3D Visualization
    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(latent_tsne[:, 0], latent_tsne[:, 1], latent_tsne[:, 2])
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.title('3D Visualization of Latent Space')
        plt.show()
    
    else:
        raise ValueError("n_components must be either 2 or 3.")
