import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from functions import get_loader, CustomSampler


def validate_cae(cae, loader, lossfunction, is_depth = False, device="cuda"):
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
    psnr_list = []
    ssim_list = []

    # We do not need to compute gradients for evaluation, so we use torch.no_grad() to prevent PyTorch from using memory to track tensors for autograd.
    with torch.no_grad():
        # Iterate over the DataLoader for the validation data.
        for batch in loader:
            # Unpack the batch. We're not interested in labels since autoencoders are unsupervised.
            images, _ = batch
            images = images.to(device)

            #print("shape images:", images.shape)
            images = images.float()
            # Prepare the images for input into the model by ensuring the data type and structure are correct.
            if not is_depth:
                images = images.squeeze(1)
                #print("shape color image after squeeze:", images.shape)
                images = images.permute(0, 3, 1, 2)
                #print("shape color image after permute:", images.shape)

            # Forward pass: pass the images through the model to get the reconstructed images.
            outputs = cae.forward(images)

            # Calculate the loss between the original and the reconstructed images.
            loss = lossfunction(outputs, images)

            images_np = images.detach().cpu().numpy() 
            outputs_np = outputs.detach().cpu().numpy()

            batch_psnr = psnr(images_np, outputs_np, data_range=images_np.max() - images_np.min())
            psnr_list.append(batch_psnr)

            batch_ssim = ssim(images_np, outputs_np, win_size=3, multichannel=True, data_range=images_np.max() - images_np.min())
            ssim_list.append(batch_ssim)

            # Accumulate the loss for reporting.
            validation_loss_history.append(loss.item())

    # Calculate the average loss for the validation set.
    avg_val_loss = sum(validation_loss_history) / len(validation_loss_history)
    print("Average Validation Loss:", avg_val_loss)

    avg_psnr = sum(psnr_list) / len(psnr_list)
    print("Average PSNR:", avg_psnr)

    avg_ssim = sum(ssim_list) / len(ssim_list)
    print("Average SSIM:", avg_ssim)

    return avg_val_loss, validation_loss_history


def train_autoencoder(model, loader, criterion, optimizer, is_depth=False, num_epochs=5, add_noise=False, device="cuda", visualize=False):

    model.train()

    epoch_loss_history = []

    for epoch in range(num_epochs):
        batch_loss_history = []
        psnr_list = []
        ssim_list = []

        for batch in loader:
            images, _ = batch
            images = images.to(device)
            
            #print("shape images:", images.shape)
            images = images.float()
            # Prepare the images for input into the model by ensuring the data type and structure are correct.
            if not is_depth:
                images = images.squeeze(1)
                #print("shape color image after squeeze:", images.shape)
                images = images.permute(0, 3, 1, 2)
                #print("shape color image after permute:", images.shape)

            # Add Gaussian noise if training a DCAE
            if add_noise:
                noisy_images = add_gaussian_noise(images)
                input_images = noisy_images
            else:
                input_images = images

            optimizer.zero_grad()
            outputs = model.forward(input_images)
            loss = criterion(outputs, images)

            images_np = images.detach().cpu().numpy() 
            outputs_np = outputs.detach().cpu().numpy()

            batch_psnr = psnr(images_np, outputs_np, data_range=images_np.max() - images_np.min())
            batch_ssim = ssim(images_np, outputs_np, win_size=3, multichannel=False, data_range=images_np.max() - images_np.min())
            psnr_list.append(batch_psnr)
            ssim_list.append(batch_ssim)

            loss.backward()
            optimizer.step()
            batch_loss_history.append(loss.item())

        avg_loss = sum(batch_loss_history) / len(batch_loss_history)
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)

        epoch_loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], AVG Loss: {avg_loss:.4f}, AVG PSNR: {avg_psnr:.4f}, AVG SSIM: {avg_ssim:.4f}")

        if visualize:
            visualize_reconstruction(model, loader, num_samples=2, depth=is_depth, device=device)

    return model, epoch_loss_history

def get_latent_dataset(model, loader, label=1, add_noise=False, is_depth=False, device="cuda"):
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

            #print("shape images:", images.shape)
            images = images.float()
            # Prepare the images for input into the model by ensuring the data type and structure are correct.
            if not is_depth:
                images = images.squeeze(1)
                #print("shape color image after squeeze:", images.shape)
                images = images.permute(0, 3, 1, 2)
                #print("shape color image after permute:", images.shape)

            # Add Gaussian noise if training a DCAE
            if add_noise:
                noisy_images = add_gaussian_noise(images)
                input_images = noisy_images
            else:
                input_images = images

            # Pass the images through the model's encoder to get the features.
            features = model.encoder(input_images)

            # Reshape the features to a 2D tensor, so that each row corresponds to a set of features from one image.
            # The '-1' tells PyTorch to infer the total number of features automatically.
            features_reshaped = features.reshape(features.size(0), -1)

            # Add the features and labels to our lists.
            features_list.append(features_reshaped)
            labels_list.append(labels[label]) 

    # Concatenate the list of tensors into a single tensor.
    # 'torch.cat' concatenates tensors along a given dimension, here it's along dimension 0.
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Convert extracted features to TensorDatasets
    latent_dataset = torch.utils.data.TensorDataset(all_features, all_labels)

    return latent_dataset


def get_latent_dataloader(cae_model, base_path, objects, tool_names, actions, sensor, set_name, shuffle, batch_size, device, indices=None):
    test_loader = get_loader(base_path, objects, tool_names, actions, sensor, set_name, shuffle=shuffle, batch_size=batch_size)
    dataset = get_latent_dataset(cae_model, test_loader, add_noise=False, is_depth=False, device=device)

    sampler = CustomSampler(dataset, indices) if indices is not None else None

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)


def visualize_reconstruction(model, test_loader, num_samples=5, depth=False, device='cuda'):
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
        
        #print("shape images:", images.shape)
        images = images.float().to(device)
        # Prepare the images for input into the model by ensuring the data type and structure are correct.
        if not depth:
            images = images.squeeze(1)
            #print("shape color image after squeeze:", images.shape)
            images = images.permute(0, 3, 1, 2)
            #print("shape color image after permute:", images.shape)

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


def collect_latent_vectors(model, loader, is_depth=False, device="cuda"):
    model.eval()  # Set the model to evaluation mode
    latent_vectors = []
    with torch.no_grad():
        for batch in loader:
            images, _ = batch
            images = images.to(device)
            #print("shape images:", images.shape)
            images = images.float()
            # Prepare the images for input into the model by ensuring the data type and structure are correct.
            if not is_depth:
                images = images.squeeze(1)
                #print("shape color image after squeeze:", images.shape)
                images = images.permute(0, 3, 1, 2)
                #print("shape color image after permute:", images.shape)
            latent_vec = model.encode(images)
            latent_vec = latent_vec.reshape(latent_vec.size(0), -1)  # Flatten the latent vectors using .reshape()
            latent_vectors.append(latent_vec.cpu().numpy())
    return np.concatenate(latent_vectors)


def visualize_latent_space(model, data_loader, is_depth=False, n_components=2, random_state=42):
    """
    Visualize the latent space of an autoencoder using t-SNE.
    
    Parameters:
    - model: The trained autoencoder model.
    - data_loader: DataLoader for the dataset.
    - n_components: Number of components for t-SNE (2 or 3). Default is 2.
    - random_state: Random state for t-SNE. Default is 42.
    """
    # Collect latent vectors
    latent_vectors = collect_latent_vectors(model, data_loader, is_depth=is_depth)
    
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
    
    
def add_gaussian_noise(images, mean=0., std=0.1):
    """
    Adds Gaussian noise to the input images.

    Args:
        images (torch.Tensor): Input images to add noise to.
        mean (float, optional): Mean of the Gaussian noise distribution. Default is 0.
        std (float, optional): Standard deviation of the Gaussian noise distribution. Default is 0.1.

    Returns:
        torch.Tensor: Noisy images with added Gaussian noise.
    """
    noise = torch.randn_like(images) * std + mean
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images

