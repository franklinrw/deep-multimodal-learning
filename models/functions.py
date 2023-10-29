import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import random

class CustomDataset(Dataset):
    def __init__(self, file_path):
        """
        Initializes the CustomDataset with the file path and loads the data.

        Parameters:
        file_path (str): The full path to the pickle file containing the data.
        """
        self.data_with_labels = []

        # Load data and labels from the pickle file.
        try:
            with open(file_path, 'rb') as f:
                self.data_with_labels = pickle.load(f)
        except FileNotFoundError:
            print(f"Data file not found: {file_path}")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_with_labels)

    def __getitem__(self, idx):
        """
        Retrieves the data and its corresponding label at a given index.
        """
        try:
            sample, label = self.data_with_labels[idx]
        except IndexError:
            print(f"Index {idx} out of range")
            sample, label = None, None  # Or handle this appropriately

        return sample, label
    
def get_datasets(base_path, objectnames, toolnames, actions, sensor, set_name):
    """
    This function generates a list of CustomDataset instances for each pickle file in the directory structure.
    It is useful for creating datasets that need to be loaded for training/testing in machine learning models.

    Parameters:
    base_path (str): The base directory where the datasets are located.
    set_name (str): The name of the dataset (e.g., 'train', 'test', 'validation').
    sensors (list of str): List of sensors to be included in the dataset.

    Returns:
    datasets (ConcatDataset): A concatenated dataset comprising all the CustomDataset instances.
    """

    # Initialize an empty list to store the datasets.
    datasets = []

    # Iterate over all combinations of object names, tool names, and actions.
    for objectname in objectnames:
        for toolname in toolnames:
            for action in actions:
                # \Bureaublad\ARC\deep-multimodal-learning\data_v2\0_woodenCube\hook\left_to_right\color
                path = os.path.join(base_path, objectname, toolname, action, sensor, f"{set_name}_{sensor}.pkl")
                
                # Check if the path exists
                if not os.path.exists(path):
                    print(f"Directory does not exist: {path}")
                    continue

                dataset = CustomDataset(file_path=path)
                datasets.append(dataset)

    # Return a ConcatDataset instance comprising all the datasets
    return ConcatDataset(datasets)

def get_loader(base_path, objectnames, toolnames, actions, sensor, set_name, batch_size=8):

    dataset = get_datasets(base_path, objectnames, toolnames, actions, sensor, set_name)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(set_name == 'training'))

    return loader

def visualize_reconstruction(model, test_loader, num_samples=5):
    """
    Visualize the original and reconstructed images from the test set.
    
    Parameters:
    - model: The trained CAE model.
    - test_loader: DataLoader for the test set.
    - num_samples: Number of samples to visualize.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get a batch of test data
        images, _ = next(iter(test_loader))
        
        # Move images to the device
        images = images.float() # Model expects float
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

def calculate_accuracy(output, labels):
    # Get predictions
    predicted = output.max(1).indices  # The underscore is used to ignore the actual maximum values returned

    # Check where the predictions and labels are the same
    correct_predictions = (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = correct_predictions / labels.size(0)  # or labels.numel() for the total number of elements

    return accuracy * 100.0  # returns as percentage

def extract_features(loaded_ae, loader, device="cuda"):
    """
    This function extracts features from the data using the encoder part of the autoencoder model.
    
    Parameters:
    loaded_ae (nn.Module): The trained autoencoder model.
    loader (DataLoader): DataLoader that provides batches of data.
    device (str): The device on which the model is, typically either "cpu" or "cuda".
    
    Returns:
    tuple: A tuple containing two elements: the extracted features and the corresponding labels.
    """

    # Set the model to evaluation mode. This is necessary because, in PyTorch, 
    # certain layers behave differently during training compared to evaluation.
    loaded_ae.to(device)
    loaded_ae.eval()

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
            features = loaded_ae.encoder(images)

            # Reshape the features to a 2D tensor, so that each row corresponds to a set of features from one image.
            # The '-1' tells PyTorch to infer the total number of features automatically.
            features_reshaped = features.reshape(features.size(0), -1)

            # Add the features and labels to our lists.
            features_list.append(features_reshaped)
            labels_list.append(labels[1])  # Assuming 'labels[1]' contains the action labels.

    # Concatenate the list of tensors into a single tensor.
    # 'torch.cat' concatenates tensors along a given dimension, here it's along dimension 0.
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    return all_features, all_labels