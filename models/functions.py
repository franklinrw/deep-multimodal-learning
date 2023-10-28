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
    """
    A custom Dataset class that loads data and labels from pickle files for each combination of parameters.

    This class is particularly useful for machine learning tasks where data is stored in a structured directory
    and needs to be loaded into memory in chunks due to the size of the dataset.

    Attributes:
    data (list or ndarray): The data loaded from the pickle file.
    labels (list or ndarray): The labels corresponding to the data.
    """

    def __init__(self, base_path, objectname, toolname, action, sensor, set_name):
        """
        Initializes the CustomDataset with the paths and loads the data.

        Parameters:
        base_path (str): The base directory where the datasets are located.
        objectname (str): The name of the object category.
        toolname (str): The name of the tool category.
        action (str): The type of action.
        sensor (str): The type of sensor data.
        set_name (str): The name of the dataset (e.g., 'train', 'test', 'validation').
        """

        # Construct the full path for the data and label files.
        data_file_path = os.path.join(base_path, objectname, toolname, action, sensor, f"{set_name}.pkl")
        labels_file_path = os.path.join(base_path, objectname, toolname, action, sensor, f"y_{set_name}.pkl")

        # Load data and labels from the pickle files.
        # It's a good practice to handle file reading errors, which might occur if the file doesn't exist.
        try:
            with open(data_file_path, 'rb') as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            print(f"Data file not found: {data_file_path}")
            self.data = []

        try:
            with open(labels_file_path, 'rb') as f:
                self.labels = pickle.load(f)
        except FileNotFoundError:
            print(f"Labels file not found: {labels_file_path}")
            self.labels = []

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        This method is required by PyTorch and allows the dataset to be used with torch.utils.data.DataLoader.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the data and its corresponding label at a given index.

        This method is required by PyTorch and allows the dataset to be used with torch.utils.data.DataLoader.

        Parameters:
        idx (int): The index of the data to retrieve.

        Returns:
        sample (ndarray or any): The data at the given index.
        label (ndarray or any): The label corresponding to the sample.
        """

        # It's a good practice to handle index errors when retrieving data by index.
        try:
            sample = self.data[idx]
            label = self.labels[idx]
        except IndexError:
            print(f"Index {idx} out of range")
            # Return empty data and label, or alternatively, you could return None, None.
            sample, label = [], []

        return sample, label

def get_datasets(base_path, objectnames, toolnames, actions, sensor, set_name):
    """
    This function generates a list of CustomDataset instances for each combination of parameters provided.
    It is useful for creating datasets that need to be loaded for training/testing in machine learning models,
    especially when dealing with multimodal data or experiments that require various combinations of inputs.

    Parameters:
    base_path (str): The base directory where the datasets are located.
    objectnames (list of str): List of object names to be included in the dataset combinations.
    toolnames (list of str): List of tool names to be included in the dataset combinations.
    actions (list of str): List of actions to be included in the dataset combinations.
    sensor (str): The type of sensor data to be included (e.g., 'color', 'depth', etc.).
    set_name (str): The name of the dataset (e.g., 'train', 'test', 'validation').

    Returns:
    datasets (list of CustomDataset): A list of CustomDataset instances for each specified combination.
    """

    # Initialize an empty list to store the datasets.
    datasets = []

    # Iterate over all combinations of object names, tool names, and actions.
    for objectname in objectnames:
        for toolname in toolnames:
            for action in actions:
                # For each combination, create a new instance of CustomDataset.
                # CustomDataset is assumed to be a class defined elsewhere that loads a specific dataset.
                dataset = CustomDataset(
                    base_path=base_path, 
                    objectname=objectname, 
                    toolname=toolname, 
                    action=action, 
                    sensor=sensor, 
                    set_name=set_name
                )

                # Add the newly created dataset to the list.
                datasets.append(dataset)

    # Return concatenated datasets of the CustomDataset
    return ConcatDataset(datasets)

def get_dummy_loader():
    BASE_PATH = 'C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/data'
    SENSOR = 'color'

    # Combine training sets of all actions for one object
    TOOL_NAMES = ['spatula']
    ACTIONS = ['pull']
    OBJECT_NAMES = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle']

    datasets = get_datasets(BASE_PATH, OBJECT_NAMES, TOOL_NAMES, ACTIONS, SENSOR, 'training')
    # concatenates the data and labels?
    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset=combined_dataset, batch_size=3, shuffle=True)

    return loader

def shuffle_and_split_object_names(object_names, train_ratio, val_ratio, test_ratio):
    """
    Shuffles and splits object names into training, validation, and test sets.

    :param object_names: List of object names.
    :param train_ratio: Proportion of object names to include in the training set.
    :param val_ratio: Proportion of object names to include in the validation set.
    :param test_ratio: Proportion of object names to include in the test set.
    :return: Three lists containing the object names for the training, validation, and test sets.
    """
    # Ensure the ratios are correct
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must add up to 1.0"

    # Shuffle the object names
    random.shuffle(object_names)

    # Calculate split indices
    total_objects = len(object_names)
    train_end_idx = int(total_objects * train_ratio)
    val_end_idx = train_end_idx + int(total_objects * val_ratio)

    # Split the object names
    train_objects = object_names[:train_end_idx]
    val_objects = object_names[train_end_idx:val_end_idx]
    test_objects = object_names[val_end_idx:]

    return train_objects, val_objects, test_objects

def get_loaders(sensor="color", batch_size=8):
    BASE_PATH = 'C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/data'

    # Define the tool names and actions
    TOOL_NAMES = ['hook', 'ruler', 'spatula', 'sshot']
    ACTIONS = ['left_to_right', 'pull', 'push', 'right_to_left']

    # All available object names
    OBJECT_NAMES = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
                '5_blackCoinbag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg','9_pumpkinToy',
                '10_tomatoCan', '11_boxMilk', '12_containerNuts', '13_cornCob', '14_yellowFruitToy',
                '15_bottleNailPolisher', '16_boxRealSense', '17_clampOrange', '18_greenRectangleToy', '19_ketchupToy']

    # Ratios for splitting the datasets
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2  # This is calculated but explicitly defining for clarity

    # Shuffle and split the object names
    train_objects, val_objects, test_objects = shuffle_and_split_object_names(OBJECT_NAMES, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    # Get datasets for the selected combinations
    train_dataset = get_datasets(BASE_PATH, train_objects, TOOL_NAMES, ACTIONS, sensor, "training")
    val_dataset = get_datasets(BASE_PATH, val_objects, TOOL_NAMES, ACTIONS, sensor, "validation")
    test_dataset = get_datasets(BASE_PATH, test_objects, TOOL_NAMES, ACTIONS, sensor, "testing")

    # Create the dataloader for all sets
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

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