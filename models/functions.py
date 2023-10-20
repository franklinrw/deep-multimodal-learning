import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, base_path, objectname, toolname, action, sensor, set_name):
        data_file_path = os.path.join(base_path, objectname, toolname, action, sensor, f"{set_name}.pkl")
        labels_file_path = os.path.join(base_path, objectname, toolname, action, sensor, f"y_{set_name}.pkl")
        
        with open(data_file_path, 'rb') as f:
            self.data = pickle.load(f)
        
        with open(labels_file_path, 'rb') as f:
            self.labels = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    
def get_names():
    sensornames = ['color', 'depthcolormap', 'icub_left', 'icub_right']
    toolnames = ['hook', 'ruler', 'spatula', 'sshot']
    actions = ['left_to_right', 'pull', 'push', 'right_to_left']
    objectnames = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
                '5_blackCoinbag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg','9_pumpkinToy',
                '10_tomatoCan', '11_boxMilk', '12_containerNuts', '13_cornCob', '14_yellowFruitToy',
                '15_bottleNailPolisher', '16_boxRealSense', '17_clampOrange', '18_greenRectangleToy', '19_ketchupToy']
    
    return sensornames, toolnames, actions, objectnames
    
def get_datasets_for_combinations(base_path, objectnames, toolnames, actions, sensor, set_name):
    datasets = []
    for objectname in objectnames:
        for toolname in toolnames:
            for action in actions:
                dataset = CustomDataset(base_path=base_path, objectname=objectname, toolname=toolname, action=action, sensor=sensor, set_name=set_name)
                datasets.append(dataset)
    return datasets

def get_dummy_loader():
    BASE_PATH = 'C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/data'
    SENSOR = 'color'

    # Combine training sets of all actions for one object
    TOOL_NAMES = ['spatula']
    ACTIONS = ['pull']
    OBJECT_NAMES = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle']

    datasets = get_datasets_for_combinations(BASE_PATH, OBJECT_NAMES, TOOL_NAMES, ACTIONS, SENSOR, 'training')
    # concatenates the data and labels?
    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset=combined_dataset, batch_size=3, shuffle=True)

    return loader

def get_color_training_loader():
    BASE_PATH = 'C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/data'
    SENSOR = 'color'

    # Combine training sets of all actions for one object
    TOOL_NAMES = ['hook', 'ruler', 'spatula', 'sshot']
    ACTIONS = ['left_to_right', 'pull', 'push', 'right_to_left']
    OBJECT_NAMES = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
                '5_blackCoinbag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg','9_pumpkinToy',
                '10_tomatoCan', '11_boxMilk', '12_containerNuts', '13_cornCob', '14_yellowFruitToy',
                '15_bottleNailPolisher', '16_boxRealSense', '17_clampOrange', '18_greenRectangleToy', '19_ketchupToy']

    datasets = get_datasets_for_combinations(BASE_PATH, OBJECT_NAMES, TOOL_NAMES, ACTIONS, SENSOR, 'training')
    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(dataset=combined_dataset, batch_size=32, shuffle=True)

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

def train_cae(model, loader, num_epochs=5, lr=1e-3):
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
    criterion = nn.MSELoss()

    # Define the optimizer that will be used to minimize the loss. Here, we're using Adam.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set the model to training mode. This activates layers like dropout and batch normalization.
    model.train()

    # Iterate over the dataset multiple times (each iteration over the entire dataset is called an epoch).
    for epoch in range(num_epochs):
        # Initialize the running loss to zero at the beginning of each epoch.
        running_loss = 0.0

        # Iterate over the DataLoader. Each iteration provides a batch of data.
        for batch in loader:
            # Unpack the batch. We're not interested in labels since autoencoders are unsupervised.
            images, _ = batch

            # Prepare the images for input into the model by ensuring the data type and structure are correct.
            images = images.float()
            images = images.squeeze(1)
            images = images.permute(0, 3, 1, 2)

            # Forward pass: pass the images through the model to get the reconstructed images.
            outputs = model(images)

            # Calculate the loss between the original and the reconstructed images.
            loss = criterion(outputs, images)

            # Zero the parameter gradients to prevent accumulation during backpropagation.
            optimizer.zero_grad()

            # Backward pass: compute the gradients of the loss w.r.t. the model parameters.
            loss.backward()

            # Update the model parameters based on the computed gradients.
            optimizer.step()

            # Accumulate the loss for reporting.
            running_loss += loss.item()

        # Calculate the average loss for this epoch.
        avg_loss = running_loss / len(loader)

        # Print the epoch's summary. The loss is averaged over all batches to get a sense of performance over the entire dataset.
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


# def train_cae(model, loader, num_epochs=5, lr=1e-3):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr)

#     for epoch in range(num_epochs):
#         for batch in loader:
#             images, _ = batch  # We don't need labels for autoencoders
#             images = images.float() # Model expects float
#             images = images.squeeze(1)  # Remove the dimension with size 1
#             images = images.permute(0, 3, 1, 2)   # Move the channels dimension to the correct position
#             outputs = model(images)
#             loss = criterion(outputs, images)  # Reconstruction loss
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def train_mlp(mlp, num_epochs, train_loader, val_loader, device="cpu"):
    """
    This function trains a Multilayer Perceptron (MLP) model.

    Parameters:
    mlp (nn.Module): The MLP model to be trained.
    num_epochs (int): The number of training epochs.
    train_loader (DataLoader): DataLoader for the training set.
    val_loader (DataLoader): DataLoader for the validation set.
    device (str): The device where the model and data should be loaded ('cpu' or 'cuda').

    Returns:
    None: The function trains the model in place and does not return anything.
    """

    # Define the loss function. CrossEntropyLoss is commonly used for classification tasks.
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer, specifying the learning rate and the parameters to optimize.
    # Adam is a commonly used optimizer due to its efficiency and minimal requirement for manual tuning.
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    # Iterate over the dataset multiple times (each iteration over the entire dataset is called an epoch).
    for epoch in range(num_epochs):
        # Set the model to training mode. This activates layers like dropout and batch normalization.
        mlp.train()

        # Initialize the running loss to zero at the beginning of each epoch.
        running_loss = 0.0

        # Iterate over the DataLoader for the training set.
        for features, labels in train_loader:
            # Move data to the specified device (if not already there).
            features, labels = features.to(device), labels.to(device)

            # Forward pass: pass the input through the model to get the predictions.
            outputs = mlp(features)

            # Calculate the loss by comparing the model output and actual labels.
            loss = criterion(outputs, labels)

            # Zero the parameter gradients to prevent accumulation during backpropagation.
            optimizer.zero_grad()

            # Backward pass: compute the gradients of the loss w.r.t. the model parameters.
            loss.backward()

            # Update the model parameters based on the computed gradients.
            optimizer.step()

            # Accumulate the loss for reporting.
            running_loss += loss.item()

        # Calculate the average loss for this epoch.
        avg_train_loss = running_loss / len(train_loader)

        # Validation phase: we evaluate the model on the validation set to check its performance on unseen data.
        mlp.eval()  # Set the model to evaluation mode.

        val_loss = 0.0
        correct = 0
        total = 0

        # We do not need to calculate gradients for evaluation, so we use torch.no_grad() to prevent PyTorch from using memory to track tensors for autograd.
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                # Forward pass: pass the input through the model to get the predictions.
                outputs = mlp(features)

                # Calculate the loss by comparing the model output and actual labels.
                loss = criterion(outputs, labels)

                # Accumulate the validation loss.
                val_loss += loss.item()

                # Calculate the number of correct predictions.
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Calculate the average validation loss and the accuracy over the entire validation set.
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / total

        # Print the summary for this epoch.
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")

# def train_mlp(mlp, num_epochs, train_loader, val_loader, device="cpu"):
#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

#     # Training loop
#     for epoch in range(num_epochs):
#         mlp.train()
#         for features, labels in train_loader:
#             #features, labels = features.to(device), labels.to(device)
            
#             # Forward pass
#             outputs = mlp(features)
#             loss = criterion(outputs, labels)
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        
#         # Validation
#         mlp.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for features, labels in val_loader:
#                 features, labels = features.to(device), labels.to(device)
#                 outputs = mlp(features)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += labels.size(0)
#                 correct += predicted.eq(labels).sum().item()
        
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100. * correct / total:.2f}%")

def extract_features(loaded_ae, loader, device="cpu"):
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
            features_reshaped = features.view(features.size(0), -1)

            # Add the features and labels to our lists.
            features_list.append(features_reshaped)
            labels_list.append(labels[1])  # Assuming 'labels[1]' contains the action labels.

    # Concatenate the list of tensors into a single tensor.
    # 'torch.cat' concatenates tensors along a given dimension, here it's along dimension 0.
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    return all_features, all_labels


# OLD
# def extract_features(loaded_ae, loader, device="cpu"):
#     loaded_ae.eval()
#     features_list = []
#     labels_list = []
    
#     with torch.no_grad():
#         for batch in loader:
#             images, labels = batch
            
#             # Check the size of the input
#             print(len(images), len(labels[1]))

#             images = images.float() # Model expects float
#             images = images.squeeze(1)  # Remove the dimension with size 1
#             images = images.permute(0, 3, 1, 2)  # Move the channels dimension to the correct position [batch_size, channels, height, width]
#             features = loaded_ae.encoder(images)

#             # Batch features shape
#             print("Batch features shape:", features.shape)

#             features_list.append(features.reshape(features.size(0), -1)) # What is happening here?
#             #features_list.append(features)

#             print("Batch labels shape:", labels[1].shape)
#             labels_list.append(labels[1]) # 0: Tools, 1: Actions

#     f_list = torch.cat(features_list, dim=0)
#     l_list = torch.cat(labels_list, dim=0)

#     # print("Concat Features List shape:", f_list.shape)
#     # print("Concat Labels List shape:", l_list.shape)

#     return f_list, l_list