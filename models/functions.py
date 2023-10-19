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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        for batch in loader:
            images, _ = batch  # We don't need labels for autoencoders
            images = images.float() # Model expects float
            images = images.squeeze(1)  # Remove the dimension with size 1
            images = images.permute(0, 3, 1, 2)   # Move the channels dimension to the correct position
            outputs = model(images)
            loss = criterion(outputs, images)  # Reconstruction loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def train_mlp(mlp, num_epochs, train_loader, val_loader, device="cpu"):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        mlp.train()
        for features, labels in train_loader:
            #features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = mlp(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        mlp.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = mlp(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100. * correct / total:.2f}%")

def extract_features(loaded_ae, loader, device="cpu"):
    loaded_ae.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            
            # Check the size of the input
            print(len(images), len(labels[1]))

            images = images.float() # Model expects float
            images = images.squeeze(1)  # Remove the dimension with size 1
            images = images.permute(0, 3, 1, 2)  # Move the channels dimension to the correct position [batch_size, channels, height, width]
            features = loaded_ae.encoder(images)

            # Batch features shape
            print("Batch features shape:", features.shape)

            features_list.append(features.reshape(features.size(0), -1)) # What is happening here?
            #features_list.append(features)

            print("Batch labels shape:", labels[1].shape)
            labels_list.append(labels[1]) # 0: Tools, 1: Actions

    f_list = torch.cat(features_list, dim=0)
    l_list = torch.cat(labels_list, dim=0)

    # print("Concat Features List shape:", f_list.shape)
    # print("Concat Labels List shape:", l_list.shape)

    return f_list, l_list