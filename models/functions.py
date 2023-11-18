from torch.utils.data import Dataset
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
import torch

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
                path = os.path.join(base_path, objectname, toolname, action, sensor, f"{set_name}.pkl")
                
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


def inspect_loader(loader, description="Loader"):
    """
    Inspect the first batch from a DataLoader.

    :param loader: DataLoader
        DataLoader to inspect.
    :param description: str
        Description of the DataLoader (e.g., 'Color Loader', 'Depth Loader').
    """
    images, labels = next(iter(loader))
    print(f"{description} - First Batch Inspection")
    print("--------------------------------------------------")
    print("Batch Shape:", images.shape)
    print("Data Type:", images.dtype)
    # print("Label Shape:", labels.shape)
    # print("Label Data Type:", labels.dtype)
    print("First Image Shape:", images[0].shape)
    print("Max Pixel Value in First Image:", images[0].max())
    print("Min Pixel Value in First Image:", images[0].min())
    print("--------------------------------------------------\n")


def plot_histories(training_loss_history, validation_loss_history=None):
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss_history, label='Training Batch Loss')
    
    if validation_loss_history is not None:
        plt.plot(validation_loss_history, label='Validation Batch Loss')
    
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Batch')
    plt.legend()
    plt.show()

def plot_history(history, x_label = 'Epoch', y_label='Loss', title='Loss per Epoch'):
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
