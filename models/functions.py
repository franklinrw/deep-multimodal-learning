from torch.utils.data import Dataset
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
import torch
from sklearn.metrics import classification_report
from torch.utils.data import Sampler

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
    

class CustomSampler(Sampler):
    def __init__(self, data_source, indices):
        self.data_source = data_source
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    
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

def get_loader(base_path, objectnames, toolnames, actions, sensor, set_name, shuffle=False, batch_size=8):

    dataset = get_datasets(base_path, objectnames, toolnames, actions, sensor, set_name)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def load_pretrained_cae(model_class, model_path, weight_name, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path + weight_name))
    model.eval()
    return model

def load_pretrained_mlp(model_class, model_path, weight_name, device, input_dim, output_dim):
    model = model_class(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path + weight_name))
    model.eval()
    return model


def get_model_predictions(model, dataloader, device):
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs)
    return predictions


def average_fusion_predictions(model_predictions):
    final_predictions = []
    num_models = len(model_predictions)
    num_batches = len(model_predictions[0])

    for i in range(num_batches):
        avg_prediction = sum(model_predictions[j][i] for j in range(num_models)) / num_models
        final_predictions.append(avg_prediction)

    return final_predictions


def calculate_fusion_accuracy(predicted_classes, true_labels):
    flattened_true_labels = [label for sublist in true_labels for label in sublist]
    flattened_predicted_classes = [pred for sublist in predicted_classes for pred in sublist]

    assert len(flattened_true_labels) == len(flattened_predicted_classes), "Mismatch in predictions and labels"
    
    correct_predictions = sum(pred == true for pred, true in zip(flattened_predicted_classes, flattened_true_labels))
    total_predictions = len(flattened_true_labels)
    return correct_predictions / total_predictions


def calculate_classification_report(predicted_classes, true_labels, class_names):
    flattened_true_labels = [label for sublist in true_labels for label in sublist]
    flattened_predicted_classes = [pred for sublist in predicted_classes for pred in sublist]

    assert len(flattened_true_labels) == len(flattened_predicted_classes), "Mismatch in predictions and labels"
    
    report = classification_report(flattened_true_labels, flattened_predicted_classes, target_names=class_names)
    print(report)


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
