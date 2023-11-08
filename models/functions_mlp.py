import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    
class rawMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(rawMLP, self).__init__()

        # It's good practice to define each layer separately in case you need individual layer references
        # or apply different specific parameters (like different dropout values, batch normalization, etc.)
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)  # Ensure output_dim matches the number of classes

        self.dropout = nn.Dropout(0.5)  # Can adjust the dropout rate if needed

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.float()
        # Apply layer by layer transformations
        x = F.relu(self.fc1(x))  # ReLU activation for non-linearity
        x = self.dropout(x)  # Dropout for regularization

        x = F.relu(self.fc2(x))  # ReLU activation for non-linearity
        x = self.dropout(x)  # Dropout for regularization

        # No activation is used before the output layer with CrossEntropyLoss
        x = self.fc3(x)  # The network's output are the class scores

        return x

class caeMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(caeMLP, self).__init__()

        # It's good practice to define each layer separately in case you need individual layer references
        # or apply different specific parameters (like different dropout values, batch normalization, etc.)
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)  # Ensure output_dim matches the number of classes

        self.dropout = nn.Dropout(0.5)  # Can adjust the dropout rate if needed

    def forward(self, x):
        # Apply layer by layer transformations
        x = F.relu(self.fc1(x))  # ReLU activation for non-linearity
        x = self.dropout(x)  # Dropout for regularization

        x = F.relu(self.fc2(x))  # ReLU activation for non-linearity
        x = self.dropout(x)  # Dropout for regularization

        # No activation is used before the output layer with CrossEntropyLoss
        x = self.fc3(x)  # The network's output are the class scores

        return x
    
def train_mlp(model, loss_function, optimizer, train_loader, num_epochs, device="cuda"):
    """
    Train the MLP model and evaluate its performance on a validation set.

    Parameters:
    model (nn.Module): The MLP model to train.
    num_epochs (int): Number of epochs to train the model.
    train_loader (DataLoader): DataLoader for the training set.
    val_loader (DataLoader): DataLoader for the validation set.
    device (str): The device to use for training ('cpu' or 'cuda').

    Returns:
    None: The model is trained in place.
    """

    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        
        all_labels = []
        all_predictions = []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear previous gradients.
            outputs = model(features)  # Forward pass.
            loss = loss_function(outputs, labels)  # Compute loss.
            loss.backward()  # Backward pass.
            optimizer.step()  # Update weights.
            running_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.max(1).indices.cpu().numpy())

        # Calculate average loss and accuracy over the epoch.
        avg_train_loss = running_loss / len(train_loader)

        # Calculate metrics using scikit-learn.
        train_accuracy_sklearn = accuracy_score(all_labels, all_predictions)
        train_precision = precision_score(all_labels, all_predictions, average='macro')
        train_recall = recall_score(all_labels, all_predictions, average='macro')
        train_f1 = f1_score(all_labels, all_predictions, average='macro')

        print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy_sklearn:.4f}, "
                f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, "
                f"Train F1: {train_f1:.4f}")
        
    return model

def validate_mlp(model, loss_function, val_loader, device):
    """
    Evaluate the mlp's performance on the validation set.

    Parameters:
    model (nn.Module): The model to evaluate.
    criterion (nn.Module): The loss function.
    data_loader (DataLoader): DataLoader for the validation set.
    device (str): The device where the model and data should be loaded ('cpu' or 'cuda').

    Returns:
    float: The average loss over the validation set.
    float: The accuracy on the validation set.
    """
    model.eval()  # Set the model to evaluation mode.
    total_loss = 0.0
    total_samples = 0

    all_val_labels = []
    all_val_predictions = []

    with torch.no_grad():  # No need to track gradients for validation.
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

            # Store true labels and predictions for accuracy calculation.
            all_val_labels.extend(labels.cpu().numpy())
            all_val_predictions.extend(outputs.max(1).indices.cpu().numpy())

    avg_loss = total_loss / total_samples

    # Calculate metrics using scikit-learn.
    val_accuracy_sklearn = accuracy_score(all_val_labels, all_val_predictions)
    val_precision = precision_score(all_val_labels, all_val_predictions, average='macro')
    val_recall = recall_score(all_val_labels, all_val_predictions, average='macro')
    val_f1 = f1_score(all_val_labels, all_val_predictions, average='macro')
    val_confusion_matrix = confusion_matrix(all_val_labels, all_val_predictions)

    print(f"Val Loss: {avg_loss:.4f}, Val Acc: {val_accuracy_sklearn:.4f}, "
        f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
        f"Val F1: {val_f1:.4f}")
    print("Confusion Matrix:\n", val_confusion_matrix)