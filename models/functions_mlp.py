import torch.nn.functional as F
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

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
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # No need to track gradients for validation.
        for features, labels in val_loader:
            if isinstance(labels, list):
                labels = labels[1]
            features, labels = features.to(device), labels.to(device)
            print(labels)
            outputs = model(features)

            loss = loss_function(outputs, labels)
            total_loss += loss.item()

            total_correct += (outputs.max(1).indices == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = (total_correct / total_samples) * 100.0

    print(f"Val Loss: {avg_loss:.4f}, Val Acc: {accuracy:.2f}")

def train_mlp(model, loss_function, num_epochs, train_loader, device="cuda"):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode.
        running_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for features, labels in train_loader:
            labels = labels[1]
                
            features, labels = features.to(device), labels.to(device)

            print(labels)

            optimizer.zero_grad()  # Clear previous gradients.
            outputs = model(features)  # Forward pass.
            loss = loss_function(outputs, labels)  # Compute loss.
            loss.backward()  # Backward pass.
            optimizer.step()  # Update weights.
            running_loss += loss.item()

            total_train_correct += (outputs.max(1).indices == labels).sum().item()
            total_train_samples += labels.size(0)

        # Calculate average loss and accuracy over the epoch.
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = total_train_correct / total_train_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}")
        
    return model