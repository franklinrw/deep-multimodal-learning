import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_mlp(model, loss_function, optimizer, train_loader, num_epochs, label=1, device="cuda"):
    model.train() 
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_labels = []
        epoch_predictions = []

        for features, labels in train_loader:
            if(type(labels) is list):
                labels = labels[label]
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            epoch_labels.extend(labels.cpu().numpy())
            epoch_predictions.extend(outputs.max(1).indices.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        train_accuracy_sklearn = accuracy_score(epoch_labels, epoch_predictions)
        print(f"Training Accuracy: {train_accuracy_sklearn:.4f}")

        # Print classification report
        print("Training Classification Report:")
        print(classification_report(epoch_labels, epoch_predictions))

    return model


def validate_mlp(model, loss_function, val_loader, label=1, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_val_labels = []
    all_val_predictions = []

    with torch.no_grad():
        for features, labels in val_loader:
            if(type(labels) is list):
                labels = labels[label]
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

            all_val_labels.extend(labels.cpu().numpy())
            all_val_predictions.extend(outputs.max(1).indices.cpu().numpy())

    avg_loss = total_loss / total_samples
    print(f"Validation Loss: {avg_loss:.4f}")

    val_accuracy_sklearn = accuracy_score(all_val_labels, all_val_predictions)
    print(f"Validation Accuracy: {val_accuracy_sklearn:.4f}")

    # Print classification report
    print("Validation Classification Report:")
    print(classification_report(all_val_labels, all_val_predictions))


def decision_level_fusion(models, data_loader, device='cuda'):
    """
    Perform decision-level fusion by averaging the outputs of multiple models and calculate the accuracy.

    Parameters:
    models (list): List of trained models.
    data_loader (DataLoader): DataLoader for the test/validation set.
    device (str): Computation device ('cuda' or 'cpu').

    Returns:
    float: The accuracy of the fused model predictions.
    """
    # Ensure models are in evaluation mode
    for model in models:
        model.eval()

    all_labels = []
    all_fused_predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = [model(inputs).cpu() for model in models]

            # Average the predictions
            avg_output = torch.mean(torch.stack(outputs), dim=0)

            # Convert averaged outputs to predicted class indices
            _, predicted_classes = torch.max(avg_output, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_fused_predictions.extend(predicted_classes.numpy())

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(all_labels, all_fused_predictions)
    print(f"Fused Model Accuracy: {accuracy:.4f}")

    print("Fused Model Classification Report:")
    print(classification_report(all_labels, all_fused_predictions))

    return accuracy