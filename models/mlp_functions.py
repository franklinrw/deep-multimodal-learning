import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
    

# def train_mlp(model, loss_function, optimizer, train_loader, num_epochs, device="cuda"):
#     """
#     Train the MLP model and evaluate its performance on a validation set.

#     Parameters:
#     model (nn.Module): The MLP model to train.
#     num_epochs (int): Number of epochs to train the model.
#     train_loader (DataLoader): DataLoader for the training set.
#     val_loader (DataLoader): DataLoader for the validation set.
#     device (str): The device to use for training ('cpu' or 'cuda').

#     Returns:
#     None: The model is trained in place.
#     """
#     model.train() 
#     for epoch in range(num_epochs):
#         running_loss = 0.0
        
#         epoch_labels = []
#         epoch_predictions = []

#         for features, labels in train_loader:
#             features, labels = features.to(device), labels.to(device)

#             optimizer.zero_grad()  # Clear previous gradients.
#             outputs = model(features)  # Forward pass.
#             #print("output: ", outputs)
#             #print("labels: ", labels)
#             loss = loss_function(outputs, labels)  # Compute loss.
#             loss.backward()  # Backward pass.
#             optimizer.step()  # Update weights.
#             running_loss += loss.item()
#             #print("indices:", outputs.max(1).indices.cpu().numpy())
#             epoch_labels.extend(labels.cpu().numpy())
#             epoch_predictions.extend(outputs.max(1).indices.cpu().numpy())

#         # Calculate average loss and accuracy over the epoch.
#         avg_train_loss = running_loss / len(train_loader)

#         # Calculate metrics using scikit-learn.
#         train_accuracy_sklearn = accuracy_score(epoch_labels, epoch_predictions)
#         train_precision = precision_score(epoch_labels, epoch_predictions, average='macro')
#         train_recall = recall_score(epoch_labels, epoch_predictions, average='macro')
#         train_f1 = f1_score(epoch_labels, epoch_predictions, average='macro')

#         print(f"Epoch [{epoch+1}/{num_epochs}], ",
#             f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy_sklearn:.4f}, ",
#             f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, ",
#             f"Train F1: {train_f1:.4f}")
    
#     return model


# def validate_mlp(model, loss_function, val_loader, device):
#     """
#     Evaluate the mlp's performance on the validation set.

#     Parameters:
#     model (nn.Module): The model to evaluate.
#     criterion (nn.Module): The loss function.
#     data_loader (DataLoader): DataLoader for the validation set.
#     device (str): The device where the model and data should be loaded ('cpu' or 'cuda').

#     Returns:
#     float: The average loss over the validation set.
#     float: The accuracy on the validation set.
#     """
#     model.eval()  # Set the model to evaluation mode.
#     total_loss = 0.0
#     total_samples = 0

#     all_val_labels = []
#     all_val_predictions = []

#     with torch.no_grad():  # No need to track gradients for validation.
#         for features, labels in val_loader:
#             features, labels = features.to(device), labels.to(device)
#             outputs = model(features)

#             # print(outputs)
#             # print(outputs.max(1).indices.cpu().numpy())
#             # print(labels)

#             loss = loss_function(outputs, labels)
#             total_loss += loss.item()
#             total_samples += labels.size(0)

#             # Store true labels and predictions for accuracy calculation.
#             all_val_labels.extend(labels.cpu().numpy())
#             all_val_predictions.extend(outputs.max(1).indices.cpu().numpy())

#     avg_loss = total_loss / total_samples

#     # Calculate metrics using scikit-learn.
#     val_accuracy_sklearn = accuracy_score(all_val_labels, all_val_predictions)
#     val_precision = precision_score(all_val_labels, all_val_predictions, average='macro')
#     val_recall = recall_score(all_val_labels, all_val_predictions, average='macro')
#     val_f1 = f1_score(all_val_labels, all_val_predictions, average='macro')
#     val_confusion_matrix = confusion_matrix(all_val_labels, all_val_predictions)

#     print(f"Val Loss: {avg_loss:.4f}, Val Acc: {val_accuracy_sklearn:.4f}, "
#         f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
#         f"Val F1: {val_f1:.4f}")
#     print("Confusion Matrix:\n", val_confusion_matrix)

def train_mlp(model, loss_function, optimizer, train_loader, num_epochs, device="cuda"):
    model.train() 
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_labels = []
        epoch_predictions = []

        for features, labels in train_loader:
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

def validate_mlp(model, loss_function, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_val_labels = []
    all_val_predictions = []

    with torch.no_grad():
        for features, labels in val_loader:
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

