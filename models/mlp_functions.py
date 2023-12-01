import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt

def train_mlp(model, loss_function, optimizer, train_loader, num_epochs, label=1, device="cuda", save_dir=None):
    model.train() 
    losses = []
    accuracies = []
    classification_reports = []
    confusion_matrices = []

    class_names = ['left_to_right', 'pull', 'push', 'right_to_left'] # Define the class names for the confusion matrix

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
        losses.append(avg_train_loss)

        train_accuracy_sklearn = accuracy_score(epoch_labels, epoch_predictions)
        accuracies.append(train_accuracy_sklearn)

        classification_report_str = classification_report(epoch_labels, epoch_predictions)
        classification_reports.append(classification_report_str)

        confusion_matrix_arr = confusion_matrix(epoch_labels, epoch_predictions)
        confusion_matrices.append(confusion_matrix_arr)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy_sklearn:.4f}")
        print("Training Classification Report:")
        print(classification_report_str)

    # Save the results to CSV files
    if save_dir is not None:
        np.savetxt(os.path.join(save_dir, "losses.csv"), losses, delimiter=",")
        np.savetxt(os.path.join(save_dir, "accuracies.csv"), accuracies, delimiter=",", fmt='%.4f')
        with open(os.path.join(save_dir, "classification_reports.csv"), "w") as f:
            f.write('\n\n'.join(classification_reports))
        
        # Save the confusion matrices as images
        confusion_matrices_dir = os.path.join(save_dir, "confusion_matrices")
        if not os.path.exists(confusion_matrices_dir):
            os.makedirs(confusion_matrices_dir)

        for i, matrix in enumerate(confusion_matrices):
            plt.figure()
            plt.imshow(matrix, cmap='Blues')
            plt.colorbar()
            plt.xticks(range(len(class_names)), class_names, rotation=45)  # Set x-axis labels
            plt.yticks(range(len(class_names)), class_names)  # Set y-axis labels
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Epoch {i+1}')
            plt.savefig(os.path.join(confusion_matrices_dir, f'confusion_matrix_epoch_{i+1}.png'))
            plt.close()

    return model


def validate_mlp(model, loss_function, val_loader, label=1, device="cuda", save_dir=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_val_labels = []
    all_val_predictions = []

    class_names = ['left_to_right', 'pull', 'push', 'right_to_left'] # Define the class names for the confusion matrix

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
    classification_report_str = classification_report(all_val_labels, all_val_predictions)
    print("Validation Classification Report:")
    print(classification_report_str)

    # Save the results to CSV files
    if save_dir is not None:
        np.savetxt(os.path.join(save_dir, "val_loss.csv"), [avg_loss], delimiter=",", fmt='%.4f')
        np.savetxt(os.path.join(save_dir, "val_accuracy.csv"), [val_accuracy_sklearn], delimiter=",", fmt='%.4f')
        with open(os.path.join(save_dir, "val_classification_report.txt"), "w") as f:
            f.write(classification_report_str)

        # Save the confusion matrix as an image
        confusion_matrix_arr = confusion_matrix(all_val_labels, all_val_predictions)
        plt.figure()
        plt.imshow(confusion_matrix_arr, cmap='Blues')
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=45)  # Set x-axis labels
        plt.yticks(range(len(class_names)), class_names)  # Set y-axis labels
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Validation Confusion Matrix')
        plt.savefig(os.path.join(save_dir, "val_confusion_matrix.png"))
        plt.close()



def decision_level_fusion(models, data_loader, device='cuda', save_dir=None):
    """
    Perform decision-level fusion by averaging the outputs of multiple models and calculate the accuracy.

    Parameters:
    models (list): List of trained models.
    data_loader (DataLoader): DataLoader for the test/validation set.
    device (str): Computation device ('cuda' or 'cpu').
    save_dir (str): Directory to save the results.

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

    classification_report_str = classification_report(all_labels, all_fused_predictions)
    print("Fused Model Classification Report:")
    print(classification_report_str)

    # Save the results to CSV files
    if save_dir is not None:
        np.savetxt(os.path.join(save_dir, "fused_accuracy.csv"), [accuracy], delimiter=",", fmt='%.4f')
        with open(os.path.join(save_dir, "fused_classification_report.txt"), "w") as f:
            f.write(classification_report_str)

        # Save the confusion matrix as an image
        confusion_matrix_arr = confusion_matrix(all_labels, all_fused_predictions)
        plt.figure()
        plt.imshow(confusion_matrix_arr, cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Fused Model Confusion Matrix')
        plt.savefig(os.path.join(save_dir, "fused_confusion_matrix.png"))
        plt.close()