import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

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
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)  # Ensure output_dim matches the number of classes

    def forward(self, x):
        # Apply layer by layer transformations
        x = F.relu(self.fc1(x))  # ReLU activation for non-linearity
        x = F.relu(self.fc2(x))  # ReLU activation for non-linearity
        x = self.fc3(x)  # The network's output are the class scores

        return x

class dropout_MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(dropout_MLP, self).__init__()

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