import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

# class rawMLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(rawMLP, self).__init__()

#         # It's good practice to define each layer separately in case you need individual layer references
#         # or apply different specific parameters (like different dropout values, batch normalization, etc.)
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, output_dim)  # Ensure output_dim matches the number of classes

#         self.dropout = nn.Dropout(0.5)  # Can adjust the dropout rate if needed

#     def forward(self, x):
#         if x.dim() > 2:
#             x = x.view(x.size(0), -1)
#         x = x.float()
#         # Apply layer by layer transformations
#         x = F.relu(self.fc1(x))  # ReLU activation for non-linearity
#         x = self.dropout(x)  # Dropout for regularization

#         x = F.relu(self.fc2(x))  # ReLU activation for non-linearity
#         x = self.dropout(x)  # Dropout for regularization

#         # No activation is used before the output layer with CrossEntropyLoss
#         x = self.fc3(x)  # The network's output are the class scores

#         return x
    
class simpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(simpleMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)  # Ensure output_dim matches the number of classes

    def forward(self, x):
        # Apply layer by layer transformations
        x = F.relu(self.fc1(x))  # ReLU activation for non-linearity
        x = F.relu(self.fc2(x)) 
        x = self.fc3(x)  

        return x
    
class improvedMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(improvedMLP, self).__init__()

        # Increasing the number of layers and introducing dropout for regularization
        self.fc1 = nn.Linear(input_dim, 1024)  # First layer with more neurons
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)  # Dropout layer for regularization

        self.fc2 = nn.Linear(1024, 512)  # Additional intermediate layer
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)

        self.fc3 = nn.Linear(512, 256)  # Further reduction in dimension
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.25)

        self.fc4 = nn.Linear(256, output_dim)  # Final layer for class scores

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)  # No activation function like softmax required if using CrossEntropyLoss
        return x