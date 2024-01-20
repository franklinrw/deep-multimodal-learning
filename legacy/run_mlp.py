import torch
from functions import get_loaders, extract_features
from functions_mlp import MLP, train_mlp, validate_mlp
from functions_cae import CAE
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Check if CUDA is available
DEVICE = ""
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA is available. Using GPU...")
else:
    DEVICE = torch.device("cpu")
    print("CUDA is not available. Using CPU...")

BATCH_SIZE = 8
LF = nn.CrossEntropyLoss()
NUM_EPOCHS = 1

# Assuming you've defined your CAE class as 'CAE'
#model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights/dummy.pth"
#model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights/color-b32-e5.pth"
model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights/color-b8-e2.pth"
loaded_cae = CAE()
loaded_cae.load_state_dict(torch.load(model_path))

train_loader, val_loader, _ = get_loaders("color", BATCH_SIZE)

# Extract features from the train and validation sets
train_features, train_labels = extract_features(loaded_cae, train_loader)
val_features, val_labels = extract_features(loaded_cae, val_loader)

print("Train features shape:", train_features.shape)
print("Train labels shape:",train_labels.shape)
print("Val features shape:", val_features.shape)
print("Val labels shape:",val_labels.shape)

# Convert extracted features to TensorDatasets
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)

# Create DataLoaders for the extracted features
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the MLP
input_dim = train_features.size(1)
output_dim = 4 # It will always be 4 and or otherwise 16
print("Input Dim:", input_dim)
print("Output Dim:", output_dim)

mlp_model = MLP(input_dim, output_dim).to(DEVICE)

mlp_model = train_mlp(mlp_model, LF, NUM_EPOCHS, train_loader, DEVICE)
validate_mlp(mlp_model, LF, val_loader, DEVICE)
