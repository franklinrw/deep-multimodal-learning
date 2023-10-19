import torch
from functions import extract_features, get_dummy_loader, train_mlp
from models import CAE, MLP
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 2
DEVICE = "cpu"

# Assuming you've defined your CAE class as 'CAE'
model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights/dummy.pth"
loaded_cae = CAE()
loaded_cae.load_state_dict(torch.load(model_path))

train_loader = get_dummy_loader()
val_loader = get_dummy_loader()

# Extract features from the train and validation sets
train_features, train_labels = extract_features(loaded_cae, train_loader)
val_features, val_labels = extract_features(loaded_cae, val_loader)

print("Train features shape:", train_features.shape)
print("Train labels shape:",train_labels.shape)

# Convert extracted features to TensorDatasets
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)

# Create DataLoaders for the extracted features
train_feature_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_feature_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the MLP
input_dim = train_features.size(1)
output_dim = len(torch.unique(train_labels))
mlp_model = MLP(input_dim, output_dim).to(DEVICE)

print("Input Dim:", input_dim)
print("Output Dim:", output_dim)

train_mlp(mlp_model, 1, train_feature_loader, val_feature_loader, device="cpu")
