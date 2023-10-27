import torch
from functions import get_loaders, train_mlp, validate_mlp, extract_features
from models import MLP, CAE
import torch.nn as nn

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{DEVICE} is available. Using {DEVICE}...")

BATCH_SIZE = 8
NUM_EPOCHS = 1
LF = nn.CrossEntropyLoss()  # Loss function

# Flag to switch between using CAE features or raw data
USE_CAE_FEATURES = False  # Set to True if you want to use CAE features

if USE_CAE_FEATURES:
    # Load CAE model and extract features
    model_path = "path_to_your_CAE_model"  # please set the correct path
    loaded_cae = CAE()
    loaded_cae.load_state_dict(torch.load(model_path))

    train_loader, val_loader, _ = get_loaders("color", BATCH_SIZE)

    # Extract features from the train and validation sets
    train_features, train_labels = extract_features(loaded_cae, train_loader)
    val_features, val_labels = extract_features(loaded_cae, val_loader)

    # Convert extracted features to TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)

    # Create DataLoaders for the extracted features
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_features.size(1)  # input dimension from CAE features
else:
    # Use raw data
    train_loader, val_loader, _ = get_loaders("color", BATCH_SIZE)
    # Here, you need to specify the correct input dimension of your raw data
    # images, labels = next(iter(train_loader))
    # images = images.float() # Model expects float
    # images = images.squeeze(1)  # Remove the dimension with size 1
    # images = images.permute(0, 3, 1, 2)   # Move the channels dimension to the correct position
    # input_dim = images.size(1) * images.size(2) * images.size(3)  # Please replace with the correct input dimension for raw data
    input_dim = 294912

output_dim = 4  # It will always be 4 and or otherwise 16
print("Input Dim:", input_dim)
print("Output Dim:", output_dim)

mlp_model = MLP(input_dim, output_dim).to(DEVICE)

# Train the model
train_mlp(mlp_model, LF, NUM_EPOCHS, train_loader, DEVICE)

# Validate the model
validate_mlp(mlp_model, LF, val_loader, DEVICE)
