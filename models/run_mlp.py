import torch
from functions import get_loader, extract_features
from functions_mlp import rawMLP, train_mlp, validate_mlp, MLP
from functions_cae import CAE, SimpleCAE
import torch.nn as nn

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{DEVICE} is available. Using {DEVICE}...")

BATCH_SIZE = 8
NUM_EPOCHS = 1
LF = nn.CrossEntropyLoss()  # Loss function
output_dim = 4 

# Flag to switch between using CAE features or raw data
USE_CAE_FEATURES = True  # Set to True if you want to use CAE features

BASE_PATH = 'C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/data_v2'

# Define the tool names and actions
TOOL_NAMES = ['hook', 'ruler', 'spatula', 'sshot']
ACTIONS = ['left_to_right', 'pull', 'push', 'right_to_left']

# All available object names
train_objects = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
            '5_blackCoinbag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg','9_pumpkinToy',
            '10_tomatoCan', '11_boxMilk']

val_objects = ['12_containerNuts', '13_cornCob', '14_yellowFruitToy',
            '15_bottleNailPolisher']

train_loader = get_loader(BASE_PATH, train_objects, TOOL_NAMES, ACTIONS, "color", "training", batch_size=8)
val_loader = get_loader(BASE_PATH, val_objects, TOOL_NAMES, ACTIONS, "color", "validation", batch_size=8)

if USE_CAE_FEATURES:
    # Load CAE model and extract features
    model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights/cae.pth"
    loaded_cae = CAE()
    loaded_cae.load_state_dict(torch.load(model_path))

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
    mlp_model = MLP(input_dim, output_dim).to(DEVICE)
else:
    # Use raw data
    # Here, you need to specify the correct input dimension of your raw data
    # images, labels = next(iter(train_loader))
    # images = images.float() # Model expects float
    # images = images.squeeze(1)  # Remove the dimension with size 1
    # images = images.permute(0, 3, 1, 2)   # Move the channels dimension to the correct position
    # input_dim = images.size(1) * images.size(2) * images.size(3)  # Please replace with the correct input dimension for raw data
    input_dim = 294912
    mlp_model = rawMLP(input_dim, output_dim).to(DEVICE)

print("Input Dim:", input_dim)
print("Output Dim:", output_dim)

# Train the model
train_mlp(mlp_model, LF, NUM_EPOCHS, train_loader, DEVICE)

# Validate the model
validate_mlp(mlp_model, LF, val_loader, DEVICE)
