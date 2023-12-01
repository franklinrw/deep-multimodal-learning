import sys
import os
sys.path.insert(0, '../')

import torch
import torch.nn as nn
from mlp_functions import train_mlp, validate_mlp
from mlp_models import improvedMLP, simpleMLP
from functions import get_loader
from ae_models import simpleCAE, improvedCAE

from ae_functions import get_latent_dataset

# Check if CUDA is available
DEVICE = ""
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA is available. Using GPU...")
else:
    DEVICE = torch.device("cpu")
    print("CUDA is not available. Using CPU...")

# Define the base path
BASE_PATH = 'C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/fusion'

# Define the tool names and actions
TOOL_NAMES = ['hook', 'ruler', 'spatula', 'sshot']
ACTIONS = ['left_to_right', 'pull', 'push', 'right_to_left']

# All available object names
OBJECTS = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
            '5_blackCoinbag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg','9_pumpkinToy',
            '10_tomatoCan', '11_boxMilk', '12_containerNuts', '13_cornCob', '14_yellowFruitToy',
            '15_bottleNailPolisher', '16_boxRealSense', '17_clampOrange', '18_greenRectangleToy', '19_ketchupToy']

# Define the sensor type
sensor_color = "color"
sensor_left = "icub_left"
sensor_right = "icub_right"
sensor_depth = "depthcolormap"

sensors = [sensor_color, sensor_left, sensor_right, sensor_depth]
batch_sizes = [4, 8, 16]
num_epochs = [3]
lr_rates = [1e-3]
dcae = False

for sensor in sensors:
    for batch_size in batch_sizes:
        for num_epoch in num_epochs:
            for lr_rate in lr_rates:
                # Define the hyperparameters
                SENSOR = sensor
                BATCH_SIZE = batch_size
                NUM_EPOCHS = num_epoch
                LR_RATE = lr_rate

                # Define the loaders
                train_loader = get_loader(BASE_PATH, OBJECTS, TOOL_NAMES, ACTIONS, SENSOR, "training", batch_size=BATCH_SIZE)
                val_loader = get_loader(BASE_PATH, OBJECTS, TOOL_NAMES, ACTIONS, SENSOR, "validation", batch_size=BATCH_SIZE)
                test_loader = get_loader(BASE_PATH, OBJECTS, TOOL_NAMES, ACTIONS, SENSOR, "testing", batch_size=BATCH_SIZE)

                model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights_ae/"
                cae_name = f"improved/improved_cae_ne3_b{batch_size}_{sensor}.pth"
                trained_cae = improvedCAE().to(DEVICE)
                trained_cae.load_state_dict(torch.load(model_path+cae_name))

                # Config MLP
                mlp_lossfunction = nn.CrossEntropyLoss()  # Loss function
                output_dim = 4 

                # Extract features from the train and validation sets
                train_dataset = get_latent_dataset(trained_cae, train_loader, label=1, add_noise=False, is_depth=False, device=DEVICE)
                val_dataset = get_latent_dataset(trained_cae, val_loader, label=1, add_noise=False, is_depth=False, device=DEVICE)

                # Create DataLoaders for the extracted features
                mlp_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                mlp_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

                # Initialize
                input_dim = train_dataset[:][0].size(1)
                mlp = improvedMLP(input_dim, output_dim).to(DEVICE)
                mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=LR_RATE)

                save_path = f"C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/models/runs/results/improvedCAEimprovedMLP/{SENSOR}_B{BATCH_SIZE}_NE{NUM_EPOCHS}_LR{LR_RATE}/"
                os.makedirs(save_path, exist_ok=True)

                # Train the model
                trained_mlp = train_mlp(mlp, mlp_lossfunction, mlp_optimizer, mlp_train_loader, NUM_EPOCHS, DEVICE, save_dir=save_path)

                # Validate the model
                validate_mlp(trained_mlp, mlp_lossfunction, mlp_val_loader, DEVICE, save_dir=save_path)

                model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights_mlp/"
                weight_name = f"simple/improved_mlp_improved_cae_ne3_b{batch_size}_{sensor}_action.pth"
                torch.save(trained_mlp.state_dict(), model_path+weight_name)