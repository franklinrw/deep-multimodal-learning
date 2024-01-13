import sys
import os
sys.path.insert(0, '../')

# Import the necessary packages
import torch
import torch.nn as nn
from functions import get_loader

from ae_functions import train_autoencoder, validate_cae, visualize_reconstruction, visualize_latent_space
from ae_models import simpleCAE

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
batch_sizes = [8]
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

                # cae_lossfunction = nn.MSELoss()
                cae_lossfunction = nn.BCELoss()

                # Define the model
                cae = simpleCAE(input_channels=3).to(DEVICE)

                # Define the optimizer
                optimizer= torch.optim.Adam(cae.parameters(), lr=LR_RATE)

                save_path = f"C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/models/runs/results/simple/{SENSOR}_B{BATCH_SIZE}_NE{NUM_EPOCHS}_LR{LR_RATE}/"
                os.makedirs(save_path, exist_ok=True)
                # Train the model
                trained_cae, cae_epoch_loss_history, avg_psnr_history, avg_ssim_history = train_autoencoder(cae,\
                                                                train_loader,\
                                                                cae_lossfunction,\
                                                                optimizer,\
                                                                is_depth=False,\
                                                                num_epochs=NUM_EPOCHS,\
                                                                add_noise=dcae,\
                                                                device=DEVICE,\
                                                                save_dir=save_path)

                # Validate the model
                avg_val_loss, avg_psnr, avg_ssim = validate_cae(trained_cae,\
                                                                test_loader,\
                                                                cae_lossfunction,\
                                                                is_depth = False,\
                                                                device = DEVICE,\
                                                                save_dir=save_path)

                # Save the model weights
                model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights_ae/"
                weight_name = f"simple/simple_cae_ne{NUM_EPOCHS}_b{BATCH_SIZE}_{SENSOR}.pth"
                torch.save(trained_cae.state_dict(), model_path+weight_name)

                save_path_recon = os.path.join(save_path, f"test_recon.png")
                visualize_reconstruction(trained_cae, test_loader, num_samples=2, save_dir=save_path_recon)
                save_path_latent_train = os.path.join(save_path, f"train_latent.png")
                visualize_latent_space(trained_cae, train_loader, n_components=2, save_dir=save_path_latent_train)
                save_path_latent_test = os.path.join(save_path, f"test_latent.png")
                visualize_latent_space(trained_cae, test_loader, n_components=2, save_dir=save_path_latent_test)