import sys
sys.path.insert(0, '../')

# Import the necessary packages
import torch
import torch.nn as nn
from functions import get_loader, plot_histories, plot_history

from ae_functions import train_autoencoder, validate_cae, visualize_latent_space, visualize_reconstruction, get_latent_dataset
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
batch_sizes = [4, 8, 16, 32]
num_epochs = [3]
lr_rates = [1e-2, 1e-3, 1e-4, 1e-5]
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

                # Train the model
                trained_cae, cae_epoch_loss_history, avg_psnr_history, avg_ssim_history = train_autoencoder(cae,\
                                                                train_loader,\
                                                                cae_lossfunction,\
                                                                optimizer,\
                                                                is_depth=False,\
                                                                num_epochs=NUM_EPOCHS,\
                                                                add_noise=dcae,\
                                                                device=DEVICE)

                # Validate the model
                avg_val_loss, avg_psnr, avg_ssim = validate_cae(trained_cae,\
                                                                test_loader,\
                                                                cae_lossfunction,\
                                                                is_depth = False,\
                                                                device = DEVICE)

                # Save the model weights
                model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights_ae/"
                weight_name = f"simple/simple_cae_ne{NUM_EPOCHS}_b{BATCH_SIZE}_{SENSOR}.pth"
                torch.save(trained_cae.state_dict(), model_path+weight_name)

for sensor_color in sensors:
    # Define the hyperparameters
    SENSOR = sensor_color
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LR_RATE = 1e-3

    # Define the loaders
    train_loader = get_loader(BASE_PATH, OBJECTS, TOOL_NAMES, ACTIONS, sensor_color, "training", batch_size=BATCH_SIZE)
    val_loader = get_loader(BASE_PATH, OBJECTS, TOOL_NAMES, ACTIONS, sensor_color, "validation", batch_size=BATCH_SIZE)
    test_loader = get_loader(BASE_PATH, OBJECTS, TOOL_NAMES, ACTIONS, sensor_color, "testing", batch_size=BATCH_SIZE)

    # cae_lossfunction = nn.MSELoss()
    cae_lossfunction = nn.BCELoss()

    # Define the model
    cae = simpleCAE(input_channels=3).to(DEVICE)

    # Define the optimizer
    optimizer= torch.optim.Adam(cae.parameters(), lr=LR_RATE)
    # optimizer = torch.optim.SGD(cae.parameters(), lr=LR_RATE, momentum=0.9)
    # optimizer = torch.optim.AdamW(cae.parameters(), lr=LR_RATE, weight_decay=1e-2)

    # Train the model
    trained_cae, cae_epoch_loss_history, avg_psnr_history, avg_ssim_history = train_autoencoder(cae,\
                                                            train_loader,\
                                                            cae_lossfunction,\
                                                            optimizer,\
                                                            is_depth=False,\
                                                            num_epochs=NUM_EPOCHS,\
                                                            add_noise=True,\
                                                            device=DEVICE,\
                                                            visualize=False)

    # Validate the model
    avg_val_loss, avg_psnr, avg_ssim = validate_cae(trained_cae,\
                                                            test_loader,\
                                                            cae_lossfunction,\
                                                            is_depth = False,\
                                                            device = DEVICE)

    # Save the model weights
    model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights_ae/"
    weight_name = f"simple/simple_dcae_ne{NUM_EPOCHS}_b{BATCH_SIZE}_{SENSOR}.pth"
    torch.save(trained_cae.state_dict(), model_path+weight_name)