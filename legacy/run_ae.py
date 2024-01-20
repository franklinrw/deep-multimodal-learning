import torch
from functions import get_loader
from ae_functions import train_autoencoder, validate_cae, visualize_latent_space, visualize_reconstruction, get_latent_dataset
from ae_models import DeepCAE, DeepBatchCAE, SimpleCAE, SimpleBatchCAE, SimpleCAE_Dropout, SimpleBatchDropoutCAE
import torch.nn as nn

##### CONFIG
NUM_EPOCHS = 5
LR_RATE = 1e-3
BATCH_SIZE = 4
WEIGHT_DECAY = 1e-5

DEVICE = ""
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA is available. Using GPU...")
else:
    DEVICE = torch.device("cpu")
    print("CUDA is not available. Using CPU...")

BASE_PATH = 'C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/data'

# Define the tool names and actions
TOOL_NAMES = ['hook', 'ruler', 'spatula', 'sshot']
ACTIONS = ['left_to_right', 'pull', 'push', 'right_to_left']

# All available object names
OBJECTS = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
            '5_blackCoinbag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg','9_pumpkinToy',
            '10_tomatoCan', '11_boxMilk', '12_containerNuts', '13_cornCob', '14_yellowFruitToy',
            '15_bottleNailPolisher', '16_boxRealSense', '17_clampOrange', '18_greenRectangleToy', '19_ketchupToy']

SENSOR = "color"

train_loader = get_loader(BASE_PATH, OBJECTS, TOOL_NAMES, ACTIONS, SENSOR, "training", batch_size=BATCH_SIZE)
val_loader = get_loader(BASE_PATH, OBJECTS, TOOL_NAMES, ACTIONS, SENSOR, "validation", batch_size=BATCH_SIZE)
test_loader = get_loader(BASE_PATH, OBJECTS, TOOL_NAMES, ACTIONS, SENSOR, "testing", batch_size=BATCH_SIZE)

cae_lossfunction = nn.BCELoss()

cae = SimpleBatchCAE(input_channels=3).to(DEVICE)

optimizer= torch.optim.Adam(cae.parameters(), lr=LR_RATE)

trained_cae, cae_epoch_loss_history = train_autoencoder(cae,\
                                                        train_loader,\
                                                        cae_lossfunction,\
                                                        optimizer,\
                                                        is_depth=False,\
                                                        num_epochs=NUM_EPOCHS,\
                                                        device=DEVICE,\
                                                        visualize=False)

model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights_ae/"
weight_name = "batchnorm/batch_cae_ne5_b4_color.pth"
torch.save(trained_cae.state_dict(), model_path+weight_name)