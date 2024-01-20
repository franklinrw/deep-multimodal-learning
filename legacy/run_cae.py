import torch
from functions import get_loader, visualize_reconstruction
from functions_cae import train_cae, validate_cae, CAE, SimpleCAE

# Check if CUDA is available
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU...")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU...")

##### CONFIG
NUM_EPOCHS = 1
LR_RATE = 1e-3
BATCH_SIZE = 8

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

# Training loop
#model = CAE().to(device)
model = SimpleCAE().to(device)
train_cae(model, train_loader, NUM_EPOCHS, LR_RATE, device)
validate_cae(model, val_loader, device)

model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights/simple-cae.pth"
torch.save(model.state_dict(), model_path)