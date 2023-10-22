import torch
from functions import train_cae, get_color_loader, get_dummy_loader
from models import CAE

# Gets the training dataset loader
# loader = get_dummy_loader()
loader = get_color_loader()

# Check if CUDA is available
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU...")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU...")

##### CONFIG
NUM_EPOCHS = 5
BATCH_SIZE = 32
LR_RATE = 1e-3

# Training loop
model = CAE().to(device)
train_cae(model, loader, NUM_EPOCHS, LR_RATE, device)

model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights/color-b32-e5.pth"
torch.save(model.state_dict(), model_path)