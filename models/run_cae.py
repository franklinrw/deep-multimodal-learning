import torch
from functions import train_cae, get_loaders, get_dummy_loader, validate_cae
from models import CAE, SimpleCAE

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
LR_RATE = 1e-5
BATCH_SIZE = 8

# Gets the training dataset loader
# loader = get_dummy_loader()
train_loader, val_loader, _ = get_loaders("color", BATCH_SIZE)

# Training loop
# model = CAE().to(device)
model = SimpleCAE().to(device)
train_cae(model, train_loader, NUM_EPOCHS, LR_RATE, device)
validate_cae(model, val_loader, device)

model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights/simple-color-b8-e2.pth"
torch.save(model.state_dict(), model_path)