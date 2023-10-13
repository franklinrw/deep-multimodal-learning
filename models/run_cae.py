import torch
from functions import train_cae, get_color_training_loader, get_dummy_loader
from models import CAE

# Gets the training dataset loader
loader = get_dummy_loader()
#loader = get_color_training_loader()

##### CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 2
BATCH_SIZE = 32
LR_RATE = 1e-3

# Training loop
model = CAE()
train_cae(model, loader, NUM_EPOCHS, LR_RATE)

model_path = "C:/Users/Frank/OneDrive/Bureaublad/ARC/deep-multimodal-learning/weights/dummy.pth"
torch.save(model.state_dict(), model_path)