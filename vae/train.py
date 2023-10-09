import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from vae.vae import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle

class CustomDataset(Dataset):
    def __init__(self, data_pickle_file_path, labels_pickle_file_path):
        with open(data_pickle_file_path, 'rb') as f:
            self.data = pickle.load(f)
        
        with open(labels_pickle_file_path, 'rb') as f:
            self.labels = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 24576  # 64 * 128 * 3
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 2
BATCH_SIZE = 32
LR_RATE = 3e-4 #Karpathy constant

# Dataset Loading
# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
# Dataset Loading
train_dataset = CustomDataset(data_pickle_file_path='data/training_color.pkl', 
                              labels_pickle_file_path='data/y_training.pkl')
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset = CustomDataset(data_pickle_file_path='data/validation_color.pkl', 
                                   labels_pickle_file_path='data/y_validation.pkl')
validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = CustomDataset(data_pickle_file_path='data/testing_color.pkl', 
                             labels_pickle_file_path='data/y_testing.pkl')
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
# loss_fn = nn.BCELoss(reduction="sum") #Check out Pytorch Documentation on Loss Functions
loss_fn = nn.MSELoss(reduction="sum") #More often used for non-binary images

# Training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        # Forward pass
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)

        # Compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x) #reconstruct the input
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) #towards standard gaussian

        # Backprop
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

model = model.to(DEVICE)
def inference(instance, num_examples=5):
    images = []
    idx = 0
    num_classes = len(set(train_dataset.labels)) # Adjust this to match the number of unique labels in your dataset

    for x, y in train_dataset:  
        if y == idx:
            images.append(x)
            idx += 1
        if idx == num_classes:  # Adjusted to match the number of unique labels
            break

    if len(images) <= instance:
        print(len(images))
        print(instance)
        print(f"No images found for instance {instance}.")
        return
    
    encodings_instance = []
    for d in range(num_classes):
        with torch.no_grad():
            mu, sigma = model.encode(torch.Tensor(images[d]).view(1, INPUT_DIM))
        encodings_instance.append((mu, sigma))

    mu, sigma = encodings_instance[instance]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon #reparameterization
        out = model.decode(z)
        out = out.view(-1, 3, 64, 128) # Adjust the reshaping to match your data dimensions
        save_image(out, f"generated_{instance}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=1)
