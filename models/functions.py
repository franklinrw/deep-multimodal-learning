from torch.utils.data import Dataset
import pickle
import os
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, base_path, objectname, toolname, action, sensor, set_name):
        # Construct the full path based on the provided parameters
        data_file_path = os.path.join(base_path, objectname, toolname, action, sensor, f"{set_name}.pkl")
        labels_file_path = os.path.join(base_path, objectname, toolname, action, sensor, f"y_{set_name}.pkl")
        
        with open(data_file_path, 'rb') as f:
            self.data = pickle.load(f)
        
        with open(labels_file_path, 'rb') as f:
            self.labels = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    
def get_names():
    sensornames = ['color', 'depthcolormap', 'icub_left', 'icub_right']
    toolnames = ['hook', 'ruler', 'spatula', 'sshot']
    actions = ['left_to_right', 'pull', 'push', 'right_to_left']
    objectnames = ['0_woodenCube', '1_pearToy', '2_yogurtYellowbottle', '3_cowToy', '4_tennisBallYellowGreen',
                '5_blackCoinbag', '6_lemonSodaCan', '7_peperoneGreenToy', '8_boxEgg','9_pumpkinToy',
                '10_tomatoCan', '11_boxMilk', '12_containerNuts', '13_cornCob', '14_yellowFruitToy',
                '15_bottleNailPolisher', '16_boxRealSense', '17_clampOrange', '18_greenRectangleToy', '19_ketchupToy']
    
    return sensornames, toolnames, actions, objectnames
    
def get_datasets_for_combinations(base_path, objectnames, toolnames, actions, sensor, set_names):
    datasets = []
    for objectname in objectnames:
        for toolname in toolnames:
            for action in actions:
                for set in set_names:
                    dataset = CustomDataset(base_path=base_path, objectname=objectname, toolname=toolname, action=action, sensor=sensor, set_name=set)
                    datasets.append(dataset)
    return datasets

def Inference(instance, num_examples=5):
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

def visualize_reconstruction(model, test_loader, num_samples=5):
    """
    Visualize the original and reconstructed images from the test set.
    
    Parameters:
    - model: The trained CAE model.
    - test_loader: DataLoader for the test set.
    - num_samples: Number of samples to visualize.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get a batch of test data
        images, _ = next(iter(test_loader))
        
        # Move images to the device
        images = images.float() # Model expects float
        images = images.squeeze(1)  # Remove the dimension with size 1
        images = images.permute(0, 3, 1, 2)   # Move the channels dimension to the correct position
        
        # Get the model's reconstructions
        reconstructions = model(images)
        
        # Move images and reconstructions to CPU for visualization
        images = images.cpu().numpy()
        reconstructions = reconstructions.cpu().numpy()
        
        # Plot the original and reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(15, 5))
        
        for i in range(num_samples):
            # Original images
            ax = axes[0, i]
            ax.imshow(np.transpose(images[i], (1, 2, 0)))
            ax.set_title("Original")
            ax.axis('off')
            
            # Reconstructions
            ax = axes[1, i]
            ax.imshow(np.transpose(reconstructions[i], (1, 2, 0)))
            ax.set_title("Reconstruction")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def train_cae(model, loader, num_epochs=5, lr=1e-3, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for batch in loader:
            images, _ = batch  # We don't need labels for autoencoders
            images = images.float() # Model expects float
            images = images.squeeze(1)  # Remove the dimension with size 1
            images = images.permute(0, 3, 1, 2)   # Move the channels dimension to the correct position
            outputs = model(images)
            loss = criterion(outputs, images)  # Reconstruction loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")