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