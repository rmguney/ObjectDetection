from src.pet_dataset import OxfordPetsDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Define transformations (resizing and normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the dataset
dataset = OxfordPetsDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Test loading a batch of images
for images, labels in dataloader:
    print("Batch of images:", images.shape)
    print("Batch of labels:", labels)
    break  # Just load one batch for testing
