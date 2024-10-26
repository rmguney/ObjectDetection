import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from src.pet_dataset import OxfordPetsDataset
from torch import nn

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset and DataLoader
dataset = OxfordPetsDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load DETR model and modify it for binary classification
model = models.detection.detr_resnet50(pretrained=True)
model.class_embed = nn.Linear(model.class_embed.in_features, 2)
model = model.to(device)  # Move model to the selected device
model.eval()  # Set model to evaluation mode

# Test with one batch of data
def test_detr():
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to selected device
            outputs = model(images)["logits"]
            _, predicted = torch.max(outputs, 1)
            print("Predicted labels:", predicted.cpu().numpy())
            print("True labels:", labels.cpu().numpy())
            break  # Only test one batch

if __name__ == "__main__":
    test_detr()
