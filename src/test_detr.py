import torch
from torchvision.models.detection import detr_resnet50
from torch.utils.data import DataLoader
from torchvision import transforms
from src.pet_dataset import OxfordPetsDataset
from torch import nn

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
model = detr_resnet50(pretrained=True)
model.class_embed = nn.Linear(model.class_embed.in_features, 2)  # 2 classes: cat and dog
model.eval()  # Set model to evaluation mode

# Test with one batch of data
def test_detr():
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)["logits"]
            _, predicted = torch.max(outputs, 1)
            print("Predicted labels:", predicted)
            print("True labels:", labels)
            break  # Only test one batch

if __name__ == "__main__":
    test_detr()
