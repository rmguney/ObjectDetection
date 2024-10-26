import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from src.pet_dataset import OxfordPetsDataset

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset
dataset = OxfordPetsDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load MobileNet model with the updated weights argument
from torchvision.models import MobileNet_V2_Weights
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # 2 classes: cat and dog

# Set the model to evaluation mode
model.eval()

# Test with one batch of data
with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        print("Predicted labels:", predicted)
        print("True labels:", labels)
        break
