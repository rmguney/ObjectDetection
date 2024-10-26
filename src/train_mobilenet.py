import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from src.pet_dataset import OxfordPetsDataset

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset and DataLoader
dataset = OxfordPetsDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load MobileNet model with the updated weights argument and adjust for binary classification
from torchvision.models import MobileNet_V2_Weights
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set the model to training mode
model.train()

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Print statistics per epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")

print("Training complete.")
