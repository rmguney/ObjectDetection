import torch
from torchvision.models.detection import detr_resnet50
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
from src.pet_dataset import OxfordPetsDataset

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
model.class_embed = nn.Linear(model.class_embed.in_features, 2)  # Adjust for 2 classes (cat and dog)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
def train_detr(epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)["logits"]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print statistics per epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    print("DETR training complete.")

# Run the training function
if __name__ == "__main__":
    train_detr()
