import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from src.pet_dataset import OxfordPetsDataset
import matplotlib.pyplot as plt

# Set up transformations and dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = OxfordPetsDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define a function to train and log performance
def train_and_log(model, optimizer, criterion, epochs=5):
    model.train()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return losses, accuracies

# Train and log MobileNet SSD
def run_mobilenet(epochs=5):
    from torchvision.models import MobileNet_V2_Weights
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return train_and_log(model, optimizer, criterion, epochs)

# Train and log DETR
def run_detr(epochs=5):
    model = models.detection.detr_resnet50(pretrained=True)
    model.class_embed = nn.Linear(model.class_embed.in_features, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    return train_and_log(model, optimizer, criterion, epochs)

# Plot results
def plot_results(mobilenet_metrics, detr_metrics):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    axs[0].plot(mobilenet_metrics[0], label="MobileNet SSD")
    axs[0].plot(detr_metrics[0], label="DETR")
    axs[0].set_title("Training Loss over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot accuracies
    axs[1].plot(mobilenet_metrics[1], label="MobileNet SSD")
    axs[1].plot(detr_metrics[1], label="DETR")
    axs[1].set_title("Training Accuracy over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()

    plt.show()

if __name__ == "__main__":
    epochs = 5
    print("Training MobileNet SSD...")
    mobilenet_metrics = run_mobilenet(epochs)
    
    print("Training DETR...")
    detr_metrics = run_detr(epochs)
    
    print("Plotting results...")
    plot_results(mobilenet_metrics, detr_metrics)
