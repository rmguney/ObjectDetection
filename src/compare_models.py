# src/compare_models.py

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from src.pet_dataset import OxfordPetsDataset
import matplotlib.pyplot as plt
import os
import time

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
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Create directories for saving models, logs, and plots
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Function to train and log performance
def train_and_log(model, optimizer, criterion, model_name, epochs=5):
    model.train()
    losses = []
    accuracies = []

    start_time = time.time()  # Start time for training

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)  # Move data to selected device

            optimizer.zero_grad()
            outputs = model(images)["logits"] if hasattr(model, "class_embed") else model(images)
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
        epoch_end = time.time()
        print(f"{model_name} - Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {epoch_end - epoch_start:.2f} sec")

    total_time = time.time() - start_time
    avg_time_per_epoch = total_time / epochs

    # Save model checkpoint
    torch.save(model.state_dict(), f"saved_models/{model_name}_checkpoint.pth")
    # Save logs
    with open(f"logs/{model_name}_log.txt", "w") as f:
        for loss, acc in zip(losses, accuracies):
            f.write(f"Loss: {loss}, Accuracy: {acc}\n")
        f.write(f"\nTotal Training Time: {total_time:.2f} seconds\n")
        f.write(f"Average Time per Epoch: {avg_time_per_epoch:.2f} seconds\n")

    print(f"{model_name} Total Training Time: {total_time:.2f} seconds")
    print(f"{model_name} Average Time per Epoch: {avg_time_per_epoch:.2f} seconds")

    return losses, accuracies, total_time, avg_time_per_epoch

# Train and log MobileNet SSD
def run_mobilenet(epochs=5):
    from torchvision.models import MobileNet_V2_Weights
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)  # Move model to selected device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"MobileNet SSD - Total Parameters: {sum(p.numel() for p in model.parameters())}")
    return train_and_log(model, optimizer, criterion, "MobileNet_SSD", epochs)

# Train and log DETR
def run_detr(epochs=5):
    model = models.detection.detr_resnet50(pretrained=True)
    model.class_embed = nn.Linear(model.class_embed.in_features, 2)
    model = model.to(device)  # Move model to selected device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    print(f"DETR - Total Parameters: {sum(p.numel() for p in model.parameters())}")
    return train_and_log(model, optimizer, criterion, "DETR", epochs)

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

    # Save the plot as an image
    fig.savefig("plots/model_comparison.png")
    plt.show()

if __name__ == "__main__":
    epochs = 5
    print("Training MobileNet SSD...")
    mobilenet_metrics = run_mobilenet(epochs)

    print("Training DETR...")
    detr_metrics = run_detr(epochs)

    print("Plotting results...")
    plot_results(mobilenet_metrics, detr_metrics)
