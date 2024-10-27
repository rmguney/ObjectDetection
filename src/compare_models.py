import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
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

# Define transformations without normalization (handled by processor)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load the dataset and DataLoader
dataset = OxfordPetsDataset(root_dir='data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Create directories for saving models, logs, and plots
os.makedirs("saved_models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Function to train and log performance
def train_and_log(model, optimizer, criterion, model_name, processor=None, epochs=5):
    model.train()
    losses = []
    accuracies = []

    start_time = time.time()  # Start time for training

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        # Updated processing section within `train_and_log` to optimize device handling:
        for images, labels in dataloader:
            labels = labels.to(device)  # Move labels to the selected device

            if processor:
                # Process images for Hugging Face DETR model
                inputs = processor(images=images, return_tensors="pt", do_rescale=False)
                inputs = {key: value.to(device) for key, value in inputs.items()}  # Move all inputs to device
                outputs = model(**inputs)
                logits = outputs.logits[:, 0, :]  # Use the first detected object for binary classification
            else:
                # For MobileNet or other classification models
                images = images.to(device)
                logits = model(images)

            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
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

# Train and log MobileNet
def run_mobilenet(epochs=5):
    from torchvision.models import MobileNet_V2_Weights
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)  # Move model to selected device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"MobileNet - Total Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Explicitly set processor=None to prevent errors
    return train_and_log(model, optimizer, criterion, "MobileNet", processor=None, epochs=epochs)

# Train and log Hugging Face DETR
def run_detr(epochs=5):
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model.class_labels_classifier = nn.Linear(model.class_labels_classifier.in_features, 2)  # Binary classification
    model = model.to(device)  # Move model to the selected device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    print(f"DETR - Total Parameters: {sum(p.numel() for p in model.parameters())}")
    return train_and_log(model, optimizer, criterion, "DETR", processor, epochs)

# Plot results
def plot_results(mobilenet_metrics, detr_metrics):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    axs[0].plot(mobilenet_metrics[0], label="MobileNet")
    axs[0].plot(detr_metrics[0], label="DETR")
    axs[0].set_title("Training Loss over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot accuracies
    axs[1].plot(mobilenet_metrics[1], label="MobileNet")
    axs[1].plot(detr_metrics[1], label="DETR")
    axs[1].set_title("Training Accuracy over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()

    # Save the plot as an image
    fig.savefig("plots/model_comparison.png")
    plt.show()

# Define main function to be called from main.py
def main(epochs=5):
    print("Training MobileNet...")
    mobilenet_metrics = run_mobilenet(epochs)

    print("Training DETR...")
    detr_metrics = run_detr(epochs)

    print("Plotting results...")
    plot_results(mobilenet_metrics, detr_metrics)

if __name__ == "__main__":
    main()
