# Import necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_processing import prepare_and_export_data  
from utility import print_classification_metrics, plot_confusion_matrix 

# Load prepared data from data_processing.py
device = "cuda" if torch.cuda.is_available() else "cpu"
data = prepare_and_export_data(device)

# Extract filtered raw data
filtered_raw_train_data, filtered_raw_train_labels = data["filtered_raw_train"]
filtered_raw_test_data, filtered_raw_test_labels = data["filtered_raw_test"]

# Define VGG11 architecture
class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Reusable training function
def train_and_evaluate_cnn(model, train_loader, test_loader, criterion, optimizer, num_epochs=20):
    """Train CNN model and return predictions and true labels."""
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}")

    # Evaluation phase
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())

    return {"predictions": predictions, "true_labels": true_labels}

# Create DataLoader instances for filtered train and test datasets
train_dataset = TensorDataset(filtered_raw_train_data, filtered_raw_train_labels)
test_dataset = TensorDataset(filtered_raw_test_data, filtered_raw_test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize and train VGG11
vgg11_model = VGG11(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg11_model.parameters(), lr=0.01, momentum=0.9)

results_cnn = train_and_evaluate_cnn(vgg11_model, train_loader, test_loader, criterion, optimizer, num_epochs=20)

# Use utility functions to print metrics and plot confusion matrix
print_classification_metrics(results_cnn["true_labels"], results_cnn["predictions"], "VGG11")
plot_confusion_matrix(results_cnn["true_labels"], results_cnn["predictions"], "VGG11", class_labels=range(10))

# Save the VGG11 model
torch.save(vgg11_model.state_dict(), "vgg11_model.pth")
print("VGG11 model saved as 'vgg11_model.pth'.")
