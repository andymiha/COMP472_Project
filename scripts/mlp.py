# Import necessary libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from data_processing import prepare_and_export_data  
from utility import print_classification_metrics, plot_confusion_matrix  

# Load prepared data from data_processing.py
device = "cpu"  # Change to "cuda" if GPU is available
data = prepare_and_export_data(device)

# Extract PCA-reduced features and labels
train_features_pca, train_labels = data["train_pca"]
test_features_pca, test_labels = data["test_pca"]

# Define MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.batchnorm(self.fc2(x))
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Reusable training function
def train_and_evaluate_mlp(model, train_loader, test_loader, criterion, optimizer, num_epochs=20):
    """
    Train the MLP model and return predictions and true labels for evaluation.

    Parameters:
        model (nn.Module): The MLP model to train.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for test data.
        criterion: Loss function (e.g., CrossEntropyLoss).
        optimizer: Optimizer (e.g., SGD).
        num_epochs (int): Number of epochs to train.

    Returns:
        dict: Dictionary containing predictions and true labels.
    """
    # Training phase
    model.train()
    print(f"[TRAIN] Starting training for {num_epochs} epochs.")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Evaluation phase
    print("[EVALUATE] Starting evaluation on test data.")
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())

    return {"predictions": predictions, "true_labels": true_labels}

# Prepare CIFAR-10 DataLoader with reduced features
train_dataset = TensorDataset(torch.tensor(train_features_pca, dtype=torch.float32),
                              torch.tensor(train_labels, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(test_features_pca, dtype=torch.float32),
                             torch.tensor(test_labels, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the MLP model and training configuration
mlp_model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)

# Train and evaluate the base MLP model
results_mlp = train_and_evaluate_mlp(mlp_model, train_loader, test_loader, criterion, optimizer, num_epochs=20)

# Use utility functions to print metrics and confusion matrix
class_labels = np.unique(test_labels)
print_classification_metrics(results_mlp["true_labels"], results_mlp["predictions"], "MLP")
plot_confusion_matrix(results_mlp["true_labels"], results_mlp["predictions"], "MLP", class_labels)

# Save the MLP model in .pth format
torch.save(mlp_model.state_dict(), "mlp_model.pth")
print("MLP model saved as 'mlp_model.pth'.")

# Experimentation with varying hidden layer sizes
hidden_layer_sizes = [256, 512, 1024]  # Experiment with varying sizes
mlp_accuracies = []

for size in hidden_layer_sizes:
    print(f"\nExperimenting with hidden layer size: {size}")

    # Define and train the MLP model with the current hidden layer size
    experiment_model = MLP(hidden_size=size).to(device)
    optimizer = optim.SGD(experiment_model.parameters(), lr=0.01, momentum=0.9)
    experiment_results = train_and_evaluate_mlp(
        experiment_model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=20
    )

    # Store accuracy for the current configuration
    accuracy = accuracy_score(test_labels, experiment_results["predictions"])
    mlp_accuracies.append(accuracy)
    print(f"Hidden Layer Size {size}, Accuracy: {accuracy:.5f}")

# Plot the results of varying hidden layer sizes
plt.figure(figsize=(8, 5))
plt.plot(hidden_layer_sizes, mlp_accuracies, marker='o', label="MLP Accuracy")
plt.xlabel("Hidden Layer Size")
plt.ylabel("Accuracy")
plt.title("Effect of Hidden Layer Size on MLP Accuracy")
plt.legend()
plt.grid()
plt.show()
