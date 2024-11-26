# Import necessary libraries
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# Function to filter raw dataset
def filter_raw_dataset(raw_data, num_samples_per_class):
    class_counts = {i: 0 for i in range(10)}
    filtered_data = []
    filtered_labels = []
    samples_collected = 0

    for img, label in raw_data:
        if class_counts[label] < num_samples_per_class:
            filtered_data.append(ToTensor()(img))  # Convert PIL image to tensor
            filtered_labels.append(label)
            class_counts[label] += 1
            samples_collected += 1

        if samples_collected >= num_samples_per_class * 10:
            break

    return torch.stack(filtered_data), torch.tensor(filtered_labels)

# Function to filter transformed dataset
def filter_dataset(dataset, num_samples_per_class, batch_size=32):
    class_counts = {i: 0 for i in range(10)}
    filtered_data = []
    filtered_targets = []
    total_samples_needed = 10 * num_samples_per_class
    samples_collected = 0

    # Use DataLoader for batch processing
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    progress_bar = tqdm(total=total_samples_needed, desc="Collecting Samples")

    for images, labels in data_loader:
        for img, label in zip(images, labels):
            class_label = label.item()

            if class_counts[class_label] < num_samples_per_class:
                filtered_data.append(img)
                filtered_targets.append(label)
                class_counts[class_label] += 1
                samples_collected += 1
                progress_bar.update(1)

            if samples_collected >= total_samples_needed:
                progress_bar.close()
                return torch.stack(filtered_data), torch.tensor(filtered_targets)

    progress_bar.close()
    return torch.stack(filtered_data), torch.tensor(filtered_targets)

# Function to extract features using ResNet18
def extract_features(model, data, device, dataset_name="Dataset"):
    features = []
    print(f"Extracting features for {dataset_name}...")

    with torch.no_grad():
        for img in tqdm(data, desc=f"Extracting {dataset_name} Features"):
            img = img.unsqueeze(0).to(device)
            feature = model(img).squeeze()
            features.append(feature.cpu().numpy())

    return np.array(features)

# Function to prepare data and export it
def prepare_and_export_data(device):
    # Load raw CIFAR-10 data
    raw_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=None, download=True)
    raw_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=None, download=True)
    raw_train_data = [(img, label) for img, label in raw_train_dataset]
    raw_test_data = [(img, label) for img, label in raw_test_dataset]

    # Filter raw datasets
    filtered_raw_train_data, filtered_raw_train_labels = filter_raw_dataset(raw_train_data, 500)
    filtered_raw_test_data, filtered_raw_test_labels = filter_raw_dataset(raw_test_data, 100)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformed CIFAR-10 datasets
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    train_data, train_labels = filter_dataset(train_dataset, 500)
    test_data, test_labels = filter_dataset(test_dataset, 100)

    # Load pre-trained ResNet model
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the fully connected layer
    model = model.to(device)
    model.eval()

    # Extract features
    train_features = extract_features(model, train_data, device, dataset_name="Training Dataset")
    test_features = extract_features(model, test_data, device, dataset_name="Testing Dataset")

    # Apply PCA
    pca = PCA(n_components=50)
    train_features_pca = pca.fit_transform(train_features)
    test_features_pca = pca.transform(test_features)

    return {
        "filtered_raw_train": (filtered_raw_train_data, filtered_raw_train_labels),
        "filtered_raw_test": (filtered_raw_test_data, filtered_raw_test_labels),
        "train_pca": (train_features_pca, train_labels.numpy()),  
        "test_pca": (test_features_pca, test_labels.numpy())      
    }
