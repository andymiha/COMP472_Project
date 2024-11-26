import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump
from utility import print_classification_metrics, plot_confusion_matrix
from data_processing import prepare_and_export_data

# Prepare and load data
device = "cpu"  # Adjust based on your setup
data = prepare_and_export_data(device)

# Load PCA-reduced data
train_features_pca, train_labels = data["train_pca"]
test_features_pca, test_labels = data["test_pca"]

# Custom Decision Tree Implementation
class CustomDecisionTree:
    def __init__(self, max_depth=50, min_samples_leaf=5, min_gini_improvement=0.01):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gini_improvement = min_gini_improvement
        self.tree = None
        print(f"[INIT] DecisionTree initialized with max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, min_gini_improvement={min_gini_improvement}")

    def gini(self, labels):
        counts = np.bincount(labels)
        probabilities = counts / len(labels)
        return 1 - np.sum(probabilities**2)

    def split(self, data, labels, feature, threshold):
        left_mask = data[:, feature] <= threshold
        right_mask = ~left_mask
        return data[left_mask], labels[left_mask], data[right_mask], labels[right_mask]

    def find_best_split(self, data, labels):
        best_gini = float('inf')
        best_split = None
        n_samples, n_features = data.shape

        for feature in range(n_features):
            thresholds = np.linspace(np.min(data[:, feature]), np.max(data[:, feature]), 10)
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split(data, labels, feature, threshold)

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                gini_split = (len(y_left) / n_samples) * self.gini(y_left) + \
                             (len(y_right) / n_samples) * self.gini(y_right)

                if gini_split < best_gini:
                    improvement = best_gini - gini_split
                    if improvement > self.min_gini_improvement:
                        best_gini = gini_split
                        best_split = {'feature': feature, 'threshold': threshold}
        return best_split

    def build_tree(self, data, labels, depth=0):
        if depth == self.max_depth or len(np.unique(labels)) == 1:
            return Counter(labels).most_common(1)[0][0]

        best_split = self.find_best_split(data, labels)
        if best_split is None:
            return Counter(labels).most_common(1)[0][0]

        feature = best_split['feature']
        threshold = best_split['threshold']
        X_left, y_left, X_right, y_right = self.split(data, labels, feature, threshold)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.build_tree(X_left, y_left, depth + 1),
            'right': self.build_tree(X_right, y_right, depth + 1)
        }

    def fit(self, data, labels):
        print("[FIT] Starting tree training...")
        self.tree = self.build_tree(data, labels)
        print("[FIT] Tree training complete.")

    def predict_one(self, x):
        node = self.tree
        while isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node

    def predict(self, data):
        print("[PREDICT] Starting predictions...")
        return np.array([self.predict_one(x) for x in data])


# Train and evaluate the custom decision tree
custom_tree = CustomDecisionTree(max_depth=50)
custom_tree.fit(train_features_pca, train_labels)
custom_tree_predictions = custom_tree.predict(test_features_pca)

# Evaluate Custom Decision Tree
print_classification_metrics(test_labels, custom_tree_predictions, "Custom Decision Tree")
plot_confusion_matrix(test_labels, custom_tree_predictions, "Custom Decision Tree", class_labels=np.unique(test_labels))

# Experiment with varying depths
depths = [10, 25, 50]
custom_accuracies = []

for depth in depths:
    tree = CustomDecisionTree(max_depth=depth)
    tree.fit(train_features_pca, train_labels)
    predictions = tree.predict(test_features_pca)
    accuracy = accuracy_score(test_labels, predictions)
    custom_accuracies.append(accuracy)
    print(f"Depth: {depth}, Accuracy: {accuracy:.2f}")

# Plot accuracy vs. depth
plt.figure(figsize=(8, 5))
plt.plot(depths, custom_accuracies, label="Custom Decision Tree", marker='o')
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Effect of Tree Depth on Accuracy (Custom Decision Tree)")
plt.legend()
plt.grid()
plt.show()

# Scikit-Learn Decision Tree
sklearn_tree = DecisionTreeClassifier(max_depth=50, random_state=42)
sklearn_tree.fit(train_features_pca, train_labels)
sklearn_tree_predictions = sklearn_tree.predict(test_features_pca)

# Evaluate Scikit-Learn Decision Tree
print_classification_metrics(test_labels, sklearn_tree_predictions, "Scikit-Learn Decision Tree")
plot_confusion_matrix(test_labels, sklearn_tree_predictions, "Scikit-Learn Decision Tree", class_labels=np.unique(test_labels))

# Save models
custom_tree_model_path = "custom_decision_tree.pkl"
with open(custom_tree_model_path, 'wb') as f:
    pickle.dump(custom_tree, f)
print(f"Custom Decision Tree model saved to {custom_tree_model_path}")

sklearn_tree_model_path = "sklearn_decision_tree.joblib"
dump(sklearn_tree, sklearn_tree_model_path)
print(f"Scikit-Learn Decision Tree model saved to {sklearn_tree_model_path}")

# Summarize metrics for Decision Trees
custom_tree_metrics = [
    accuracy_score(test_labels, custom_tree_predictions),
    precision_score(test_labels, custom_tree_predictions, average='weighted'),
    recall_score(test_labels, custom_tree_predictions, average='weighted'),
    f1_score(test_labels, custom_tree_predictions, average='weighted'),
]

sklearn_tree_metrics = [
    accuracy_score(test_labels, sklearn_tree_predictions),
    precision_score(test_labels, sklearn_tree_predictions, average='weighted'),
    recall_score(test_labels, sklearn_tree_predictions, average='weighted'),
    f1_score(test_labels, sklearn_tree_predictions, average='weighted'),
]

results_tree_comp = pd.DataFrame(
    {
        "Model": ["Custom Decision Tree", "Scikit-Learn Decision Tree"],
        "Accuracy": [custom_tree_metrics[0], sklearn_tree_metrics[0]],
        "Precision": [custom_tree_metrics[1], sklearn_tree_metrics[1]],
        "Recall": [custom_tree_metrics[2], sklearn_tree_metrics[2]],
        "F1-Score": [custom_tree_metrics[3], sklearn_tree_metrics[3]],
    }
)

print("\nEvaluation Summary:")
print(results_tree_comp)
