import numpy as np
import pickle
import pandas as pd
from sklearn.naive_bayes import GaussianNB
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

# Custom Naive Bayes Implementation
class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(self._gaussian_likelihood(c, x))  # Sum log of Gaussian likelihoods
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _gaussian_likelihood(self, c, x):
        mean = self.mean[c]
        var = self.var[c]
        numerator = -0.5 * ((x - mean) ** 2) / (var + 1e-9)
        denominator = -0.5 * np.log(2 * np.pi * (var + 1e-9))
        return numerator + denominator


# Train and evaluate Custom Naive Bayes
custom_nb = GaussianNaiveBayes()
custom_nb.fit(train_features_pca, train_labels)
custom_nb_predictions = custom_nb.predict(test_features_pca)

# Train and evaluate Scikit-Learn Naive Bayes
scikit_nb = GaussianNB()
scikit_nb.fit(train_features_pca, train_labels)
scikit_nb_predictions = scikit_nb.predict(test_features_pca)

# Save models
with open("custom_nb_model.pkl", "wb") as f:
    pickle.dump(custom_nb, f)
print("Custom Naive Bayes model saved as 'custom_nb_model.pkl'.")

dump(scikit_nb, "scikit_nb_model.joblib")
print("Scikit-Learn Naive Bayes model saved as 'scikit_nb_model.joblib'.")

# Class labels for confusion matrix
class_labels = np.unique(test_labels)

# Evaluate Custom Naive Bayes
print_classification_metrics(test_labels, custom_nb_predictions, "Custom Naive Bayes")
plot_confusion_matrix(test_labels, custom_nb_predictions, "Custom Naive Bayes", class_labels)

# Evaluate Scikit-Learn Naive Bayes
print_classification_metrics(test_labels, scikit_nb_predictions, "Scikit-Learn Naive Bayes")
plot_confusion_matrix(test_labels, scikit_nb_predictions, "Scikit-Learn Naive Bayes", class_labels)

# Summary table
custom_nb_metrics = [
    accuracy_score(test_labels, custom_nb_predictions),
    precision_score(test_labels, custom_nb_predictions, average='weighted'),
    recall_score(test_labels, custom_nb_predictions, average='weighted'),
    f1_score(test_labels, custom_nb_predictions, average='weighted'),
]

scikit_nb_metrics = [
    accuracy_score(test_labels, scikit_nb_predictions),
    precision_score(test_labels, scikit_nb_predictions, average='weighted'),
    recall_score(test_labels, scikit_nb_predictions, average='weighted'),
    f1_score(test_labels, scikit_nb_predictions, average='weighted'),
]

results = pd.DataFrame(
    {
        "Model": ["Custom Naive Bayes", "Scikit-Learn Naive Bayes"],
        "Accuracy": [custom_nb_metrics[0], scikit_nb_metrics[0]],
        "Precision": [custom_nb_metrics[1], scikit_nb_metrics[1]],
        "Recall": [custom_nb_metrics[2], scikit_nb_metrics[2]],
        "F1-Score": [custom_nb_metrics[3], scikit_nb_metrics[3]],
    }
)

print("\nEvaluation Summary:")
print(results)
