# Image Classification Project: CIFAR-10 Dataset

## Overview

This repository contains the implementation of an **Image Classification Project** using the CIFAR-10 dataset. The project explores multiple machine learning and deep learning models, including:

- Naive Bayes (Custom and Scikit-learn)
- Decision Tree (Custom and Scikit-learn)
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN) using VGG architectures

While the repository includes modular scripts for running models, saving/loading data, and processing, much of the development and experimentation were conducted in a Jupyter Notebook. It is recommended to run the `.ipynb` file in Google Colab for ease of use and GPU acceleration.

---

## Repository Structure

- **`scripts/`**: Contains Python scripts for individual models:
  - `naive_bayes.py`: Implements and evaluates Naive Bayes models.
  - `decision_tree.py`: Implements and evaluates Decision Tree models.
  - `mlp.py`: Implements and evaluates the Multi-Layer Perceptron model.
  - `cnn.py`: Implements and evaluates Convolutional Neural Networks.
- **`data_processing.py`**: Prepares and exports filtered raw data and PCA-reduced data.
- **`utility.py`**: Contains reusable utility functions for metrics calculation and plotting confusion matrices.
- **Saved Models**: Includes `.pkl` or `.pth` files for trained models.
- **Notebook**: The primary development was done in the `main.ipynb` file, designed for Google Colab execution.

---

## Requirements

This project uses Python and various libraries for machine learning, data processing, and visualization. Install the necessary dependencies using:

```bash
pip install -r requirements.txt
```
