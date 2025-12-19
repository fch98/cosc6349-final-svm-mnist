"""
Handwritten Digit Classification using Support Vector Machines (SVM)

This script trains and evaluates two SVM models (Linear and RBF)
to classify handwritten digits 2 and 3 from the MNIST dataset.
"""
import numpy as np
import struct

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
def load_mnist_images(filepath):
    """
    Load MNIST image data from an IDX file and return it as a NumPy array.
    Each image is flattened into a 1D vector.
    """
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
    return images
def load_mnist_labels(filepath):
    """
    Load MNIST label data from an IDX file and return it as a NumPy array.
    """
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load MNIST data

train_images_path = "data/train-images.idx3-ubyte"
train_labels_path = "data/train-labels.idx1-ubyte"
test_images_path  = "data/t10k-images.idx3-ubyte"
test_labels_path  = "data/t10k-labels.idx1-ubyte"

X_train = load_mnist_images(train_images_path)
y_train = load_mnist_labels(train_labels_path)

X_test = load_mnist_images(test_images_path)
y_test = load_mnist_labels(test_labels_path)

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# Filter digits 2 and 3

train_mask = (y_train == 2) | (y_train == 3)
test_mask  = (y_test == 2) | (y_test == 3)

X_train = X_train[train_mask]
y_train = y_train[train_mask]

X_test = X_test[test_mask]
y_test = y_test[test_mask]

print("Filtered training shape:", X_train.shape)
print("Filtered test shape:", X_test.shape)
print("Unique training labels:", np.unique(y_train))
print("Unique test labels:", np.unique(y_test))

# Convert labels to binary: 2 -> 0, 3 -> 1
y_train = (y_train == 3).astype(int)
y_test  = (y_test == 3).astype(int)

print("After binary conversion - unique y_train:", np.unique(y_train))
print("After binary conversion - unique y_test:", np.unique(y_test))

# Scale features (very important for SVM)

scaler = StandardScaler()

# Fit ONLY on training data, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling complete.")
print("Scaled training shape:", X_train_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)

# Train SVM models

# 1) Linear SVM (baseline)
linear_svm = SVC(kernel="linear", C=1.0)
linear_svm.fit(X_train_scaled, y_train)

# 2) RBF SVM (nonlinear)
rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale")
rbf_svm.fit(X_train_scaled, y_train)

print("Model training complete.")

# Evaluate models

def evaluate_model(model, X, y, model_name="model"):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)

    print(f"\n--- {model_name} Results ---")
    print("Accuracy:", acc)
    print("Confusion Matrix:")
    print(cm)

    return acc, cm


linear_acc, linear_cm = evaluate_model(linear_svm, X_test_scaled, y_test, "Linear SVM")
rbf_acc, rbf_cm = evaluate_model(rbf_svm, X_test_scaled, y_test, "RBF SVM")

# Save results (connects to report)

import os

os.makedirs("results", exist_ok=True)

with open("results/metrics.txt", "w") as f:
    f.write("MNIST Digits 2 vs 3 (Binary: 2->0, 3->1)\n\n")

    f.write("Linear SVM\n")
    f.write(f"Accuracy: {linear_acc}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(linear_cm))
    f.write("\n\n")

    f.write("RBF SVM\n")
    f.write(f"Accuracy: {rbf_acc}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(rbf_cm))
    f.write("\n")

print("\nSaved results to: results/metrics.txt")
