# Handwritten Digit Classification using SVM (MNIST)

## Overview
This project implements Support Vector Machine (SVM) classifiers to distinguish
between handwritten digits **2** and **3** from the MNIST dataset.

Two models are evaluated:
- Linear SVM
- RBF (Gaussian) SVM

## Dataset
The MNIST dataset is provided in IDX binary format and loaded manually
without using high-level libraries.

Images are:
- 28×28 grayscale
- Flattened into 784-dimensional feature vectors

## Project Structure
cosc6349-final-svm-mnist/
├── src/
│ └── train_eval.py
├── data/
│ ├── train-images.idx3-ubyte
│ ├── train-labels.idx1-ubyte
│ ├── t10k-images.idx3-ubyte
│ └── t10k-labels.idx1-ubyte
├── results/
│ └── metrics.txt
├── requirements.txt
└── README.md

## Methodology
1. Load MNIST IDX image and label files
2. Filter dataset to digits 2 and 3
3. Convert labels to binary (2 → 0, 3 → 1)
4. Standardize features using `StandardScaler`
5. Train Linear and RBF SVM classifiers
6. Evaluate using accuracy and confusion matrix

## Results
The RBF SVM achieved approximately **99% accuracy** on the test set.

Confusion matrix shows very low misclassification between digits 2 and 3.

Detailed metrics are saved in:

results/metrics.txt

## Project Report

The final project report is included in this repository:

📄 Chaudhry_Binary_Classification_of_Handwritten_Digits_Using_Support_Vector_Machines.pdf


## Dependencies

See requirements.txt.

## How to Run
Activate virtual environment and run:

```bash
python src/train_eval.py
