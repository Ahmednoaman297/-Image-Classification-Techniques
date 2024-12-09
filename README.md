# Image Classification Techniques

This repository contains an exploration of various techniques for image classification. The goal of this project is to compare different feature extraction methods and classifiers to evaluate their performance and identify the best approach for classifying RGB images into categories.

---

## Objectives

- Understand the importance of image classification in computer vision.
- Explore traditional feature extraction methods such as HOG, color histograms, and Local Binary Patterns (LBP).
- Implement classifiers like KNN, K-Means (from scratch), and SVM (using Scikit-learn).
- Compare these methods with state-of-the-art Convolutional Neural Networks (CNNs).

---

## Features and Methods

### Feature Extraction
1. **HOG (Histogram of Oriented Gradients):**
   Extracts the structure and texture of images by analyzing gradients.
2. **Color Histograms:**
   Encodes color distribution in the image.
3. **LBP (Local Binary Patterns):**
   Captures texture by comparing pixel intensities.

### Classification Methods
1. **KNN (K-Nearest Neighbors):**
   Custom implementation using Euclidean distance to classify.
2. **K-Means Clustering:**
   Custom implementation to group images into clusters.
3. **SVM (Support Vector Machine):**
   Utilized from Scikit-learn for robust classification.
4. **CNN (Convolutional Neural Network):**
   Deep learning-based end-to-end feature extraction and classification.

---

## Dataset

This project uses an RGB image dataset with at least 100 categories (e.g., Caltech 101). Preprocessing includes resizing, normalization, and splitting into training and testing sets.

---

## Repository Structure

```plaintext
├── FeatureExtraction/
│   ├── hog.py
│   ├── color_histograms.py
│   ├── lbp.py
├── Classifiers/
│   ├── knn.py
│   ├── kmeans.py
│   ├── svm.py
│   ├── cnn.py
├── Utils/
│   ├── image_preprocessing.py
│   ├── metrics.py
│   ├── visualization.py
├── Dataset/
│   ├── train/
│   ├── test/
├── Results/
│   ├── graphs/
│   ├── comparison_report.md
├── README.md
├── requirements.txt
└── main.py
