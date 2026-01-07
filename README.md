OsteoCancerNet - Bone X-ray Classification
Author: Nashaat M. Hussein
Date: 2026-01-07
GitHub Repository: [https://github.com/Nashaat-Mohamed/OsteoCancerNet]
________________________________________
Overview
OsteoCancerNet is a complete Python-based pipeline for classifying bone X-ray images into two categories: Normal and BoneCancer. The framework integrates advanced image preprocessing, feature extraction using EfficientNet-B4, classification with SVM, performance evaluation, and Grad-CAM visualization for interpretability. This repository provides the full implementation to enable reproducibility of the reported results.
Features
•	Preprocessing: CLAHE for contrast enhancement and Unsharp Masking for edge enhancement.
•	Data Augmentation: Rotation, horizontal flipping, and brightness adjustment (training set only).
•	Feature Extraction: EfficientNet-B4 (pretrained on ImageNet) for deep feature extraction.
•	Classification: Support Vector Machine (SVM) with RBF kernel.
•	Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC curve, and Confusion Matrix.
•	Grad-CAM Visualization: Highlighting image regions most influential for predictions.
Requirements
•	Python >= 3.9
•	TensorFlow >= 2.10
•	OpenCV
•	NumPy
•	Matplotlib
•	scikit-learn
Install dependencies using:
pip install tensorflow opencv-python numpy matplotlib scikit-learn
Dataset Structure
The pipeline requires a dataset organized as follows:
dataset/
    train/
        Normal/
        BoneCancer/
    val/
        Normal/
        BoneCancer/
    test/
        Normal/
        BoneCancer/
Note: For reproducibility, you may provide a small subset of images in the repository. The full dataset can be obtained externally as per ethical and legal permissions.
Usage Instructions
1.	Preprocess Images: Images are automatically preprocessed (grayscale → CLAHE → Unsharp Mask → RGB conversion).
2.	Data Augmentation: Applied to training images during model training.
3.	Feature Extraction: EfficientNet-B4 extracts feature vectors from images.
4.	Feature Standardization: Features are scaled using StandardScaler.
5.	SVM Classification: Train the RBF SVM on the training set features.
6.	Evaluation: Compute Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC curve.
7.	Grad-CAM Visualization: Generates heatmaps over X-ray images for interpretability.
Run the complete pipeline with:
python osteocancernet.py
Reproducibility Notes
•	Random Seeds: Set seeds for Python, NumPy, and TensorFlow to ensure consistent results:
import numpy as np
import random
import tensorflow as tf
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
•	Dataset Split: Maintain consistent train/validation/test split.
•	Preprocessing & Hyperparameters: Ensure CLAHE settings, SVM parameters (C=10, gamma=0.01, kernel='rbf'), and data augmentation settings remain the same.
Code Availability Statement
The complete implementation of OsteoCancerNet, including preprocessing scripts, feature extraction, SVM training, evaluation routines, and Grad-CAM visualization, is publicly available at:
https://github.com/USERNAME/OsteoCancerNet
This ensures reproducibility and allows other researchers to validate, replicate, and compare the methodology reliably.
Contact
For questions or collaboration, please contact:
Nashaat M. Hussein
Email: [nmh01@fayoum.edu.eg]
