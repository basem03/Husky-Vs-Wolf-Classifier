# Wolf vs. Husky Image Classification
This repository implements a CNN to classify wolf and husky images using PyTorch. It includes data preprocessing, training, hyperparameter tuning, evaluation, and Grad-CAM visualization.
# Features

Data preprocessing with augmentation (resize, flip, rotation).
Custom CNN for binary classification.
Hyperparameter tuning (learning rate, batch size).
K-Fold cross-validation for robust evaluation.
Grad-CAM for visualizing model decisions.
Test set evaluation with data leakage prevention.

 

# Requirements

Python 3.10+
PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Scikit-learn, TQDM, Jupyter
See requirements.txt for details.



# Model Architecture

CustomCNN: Conv2D layers with ReLU, max-pooling, and dense layers for classification.
Supports GPU acceleration.

# Training and Evaluation

Uses Adam optimizer and cross-entropy loss.
Data augmentation for training.
K-Fold cross-validation and hyperparameter tuning.
Evaluates on separate test sets to ensure no data leakage.

# Grad-CAM
Visualizes image regions influencing predictions using heatmaps.
Contributing

Fork and create a branch (git checkout -b feature-branch).
Commit changes (git commit -m "Add feature").
Push (git push origin feature-branch).
Open a pull request.


For issues, open a GitHub issue or email basemhesham200318@gmail.com.
