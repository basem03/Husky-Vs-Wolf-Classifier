# Wolf vs. Husky Image Classification
This repository contains a Python implementation of an image classification project using a Convolutional Neural Network (CNN) to distinguish between images of wolves and huskies. The project includes data preprocessing, model training, hyperparameter tuning, evaluation, and visualization of model decisions using Grad-CAM. The codebase is organized in a Jupyter Notebook (WolfVsHusky.ipynb) and is designed to run on systems with or without GPU support.
Table of Contents

**Project Overview**
Features
Dataset
Requirements
Installation
Usage
File Structure
Model Architecture
Training and Evaluation
Grad-CAM Visualization
Contributing
License
Acknowledgments

Project Overview
The goal of this project is to build and evaluate a CNN model to classify images as either wolves or huskies. The project employs PyTorch for model implementation, includes data augmentation to improve generalization, and uses K-Fold cross-validation for robust evaluation. Hyperparameter tuning is performed to optimize model performance, and Grad-CAM is used to visualize the regions of images that influence the model's predictions.
Features

Data Preprocessing: Resizing, normalization, and augmentation (random flips and rotations) for training data.
Custom CNN Model: A convolutional neural network tailored for binary classification.
Hyperparameter Tuning: Automated tuning of learning rate and batch size using validation performance.
K-Fold Cross-Validation: Ensures robust evaluation by splitting the dataset into multiple folds.
Grad-CAM Visualization: Visualizes the areas of images that contribute to the model's decisions.
Test Set Evaluation: Evaluates model performance on separate test datasets to assess generalization.
Data Leakage Prevention: Ensures no overlap between training, validation, and test sets.

Dataset
The dataset consists of images of wolves and huskies, organized into training and test directories. The dataset is expected to be structured as follows:
data/
├── train/
│   ├── wolf/
│   ├── husky/
├── test/
│   ├── wolf/
│   ├── husky/


Training Data: Located at C:\Users\Mohamed Sakr\Downloads\data\train (update paths as needed).
Test Data: Located at C:\Users\Mohamed Sakr\Downloads\data\test (update paths as needed).
Additional Test Data: An extra test set is used to validate model generalization.

You can replace the dataset paths in the notebook with your own dataset. Ensure the dataset is organized in the same structure as above, with subfolders for each class (wolf and husky).
Requirements
To run this project, you need the following dependencies:

Python 3.10 or higher
PyTorch
Torchvision
NumPy
Pandas
Matplotlib
Scikit-learn
TQDM
Jupyter Notebook

A full list of dependencies is provided in the requirements.txt file.
Installation

Clone the Repository:
git clone https://github.com/your-username/wolf-vs-husky-classification.git
cd wolf-vs-husky-classification


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Install Jupyter Notebook:
pip install jupyter


Prepare the Dataset:

Download or prepare your dataset and place it in the data/ directory with the structure described above.
Update the dataset paths in the WolfVsHusky.ipynb notebook to point to your dataset location.



Usage

Launch Jupyter Notebook:
jupyter notebook


Open the Notebook:

Navigate to WolfVsHusky.ipynb in the Jupyter interface and open it.


Run the Notebook:

Execute the cells sequentially to preprocess the data, train the model, evaluate performance, and generate Grad-CAM visualizations.
Ensure the dataset paths are correctly set in the notebook.


Command-Line Execution (optional):If running the script outside Jupyter, provide the test data directory as an argument:
python wolf_vs_husky.py --test_data_dir /path/to/test/data


Outputs:

Model Weights: The best model weights are saved as best_model.pth.
Performance Metrics: Test accuracy is printed for both the primary and additional test datasets.
Visualizations: Grad-CAM heatmaps are displayed to show the model's focus areas.



File Structure
wolf-vs-husky-classification/
├── data/                    # Dataset directory (not included in repo)
│   ├── train/
│   ├── test/
├── WolfVsHusky.ipynb        # Main Jupyter Notebook with implementation
├── requirements.txt         # List of dependencies
├── README.md                # This file
├── best_model.pth           # Saved model weights (generated after training)

Model Architecture
The CustomCNN model is a convolutional neural network with the following structure:

Convolutional Layers: Multiple Conv2D layers with ReLU activation and max-pooling.
Fully Connected Layers: Dense layers for classification.
Output Layer: Outputs probabilities for two classes (wolf and husky).

The model is implemented in PyTorch and supports GPU acceleration if available.
Training and Evaluation

Training: The model is trained using the Adam optimizer with cross-entropy loss. Data augmentation (random flips and rotations) is applied to the training set.
Hyperparameter Tuning: The notebook tunes the learning rate and batch size to maximize validation accuracy.
Evaluation: The model is evaluated on a separate test set, achieving an accuracy of approximately 89.06% on the additional test dataset.
K-Fold Cross-Validation: Used to ensure robust performance estimates.
Data Leakage Fix: The notebook includes a section to evaluate the model on an additional test dataset to confirm no data leakage occurs.

Grad-CAM Visualization
Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the regions of input images that contribute most to the model's predictions. The show_gradcam_samples function generates heatmaps overlaid on sample images, highlighting areas of interest (e.g., facial features or fur patterns).
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Open a pull request with a detailed description of your changes.

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

PyTorch: For providing an efficient deep learning framework.
Torchvision: For dataset and transformation utilities.
Grad-CAM: For the visualization technique used to interpret model decisions.
Dataset Providers: For the wolf and husky image datasets (replace with specific credits if applicable).

For questions or issues, please open an issue on the GitHub repository or contact your-email@example.com.
