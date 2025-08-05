
# Traffic Sign Classification using CNN

This project implements a deep learning model to classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset. It involves training a Convolutional Neural Network (CNN) to recognize 43 different classes of traffic signs and then uses the trained model for real-time classification via a webcam.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Workflow](#workflow)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)

## Project Overview

The primary goal of this project is to build an accurate and robust traffic sign classifier. The process is divided into two main parts:

1.  **Model Training:** A CNN is built using Keras and trained on a dataset of traffic sign images. This involves loading the data, extensive preprocessing, data augmentation to improve generalization, and training the model to achieve high accuracy.
2.  **Real-Time Inference:** The trained model is saved and then loaded into a separate script that captures video from a webcam. Each frame is processed and fed to the model, which predicts the traffic sign in real-time and displays the result on the screen.

## Features

- **Deep Learning Model:** Utilizes a Convolutional Neural Network (CNN) for high-accuracy image classification.
- **Image Preprocessing:** Implements grayscale conversion, histogram equalization, and normalization for robust performance.
- **Data Augmentation:** Uses `ImageDataGenerator` to create variations of training images (rotation, zoom, shifts), preventing overfitting and enhancing model generalization.
- **Real-Time Detection:** Employs OpenCV to capture webcam feed and perform live classification of traffic signs.
- **Performance Evaluation:** Plots training/validation accuracy and loss curves to visualize the model's learning process.

## Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading:** Images are loaded from a directory structure where each subdirectory corresponds to a class of traffic signs.
2.  **Data Splitting:** The dataset is divided into training, validation, and testing sets using `train_test_split`.
3.  **Exploratory Data Analysis (EDA):** The class distribution is visualized using a bar chart, and sample images from each class are displayed.
4.  **Preprocessing:** Each image undergoes:
    - Conversion to grayscale.
    - Histogram equalization to handle lighting variations.
    - Normalization of pixel values to a range of.
5.  **Model Training:** The CNN model is trained on the preprocessed and augmented training data.
6.  **Evaluation:** The model's performance is measured on the unseen test set.
7.  **Model Saving:** The trained model is serialized and saved to a file (`model_trained.p`) using `pickle`.
8.  **Inference:** The saved model is loaded for real-time classification on a live video stream.

## Model Architecture

The CNN architecture is defined in the `myModel()` function and consists of the following layers:

| Layer Type            | Details                                           | Activation |
| --------------------- | ------------------------------------------------- | ---------- |
| Conv2D                | 60 filters, (5x5) kernel size                     | ReLU       |
| Conv2D                | 60 filters, (5x5) kernel size                     | ReLU       |
| MaxPooling2D          | (2x2) pool size                                   | -          |
| Conv2D                | 30 filters, (3x3) kernel size                     | ReLU       |
| Conv2D                | 30 filters, (3x3) kernel size                     | ReLU       |
| MaxPooling2D          | (2x2) pool size                                   | -          |
| Dropout               | Rate = 0.5 (for regularization)                   | -          |
| Flatten               | -                                                 | -          |
| Dense                 | 500 nodes                                         | ReLU       |
| Dropout               | Rate = 0.5 (for regularization)                   | -          |
| **Dense (Output Layer)** | **43 nodes (one for each class)**              | **Softmax**  |

The model is compiled using the `Adam` optimizer with a learning rate of `0.001` and `categorical_crossentropy` as the loss function.

## Dataset

This project is designed to work with the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

-   **Download:** You can find the dataset on Kaggle: [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
-   **Setup:**
    1.  Create a directory named `myData`.
    2.  Inside `myData`, create subdirectories for each of the 43 classes, named `0`, `1`, `2`, ..., `42`.
    3.  Place the corresponding images for each class into these subdirectories.
    4.  A `labels.csv` file is required in the root directory, containing the mapping between class numbers and their names (e.g., `0,Speed Limit 20 km/h`).

## Requirements

You can install the necessary libraries using pip:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Set up the dataset** as described in the [Dataset](#dataset) section.

3.  **Run the script:**
    The provided script performs both training and real-time detection.
    - First, it will train the model and save it as `model_trained.p`.
    - After the training plots are closed, it will automatically launch the webcam for real-time classification.

    ```bash
    python your_script_name.py
    ```

4.  **Real-Time Classification:**
    - Point your webcam at a traffic sign (you can use images on your phone or printed paper).
    - The application will display a "Processed Image" window showing the camera's input after preprocessing.
    - The main "Result" window will show the live camera feed with the predicted class name and confidence score overlaid.
    - Press `q` to quit the real-time detection window.

## Results

After training, the script will display two plots:
1.  **Loss Curve:** Shows the `categorical_crossentropy` loss for both training and validation sets over each epoch. A decreasing loss indicates that the model is learning.
2.  **Accuracy Curve:** Shows the classification accuracy for both training and validation sets. An increasing accuracy indicates improved performance.

The final test accuracy will be printed to the console, representing the model's performance on completely unseen data.

Example of the output window for real-time classification:
*(A placeholder image would go here showing a webcam feed with an overlay of "CLASS: Stop" and "PROBABILITY: 99.50%")*
