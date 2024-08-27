# End-to-End MLOps Project: Iris Classification using Logistic Regression

This project demonstrates the implementation of an end-to-end MLOps pipeline for a classification task on the famous **Iris dataset** using **Logistic Regression**. The entire machine learning workflow is tracked and managed using **MLflow** for model versioning, experiment tracking, and deployment.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Model Training and Tracking with MLflow](#model-training-and-tracking-with-mlflow)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to implement an end-to-end machine learning pipeline that includes:

- **Data loading and preprocessing**: Working with the Iris dataset and preparing it for model training.
- **Model development**: Training a Logistic Regression model for classification.
- **Experiment tracking**: Using MLflow to track different model versions, hyperparameters, and evaluation metrics.
- **Model evaluation**: Assessing the performance of the model using standard classification metrics.
- **Deployment (Future Work)**: Optionally extending the project for deploying the model.

This project uses MLflow to ensure the machine learning workflow is reproducible and scalable.

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) is a classic dataset in machine learning for multi-class classification tasks. It contains 150 samples with 4 features: 

- **Sepal length**
- **Sepal width**
- **Petal length**
- **Petal width**

There are 3 classes (Iris-setosa, Iris-versicolor, and Iris-virginica), with 50 samples per class.

## Project Structure


### Files

- **`main.py`**: Contains the main logic for loading data, training the model, and logging experiments to MLflow.
- **`scripts/train.py`**: Script to train the logistic regression model on the Iris dataset.
- **`scripts/evaluate.py`**: Script to evaluate the trained model and log results to MLflow.

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/end-to-end-mlops-project.git
    cd end-to-end-mlops-project
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install and run MLflow**:
    - Install MLflow if not already installed:
        ```bash
        pip install mlflow
        ```

    - Start the MLflow server:
        ```bash
        mlflow ui
        ```

    You can now access the MLflow UI by navigating to `http://127.0.0.1:5000/` in your web browser.

## Model Training and Tracking with MLflow

The model training pipeline can be run using the `main.py` script, which performs the following steps:

1. **Load and preprocess the Iris dataset**.
2. **Train a Logistic Regression model** using scikit-learn.
3. **Log the model and its parameters** (such as regularization strength, learning rate, etc.) to MLflow.
4. **Evaluate the model** and log performance metrics (accuracy, precision, recall, F1-score) to MLflow.

To train the model and log the experiments:

```bash
python main.py

### Instructions for use:
- Replace `https://github.com/your-username/end-to-end-mlops-project.git` with your actual GitHub repository link.
- Add more details under "Results" if needed after running your experiments.

This README serves as a comprehensive guide for anyone wanting to understand or contribute to your project. Let me know if you need any further adjustments!
