# Melanoma Classification Project

## Overview
This project focuses on the classification of melanoma, a type of skin cancer, using machine learning techniques. Melanoma classification is crucial for early detection and treatment planning.

## Project Goals
- Develop a machine learning model to classify melanoma from skin lesion images.
- Achieve high accuracy and F1 score to ensure reliable classification.
- Explore and implement various preprocessing techniques, feature engineering, and model evaluation strategies.

## Project Structure
The project includes the following components and directories:

- `data/`: Directory containing the dataset in CSV format (`train.csv` and `test.csv`).
- `logs/`: Directory for storing log files (`melanoma_classification.log`).
- `models/`: Directory where trained models are saved (`melanoma_predict_model.pkl`).
- `config.py`: Configuration file for storing constants and parameters used across the project.
- `data_visualization.py`: Script for visualizing data distribution and feature correlations.
- `main.py`: Main script for data loading, preprocessing, model training, evaluation, and prediction.
- `requirements.txt`: File listing Python dependencies required for the project.
- `test_main.py`: Script for testing the main functionalities of the project.
- `.gitignore`: Git ignore file to exclude unnecessary files from version control.
- `README.md`: This file, providing an overview of the project, its structure, and usage.

## Installation and Setup
To run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/bantoinese83/melanoma-classification.git
   cd melanoma-classification
    ```
2. Install the required dependencies:
3. Run the main script to train the model and make predictions:
   ```bash
   python main.py
   ```
   
