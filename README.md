# Heart Disease Classification System

## Overview

This repository contains an end-to-end machine learning project focused on predicting heart disease based on clinical parameters. The project uses a dataset from the UCI Machine Learning Repository (Cleveland Heart Disease dataset) and applies various machine learning models to classify whether a patient has heart disease.

## Project Structure

1. **Problem Definition**  
   - Goal: Predict the presence of heart disease in a patient based on a set of medical attributes.

2. **Dataset**  
   - The dataset used is the Cleveland Heart Disease dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
   - The dataset contains 303 rows and 14 features, including clinical and medical information such as:
     - Age
     - Sex
     - Chest pain type
     - Resting blood pressure
     - Cholesterol
     - Fasting blood sugar
     - Maximum heart rate
     - Exercise-induced angina
     - ST depression
     - Thalium stress test result
     - Target (presence of heart disease)

## Key Steps

1. **Data Preprocessing**  
   - Loaded and cleaned the data using `pandas`.
   - Visualized data distributions and relationships using `matplotlib` and `seaborn`.

2. **Modeling**  
   - Employed multiple machine learning models such as:
     - Logistic Regression
     - K-Neighbors Classifier
     - RandomForest Classifier
   - Used `train_test_split` for splitting the dataset into training and test sets.
   - Applied cross-validation and hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV` for optimal performance.

3. **Evaluation Metrics**  
   - Evaluated model performance using:
     - Confusion Matrix
     - Precision, Recall, F1 Score
     - ROC Curve

4. **Results**  
   - Achieved over 95% accuracy, exceeding the project goal.

## Requirements

To run this project, you will need to install the following dependencies:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using:

```bash
pip install -r requirements.txt
