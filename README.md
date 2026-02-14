# Bank-Marketing-Classification-System-ML-ASSIGNMENT-

# Problem Statement
This project implements multiple machine learning models to predict the success of bank marketing campaigns. Users can upload test data in CSV format and choose among several classifiers to evaluate performance.

The goal of this project is to build machine learning models that predict whether a customer will subscribe to a term deposit. This helps improve marketing efficiency, reduce unnecessary customer contact, and increase campaign success rates through data-driven decision-making.

# Dataset Description
Bank Marketing Dataset

This project uses the Bank Marketing Dataset, which contains data related to direct marketing campaigns (phone calls) conducted by a Portuguese banking institution. The objective is to predict whether a customer will subscribe to a term deposit based on demographic, financial, and campaign-related attributes.

# * Target Varible

deposit
yes – Customer subscribed to a term deposit
no – Customer did not subscribe
This is a binary classification problem.

# Dataset Overview

Type: Structured tabular dataset
Task: Supervised Machine Learning (Classification)
Number of Features: 16 input features + 1 target variable
Target Type: Binary (Yes/No)

# Feature Description

age – Age of the client
job – Type of job
marital – Marital status
education – Education level
default – Has credit in default
balance – Average yearly account balance (in euros)
housing – Has housing loan
loan – Has personal loan
contact – Contact communication type
day – Last contact day of the month
month – Last contact month
duration – Duration of last contact (in seconds)
campaign – Number of contacts performed during this campaign
pdays – Number of days since client was last contacted
previous – Number of contacts performed before this campaign
outcome – Outcome of the previous marketing campaign

# Models Used

Models Used and Evaluation Metrics

In this project, six different machine learning classification algorithms were implemented and compared to predict whether a customer will subscribe to a term deposit.

-> Models Implemented

1. Logistic Regression
A linear model used for binary classification problems. It estimates the probability of a class using the logistic (sigmoid) function.

2. Decision Tree Classifier
A tree-based model that splits data based on feature values to make classification decisions.

3. k-Nearest Neighbors (kNN)
A distance-based algorithm that classifies a data point based on the majority class among its nearest neighbors.

4. Naive Bayes (GaussianNB)
A probabilistic classifier based on Bayes’ Theorem with an assumption of feature independence.

5. Random Forest Classifier
An ensemble learning method that builds multiple decision trees and combines their outputs for better accuracy and reduced overfitting.

6. XGBoost Classifier
An advanced gradient boosting algorithm that builds trees sequentially to improve performance and handle complex patterns.

# Evaluation Matrices

|Model Name         |Accuracy| AUC      | Precision | Recall   | F1       | MCC   |
|Logistic Regression|0.825   | 0.907    | 0.827     | 0.797    | 0.812    | 0.649 |
|Decision Tree      |0.792   | 0.791    | 0.785     | 0.772    | 0.778    | 0.582 |
|KNN                |0.829   | 0.884    | 0.829     | 0.807    | 0.818    | 0.658 |
|Naive Bayes        |0.717   | 0.810    | 0.804     | 0.534    | 0.642    | 0.448 |
|Random Forest      |0.861   | 0.919    | 0.831     | 0.887    | 0.858    | 0.724 |
|XGBoost            |0.864   | 0.925    | 0.841     | 0.879    | 0.860    | 0.729 |