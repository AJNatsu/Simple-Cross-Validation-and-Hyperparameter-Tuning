# Simple-Cross-Validation-and-Hyperparameter-Tuning
 This project demonstrates the application of Ridge regression to a polynomial regression problem. It includes data preparation, model training with hyperparameter tuning using cross-validation, and model evaluation with visualizations.

## Overview

The task involves fitting a polynomial regression model to noisy sine wave data. The model complexity is controlled using Ridge regularization, and the optimal regularization strength (`alpha`) is determined through cross-validation.

## Getting Started

### Dependencies

- Python 3.8 <
- NumPy
- Matplotlib
- Scikit-learn

### Installation

Ensure you have Python installed and then install the required Python packages:

```bash
pip install numpy matplotlib scikit-learn 
```

### Data Preparation
The sine wave data is synthesized with added Gaussian noise. The dataset is split into training and test sets, with 85% of the data used for training and 15% for testing.

### Model Training
Polynomial features of the training data are generated up to a specified degree. The data is then standardized, and Ridge regression models are trained for a range of alpha values. Cross-validation is used to select the optimal alpha.

### Model Evaluation
The model with the best alpha is evaluated on both training and test sets. The evaluation metric is the root mean squared error (RMSE). Plots are generated to visualize the model's fit and to compare true vs. predicted values.
