# Diabetes Progression Prediction Using Linear Regression

## Overview
This project aims to predict diabetes progression using the diabetes dataset provided by `scikit-learn`. The model leverages linear regression to estimate the disease progression based on various medical and demographic features.

## Dataset
The diabetes dataset consists of 442 instances with ten baseline variables, including age, sex, body mass index (BMI), average blood pressure, and six blood serum measurements.

## Features
- **age**: Age in years
- **sex**: Gender
- **bmi**: Body mass index
- **bp**: Average blood pressure
- **s1**: Total serum cholesterol (tc)
- **s2**: Low-density lipoproteins (ldl)
- **s3**: High-density lipoproteins (hdl)
- **s4**: Total cholesterol / HDL (tch)
- **s5**: Possibly log of serum triglycerides level (ltg)
- **s6**: Blood sugar level (glu)

## Target Variable
A quantitative measure of disease progression one year after baseline.

## Project Steps
1. **Data Loading and Preprocessing**:
   - Loaded the diabetes dataset from `scikit-learn`.
   - Split the dataset into training and test sets.

2. **Model Training**:
   - Utilized `linear_model.LinearRegression` from `scikit-learn` to train the model using the training set.
   - Fitted the model with the training data.

3. **Prediction and Evaluation**:
   - Made predictions on the test set.
   - Evaluated the model using Mean Squared Error (MSE).

4. **Single Data Point Prediction**:
   - Predicted diabetes progression for a new data point.

5. **Visualization**:
   - Used `matplotlib` to visualize the data and the model's predictions.

## Results
- **Mean Squared Error**: 1826.48
- **Weights (Coefficients)**: [ -1.16678648 -237.18123633 518.31283524 309.04204042 -763.10835067 458.88378916 80.61107395 174.31796962 721.48087773 79.1952801 ]
- **Intercept**: 153.05824267739402
- **New Data Point for Testing**: [[0.04, 0.05, 0.06, 0.02, 0.08, 0.03, 0.07, 0.01, 0.09, 0.04]]
- **Prediction for New Data Point**: 206.64
- 
## Conclusion This project demonstrates the application of linear regression to predict diabetes progression. The model's accuracy can be further improved by incorporating additional features, fine-tuning, and exploring other regression techniques.
