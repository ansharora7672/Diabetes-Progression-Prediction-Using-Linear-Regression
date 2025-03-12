import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use all available features
diabetes_X = diabetes.data

# Split the data into training and testing sets
# Using the first 412 samples for training and the last 30 for testing
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# Create a linear regression model
model = linear_model.LinearRegression()

# Train the model with the training data
model.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the test data
diabetes_y_pred = model.predict(diabetes_X_test)

# Display the model's coefficients
print('Coefficients:', model.coef_)

# Compute and print the mean squared error
print('Mean squared error: %.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Compute and print the coefficient of determination (R²)
r2 = model.score(diabetes_X_test, diabetes_y_test)
print('Coefficient of determination (R²): %.2f' % r2)
print('R² as a percentage: %.2f%%' % (r2 * 100))