# -*- coding: utf-8 -*-
"""
Created on Sun May 21 01:01:49 2023

@author: Administrator
"""



# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import cross_val_score

# Importing the dataset
dataset = pd.read_csv(r'D:\Extra\Sarmat\Dataset.csv')

# Replace NaN values with column means
dataset = dataset.fillna(dataset.mean())

# Define radius and mass
radius = 18.5
mass = 3.281

# Create the new column "force" based on the calculation
dataset['Force'] = mass * radius * dataset['Measured_RPM']

# Save the updated dataset to a new CSV file
dataset.to_csv(r'D:\Extra\Sarmat\0D.csv', index=False)

# Separating the features and labels
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=0)

# Scaling the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)

# Predicting on the test set
y_pred = regressor.predict(X_test)

# Print evaluation parameters for regression
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Calculate the Durbin-Watson score
dw_score = durbin_watson(y_test - y_pred)

# Print the Durbin-Watson score
print("Durbin-Watson Score:", dw_score)

# Perform cross-validation with 5 folds
cv_scores = cross_val_score(regressor, features, labels, cv=5)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Average MSE:", cv_scores.mean())


