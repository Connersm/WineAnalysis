"""
This script trains a linear regression model to predict wine density
using five physicochemical features, and visualizes the model prediction
with respect to fixed acidity while holding other features constant.

Steps:
1. Load the red wine dataset.
2. Define features and target (density).
3. Split the dataset into training and testing sets, and standardize features.
4. Train a linear regression model.
5. Construct a dataset varying fixed acidity while keeping other features fixed at their means.
6. Predict density for the constructed data.
7. Plot predictions along with training and testing data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
Load dataset.
"""
df = pd.read_csv("winequality-red.csv", delimiter=';')

"""
Define features (fixed acidity, residual sugar, volatile acidity,
citric acid, chlorides) and target (density).
"""
X = df[['fixed acidity', 'residual sugar', 'volatile acidity', 'citric acid', 'chlorides']]
y = df['density']

"""
Split dataset into training and testing sets.
Standardize features using training set mean and variance.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
Fit linear regression model.
"""
model = LinearRegression()
model.fit(X_train_scaled, y_train)

"""
Prepare plotting data by varying fixed acidity across its range
while holding other features constant at their mean values.
"""
fixed_acidity_range = np.linspace(X['fixed acidity'].min(), X['fixed acidity'].max(), 100).reshape(-1, 1)
residual_sugar_mean = np.full_like(fixed_acidity_range, X['residual sugar'].mean())
volatile_acidity_mean = np.full_like(fixed_acidity_range, X['volatile acidity'].mean())
citric_acid_mean = np.full_like(fixed_acidity_range, X['citric acid'].mean())
chlorides_mean = np.full_like(fixed_acidity_range, X['chlorides'].mean())

"""
Stack features in correct order for prediction.
"""
X_plot = np.hstack((
    fixed_acidity_range,
    residual_sugar_mean,
    volatile_acidity_mean,
    citric_acid_mean,
    chlorides_mean
))

"""
Transform plotting data and predict density.
"""
X_plot_scaled = scaler.transform(X_plot)
y_plot = model.predict(X_plot_scaled)

"""
Plot training data, testing data, and model prediction curve.
"""
plt.figure(figsize=(10, 6))
plt.scatter(X_train['fixed acidity'], y_train, color='blue', alpha=0.5, label='Train Data')
plt.scatter(X_test['fixed acidity'], y_test, color='red', alpha=0.5, label='Test Data')
plt.plot(fixed_acidity_range, y_plot, color='black', linewidth=2, label='Model Prediction')

plt.xlabel('Fixed Acidity')
plt.ylabel('Density')
plt.title('Model Prediction vs Fixed Acidity (Others held constant)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fixed_acidity_model_plot.png")
plt.show()
