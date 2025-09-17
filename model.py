"""
This script applies linear regression with 5-fold cross-validation
to predict wine density using physicochemical features.

Steps:
1. Load the dataset and extract features/labels.
2. Clean and preprocess the data (remove NaNs, impute with median, standardize).
3. Split the dataset into 5 folds with KFold cross-validation.
4. Train and evaluate linear regression models on each fold.
5. Record regression coefficients, RMSE, and R2 scores.
6. Generate plots of predictions and performance metrics across folds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

"""
Load the wine quality dataset.
"""
file_path = "winequality-red.csv"  
df = pd.read_csv(file_path, delimiter=';')

"""
Extract selected features (fixed acidity, residual sugar)
and the target variable (density).
"""
data = df[['fixed acidity', 'residual sugar']]
print(data.head())

labels = df['density']
print(labels[0:5])  # print the first 5 labels

X = data
y = np.array(labels)

"""
Remove rows with NaN labels.
"""
ynan = np.isnan(y)
not_ynan = [not y for y in ynan]  # flip truth values for masking
X = X[not_ynan]
y = y[not_ynan]
X.reset_index(drop=True, inplace=True)

"""
Set up 5-fold cross-validation.
"""
kf = KFold(n_splits=5)

beta = []
RMSE_train = []
RMSE_test = []
R2_train = []
R2_test = []

"""
Perform linear regression with cross-validation.
"""
for i, (train_index, test_index) in enumerate(kf.split(X)):

    # Define the training and testing sets
    xTrain = X.loc[train_index, :]
    xTest = X.loc[test_index, :]
    yTrain = y[train_index]
    yTest = y[test_index]

    # Replace NaNs in x with median from xTrain
    med_val = xTrain.median()
    xTrain = xTrain.fillna(med_val)
    xTest = xTest.fillna(med_val)

    # Standardize features using training set mean and std
    mean_val = xTrain.mean()
    std_val = xTrain.std()
    xTrain = (xTrain - mean_val) / std_val
    xTest = (xTest - mean_val) / std_val

    # Create and train linear regression model
    regr = linear_model.LinearRegression()
    regr.fit(xTrain, yTrain)

    # Predictions for training and testing sets
    y_pred_train = regr.predict(xTrain)
    y_pred_test = regr.predict(xTest)

    """
    Save scatter plot of predicted vs true values for test fold.
    """
    plt.scatter(yTest, y_pred_test)
    plt.xlabel('True Y Values')
    plt.ylabel('Predicted Y Values')
    plt.savefig('y_pred_vs_y_true.png')

    # Save regression coefficients
    beta.append(regr.coef_)

    # Compute and save RMSE
    this_rmse_train = mean_squared_error(yTrain, y_pred_train)**0.5
    this_rmse_test = mean_squared_error(yTest, y_pred_test)**0.5
    RMSE_train.append(this_rmse_train)
    RMSE_test.append(this_rmse_test)

    # Compute and save R2 scores
    R2_train.append(r2_score(yTrain, y_pred_train))
    R2_test.append(r2_score(yTest, y_pred_test))

"""
Print regression coefficients, RMSE, and R2 scores across folds.
"""
print("\nthis is the beta: ")
print(beta)
print("this is beta avg: ")
print(sum(beta) / len(beta))

print("\nthis is the RMSE for the training set: ")
print(RMSE_train)
print("this is RMSE train avg: ")
print(sum(RMSE_train) / len(RMSE_train))

print("\nthis is the RMSE for the testing set: ")
print(RMSE_test)
print("this is RMSE test avg: ")
print(sum(RMSE_test) / len(RMSE_test))

print("\nthis is the R2 for the training set: ")
print(R2_train)
print("this is R2 train avg: ")
print(sum(R2_train) / len(R2_train))

print("\nthis is the R2 for the testing set: ")
print(R2_test)
print("this is R2 test avg: ")
print(sum(R2_test) / len(R2_test))

"""
Plot RMSE and R2 values across folds.
"""
plt.clf
plt.subplot(121)
plt.scatter(range(1, 6), RMSE_train, color="black")
plt.scatter(range(1, 6), RMSE_test, color="blue")
plt.xticks((1, 2, 3, 4, 5))
plt.xlabel('fold number')
plt.ylabel('RMSE')

plt.subplot(122)
plt.scatter(range(1, 6), R2_train, color="black")
plt.scatter(range(1, 6), R2_test, color="blue")
plt.xticks((1, 2, 3, 4, 5))
plt.xlabel('fold number')
plt.ylabel('R2')

plt.savefig('ml_results.png')
