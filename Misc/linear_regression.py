#!/usr/bin/enc python3
# _*_ coding: utf-8 _*_

print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# # Load the diabetes dataset
# diabetes = datasets.load_diabetes()


# # Use only one feature
# diabetes_X = diabetes.data[:, np.newaxis, 2]

# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test = diabetes.target[-20:]

x= [80, 68, 94, 72, 74, 83, 56, 68, 65, 75, 88]
y= [72, 71, 96, 77, 82, 72, 58, 83, 78, 80, 92]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
# regr.fit(diabetes_X_train, diabetes_y_train)
regr.fit(x, y)

# Make predictions using the testing set
# diabetes_y_pred = regr.predict(diabetes_X_test)
y_pred = regr.predict(x)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(x, y_pred))
    #   % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
print('Variance score: %.2f' % r2_score(y, y_pred))

# Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.scatter(x, y,  color='black')
plt.plot(x, y, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

