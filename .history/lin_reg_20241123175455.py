import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage import io
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data_train = np.loadtxt("regression_train.txt")
data_test = np.loadtxt("regression_test.txt")

X_train = data_train[:,0].reshape(-1, 1)
y_train = data_train[:,1] 

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(X_train)

x_test = data_test[:,0].reshape(-1, 1)
y_test = data_test[:,1] 

poly2 = PolynomialFeatures(degree=3)
x_poly_test = poly2.fit_transform(x_test)

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_train)

train_score = lin_reg.score(x_poly, y_train)
test_score = lin_reg.score(x_poly_test, y_test)

print("Score on train data: ", train_score)
print("Score on test data: ", test_score)

RMSE_train = np.sqrt(mean_squared_error(y_train, lin_reg.predict(x_poly)))
print('Train RMSE: ', RMSE_train)
RMSE_test = np.sqrt(mean_squared_error(y_test, lin_reg.predict(x_poly_test)))
print('Test RMSE: ', RMSE_test)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X_train, y_train, c='#8c564b')
ax.scatter(x_test,y_test, c='#ff7f0e')
ax.plot(x_poly_test, lin_reg.predict(x_poly_test))
ax.set_xlabel('Real value')
ax.set_ylabel('Prediction')