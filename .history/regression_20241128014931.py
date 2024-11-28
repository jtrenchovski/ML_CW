import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data_train = np.loadtxt("regression_train.txt")
data_test = np.loadtxt("regression_test.txt")

X_train = data_train[:,0].reshape(-1, 1)
y_train = data_train[:,1] 

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train)

x_test = data_test[:,0].reshape(-1, 1)
y_test = data_test[:,1] 

poly2 = PolynomialFeatures(degree=3)
X_poly_test = poly2.fit_transform(x_test)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)

train_score = lin_reg.score(X_poly, y_train)
test_score = lin_reg.score(X_poly_test, y_test)

print("Score on train data: ", train_score)
print("Score on test data: ", test_score)

RMSE_train = np.sqrt(mean_squared_error(y_train, lin_reg.predict(X_poly)))
print('Train RMSE: ', round(RMSE_train,2))
RMSE_test = np.sqrt(mean_squared_error(y_test, lin_reg.predict(X_poly_test)))
print('Test RMSE: ', round(RMSE_test,2))
