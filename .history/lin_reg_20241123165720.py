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

X = np.loadtxt("regression_train.txt")
y = np.loadtxt("regression_test.txt")
print(X)

lin_reg = LinearRegression()
lin_reg.fit(X)