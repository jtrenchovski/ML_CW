import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler

data_train = np.loadtxt("regression_train.txt") 
data_test = np.loadtxt("regression_test.txt") 

X_train = data_train[:,0].reshape(-1, 1)
y_train = data_train[:,1] 

X_test = data_test[:,0].reshape(-1, 1) 
y_test = data_test[:,1] 

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(X_train)

poly2 = PolynomialFeatures(degree=3)
x_poly_test = poly2.fit_transform(X_test)

mlp_classifier = MLPRegressor(
                    hidden_layer_sizes=(10, 3), # 100, 3
                    activation='relu', #relu
                    solver='lbfgs', #lbfgs
                    learning_rate_init=0.001,
                    max_iter=2000,
                    n_iter_no_change=25,
                    random_state=1,
                    alpha=0
                    )
mlp_classifier.fit(x_poly, y_train)

print("Score of train data: ", mlp_classifier.score(x_poly, y_train))
print("Score of test data: ", mlp_classifier.score(x_poly_test, y_test))