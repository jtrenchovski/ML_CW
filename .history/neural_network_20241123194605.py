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
from sklearn.preprocessing import StandardScaler

data_train = np.loadtxt("regression_train.txt")
data_test = np.loadtxt("regression_test.txt")

X_train = data_train[:,0].reshape(-1, 1)
y_train = data_train[:,1] 

x_test = data_test[:,0].reshape(-1, 1)
y_test = data_test[:,1] 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(x_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(X_train_scaled)

poly2 = PolynomialFeatures(degree=3)
x_poly_test = poly2.fit_transform(X_test_scaled)

mlp_classifier = MLPRegressor(
                    hidden_layer_sizes=(100, 100),
                    activation='relu',
                    solver='lbfgs',
                    learning_rate_init=5,
                    max_iter=2000,
                    n_iter_no_change=25,
                    random_state=1,
                    batch_size=10
                    )
mlp_classifier.fit(X_train_scaled, y_train_scaled)

print("Score of train data: ", mlp_classifier.score(X_train, y_train))
print("Score of test data: ", mlp_classifier.score(x_test, y_test))