import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.svm import SVC

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

N = 300000
X = XFull[:N]
y = yFull[:N]

scaler = StandardScaler().fit(XFull)
X_scaled = scaler.transform(XFull)

X_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
