import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

import graphviz

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

N = 10000
X = XFull[:N]
y = yFull[:N]

# scaler = StandardScaler().fit(XFull)
# X_scaled = scaler.transform(XFull)

# no pca

tree_model2 = DecisionTreeClassifier(max_depth=5)

X_train, X_test, y_train, y_test = train_test_split(XFull, yFull, test_size=0.2, random_state=42)

ada = AdaBoostClassifier(n_estimators=20, estimator=tree_model2).fit(X_train, y_train)
y_pred = ada.predict(X_train)
y_pred_test = ada.predict(X_test)

print("Tree - accuracy on train data: ", ada.score(X_train, y_train))
print("Tree - accuracy on test data: ", ada.score(X_test, y_test))