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

import graphviz

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

N = 100000
X = XFull[:N]
y = yFull[:N]

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

pca = PCA(n_components='mle')
X_final = pca.fit_transform(X)

X_train, x_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier().fit(X_train, y_train)

print("Tree - accuracy on train data: ", tree_model.score(X_train, y_train))
print("Tree - accuracy on train data: ", tree_model.score(x_test, y_test))

tree.plot_tree(tree_model, proportion=True)
plt.show()
