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

N = 10000
X = XFull[:N]
y = yFull[:N]

scaler = StandardScaler().fit(XFull)
X_scaled = scaler.transform(XFull)

pca = PCA(n_components='mle')
X_final = pca.fit_transform(X)

X_train, x_test, y_train, y_test = train_test_split(X_scaled, yFull, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier(max_depth=10).fit(X_train, y_train)

print("Tree - accuracy on train data: ", tree_model.score(X_train, y_train))
print("Tree - accuracy on train data: ", tree_model.score(x_test, y_test))

depth = tree_model.tree_.max_depth
print(depth)
# tree.plot_tree(tree_model, proportion=True)
# plt.show()
