import numpy as np
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

scaler = StandardScaler().fit(XFull)
X_scaled = scaler.transform(XFull)

pca = PCA(n_components='mle')
X_final = pca.fit_transform(X_scaled)

X_train, x_test, y_train, y_test = train_test_split(X_final, yFull, test_size=0.2, random_state=42)

N = 10000
X = XFull[:N]
y = yFull[:N]

tree_model = DecisionTreeClassifier().fit(X_train, y_train)

print("Tree - accuracy on train data: ", tree_model.score(X_train, y_train))
print("Tree - accuracy on train data: ", tree_model.score(x_test, y_test))

dot_data = sklearn.tree.export_graphviz(tree_model, out_file=None, filled=True, rounded=True, special_characters=True)  
graph = graphviz.Source(dot_data)
graph.render("iris")
display(graph)