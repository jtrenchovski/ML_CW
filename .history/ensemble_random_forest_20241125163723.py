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
from sklearn.ensemble import RandomForestClassifier

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

N = 100000
X = XFull[:N]
y = yFull[:N]

# scaler = StandardScaler().fit(XFull)
# X_scaled = scaler.transform(XFull)

# no pca

# max depth - 20, n-estimators - 10 gives 95% on test data
# normal random forest - 95% 
# tree_model2 = DecisionTreeClassifier(max_depth=5)

X_train, X_test, y_train, y_test = train_test_split(XFull, yFull, test_size=0.2, random_state=42)

rand_forest = RandomForestClassifier(max_features=8, min_samples_leaf=3).fit(X_train, y_train)
y_pred = rand_forest.predict(X_train)
y_pred_test = rand_forest.predict(X_test)

print("Tree - accuracy on train data: ", rand_forest.score(X_train, y_train))
print("Tree - accuracy on test data: ", rand_forest.score(X_test, y_test))