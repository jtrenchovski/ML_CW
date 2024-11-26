import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.svm import SVC, LinearSVC

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

# N = 300000
# X = XFull[:N]
# y = yFull[:N]

scaler = StandardScaler().fit(XFull)
X_scaled = scaler.transform(XFull)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, yFull, test_size=0.2, random_state=42)

svc = LinearSVC(kernel='linear').fit(X_train, y_train)
# y_pred = svc.predict(X_train)

print("Score train: ", svc.score(X_train, y_train))