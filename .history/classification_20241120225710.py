import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

N = 100000
X = XFull[:N]
y = yFull[:N]

X_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression().fit(X)

print(y[:100])