import numpy as np

from sklearn.datasets import fetch_covtype

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

N = 10000
X = XFull[:N]
y_true = yFull[:N]

