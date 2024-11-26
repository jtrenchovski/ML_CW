import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

N = 10000
X = XFull[:N]
y = yFull[:N]

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000).fit(X_train, np.reshape(y_train, (-1, 1)))
y_pred_train = log_reg.predict(X_train)

y_pred_test = log_reg.predict(x_test)

print("Accuracy on train data: ", log_reg.score(y_pred_train, y_train))
print("Accuracy on test data: ", log_reg.score(y_pred_test, np.reshape(y_train, (-1, 1))))

print(y[:100])