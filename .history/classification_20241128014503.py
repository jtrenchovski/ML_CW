import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

X_train, X_test, y_train, y_test = train_test_split(XFull, yFull, test_size=0.2, random_state=42)

log_reg2 = LogisticRegression()

log_reg = LogisticRegression(max_iter= 50, solver='newton-cholesky', penalty='l2').fit(X_train, y_train)
y_pred_train = log_reg.predict(X_train)
y_pred_test = log_reg.predict(X_test)

print("Logistic regression - accuracy on train data: ", log_reg.score(X_train, y_train))
print("Logistic regression - accuracy on test data: ", log_reg.score(X_test, y_test))

tree_model = DecisionTreeClassifier(max_depth=25).fit(X_train, y_train)

print("Tree - accuracy on train data: ", tree_model.score(X_train, y_train))
print("Tree - accuracy on test data: ", tree_model.score(X_test, y_test))

rand_forest = RandomForestClassifier(n_estimators=50, max_features=10).fit(X_train, y_train) # features, min samples 3  - 0.948
y_pred = rand_forest.predict(X_train)
y_pred_test = rand_forest.predict(X_test)

print("Random forest - accuracy on train data: ", rand_forest.score(X_train, y_train))
print("Random forest - accuracy on test data: ", rand_forest.score(X_test, y_test))




# scaler_robust = RobustScaler().fit(XFull)
# X_scaled_r = scaler_robust.transform(XFull)

# pca = PCA(n_components=25)
# X_final = pca.fit_transform(X)

# poly = PolynomialFeatures(degree=2)
# x_poly = poly.fit_transform(X)

# scaler = StandardScaler().fit(X_final)
# X_scaled = scaler.transform(X_final)

# X_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=42)

# log_reg = LogisticRegression(max_iter=50).fit(X_train, y_train)
# y_pred_train = log_reg.predict(X_train)
# y_pred_test = log_reg.predict(x_test)

# tree = DecisionTreeClassifier().fit(X_train, y_train)

# print("Tree - accuracy on train data: ", tree.score(X_train, y_train))
# print("Tree - accuracy on train data: ", tree.score(x_test, y_test))

# print("Logistic regression - accuracy on train data: ", log_reg.score(X_train, y_train))
# print("Logistic regression - accuracy on test data: ", log_reg.score(x_test, y_test))
