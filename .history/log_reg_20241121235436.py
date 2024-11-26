import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import graphviz

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

N = 500000
X = XFull[:N]
y = yFull[:N]

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# pca = PCA(n_components='mle')
# X_final = pca.fit_transform(X_scaled)

X_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['saga', 'liblinear'],
}
log_reg2 = LogisticRegression()

# grid_search = GridSearchCV(estimator=log_reg2, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
# grid_search.fit(X_train, y_train)

# results = grid_search.cv_results_
# for i in range(len(results['mean_test_score'])):
#     print(f"Trial {i + 1}:")
#     print(f"  Parameters: {results['params'][i]}")
#     print(f"  Mean Test Score: {results['mean_test_score'][i]:.4f}")
#     print(f"  Std Test Score: {results['std_test_score'][i]:.4f}")
#     print()

log_reg = LogisticRegression(max_iter= 100, solver='liblinear').fit(X_train, y_train)
y_pred_train = log_reg.predict(X_train)

y_pred_test = log_reg.predict(x_test)

print("Logistic regression - accuracy on train data: ", log_reg.score(X_train, y_train))
print("Logistic regression - accuracy on test data: ", log_reg.score(x_test, y_test))
