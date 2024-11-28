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

X_train, X_test, y_train, y_test = train_test_split(XFull, yFull, test_size=0.2, random_state=42)

# scaler = StandardScaler().fit(XFull)
# X_scaled = scaler.transform(XFull)

# no pca

# max depth - 20, n-estimators - 10 gives 95% on test data
# normal random forest - 95% 
# tree_model2 = DecisionTreeClassifier(max_depth=5)

rand_forest1 = RandomForestClassifier()

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'max_features': ['sqrt', 'log2']
}

param_grid1 = {
    'n_estimators': [50],
    'max_depth': [25, 30, 35],
    'max_features': ['sqrt']
}

param_grid2 = {
    'n_estimators': [50],
    'max_depth': [35],
    'max_features': [5, 6, 7, 8, 9]
}

grid_search = GridSearchCV(estimator=rand_forest1,  param_grid=param_grid2, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

results = grid_search.cv_results_
for i in range(len(results['mean_test_score'])):
    print(f"Trial {i + 1}:")
    print(f"  Parameters: {results['params'][i]}")
    print(f"  Mean Test Score: {results['mean_test_score'][i]:.4f}")
    print(f"  Std Test Score: {results['std_test_score'][i]:.4f}")
    print()

print(grid_search.cv_results_)

rand_forest = RandomForestClassifier(n_estimators=50).fit(X_train, y_train) # features, min samples 3  - 0.948
y_pred = rand_forest.predict(X_train)
y_pred_test = rand_forest.predict(X_test)

print("Tree - accuracy on train data: ", rand_forest.score(X_train, y_train))
print("Tree - accuracy on test data: ", rand_forest.score(X_test, y_test))