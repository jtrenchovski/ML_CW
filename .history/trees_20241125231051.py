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

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

# N = 10000
# X = XFull[:N]
# y = yFull[:N]

# scaler = StandardScaler().fit(XFull)
# X_scaled = scaler.transform(XFull)

pca = PCA(n_components=30)
X_final = pca.fit_transform(XFull)

X_train, x_test, y_train, y_test = train_test_split(XFull, yFull, test_size=0.2, random_state=42)
# param_grid1 = {
#     'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45]
# }
# param_grid2 = {
#     'max_depth': [25],
#     'criterion': ['entropy', "log_loss", "gini"]
# }
# tree_model2 = DecisionTreeClassifier()

# grid_search = GridSearchCV(estimator=tree_model2, param_grid=param_grid1, cv=4, scoring='accuracy', verbose=1)
# grid_search.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(max_depth=25).fit(X_train, y_train)
# print(grid_search.cv_results_)

# print("Best Parameters:", grid_search.best_params_)
# print("Best Cross-Validation Accuracy:", grid_search.best_score_)

print("Tree - accuracy on train data: ", tree_model.score(X_train, y_train))
print("Tree - accuracy on test data: ", tree_model.score(x_test, y_test))

depth = tree_model.tree_.max_depth
print(depth)
# tree.plot_tree(tree_model, proportion=True)
# plt.show()
