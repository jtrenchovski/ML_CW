import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA

XFull, yFull = fetch_covtype(shuffle=True, return_X_y=True, download_if_missing=True)

K = 7
N = 10000
X = XFull[:N]
y_true = yFull[:N]

print("Shape of X: ", X.shape)

# scaler = StandardScaler().fit(X)
# X_scaled = scaler.transform(X)

# min_max_scaler = MinMaxScaler().fit(X)
# X_scaled_MM = min_max_scaler.transform(X)

pca = PCA(n_components=0.99999999999999)

X_pca = pca.fit_transform(X)
print(pca.n_components_)

X_pca_gmm = pca.fit_transform(X)
print(pca.n_components_)

power_scaler = PowerTransformer().fit(X_pca)
X_scaled_power = power_scaler.transform(X_pca)

scaler = RobustScaler().fit(X_pca)
X_final = scaler.transform(X_pca)
# without std the results get worse by 7%. Without mean around 3 maybe 
# not even that. There are probably some outliers

kmeans = KMeans(K, init='random', n_init=1).fit(X_final)
y_pred = kmeans.predict(X_final)

# kmeans is the best initialisation method. Other give poor result. This is ran 
# on not normalised dimension pca reduced data. Best us 51 and i think it was with the parameters mentioned
# above
gmm = GaussianMixture(n_components=K, n_init=1, covariance_type='tied').fit(X_final)
y_pred_gmm = gmm.predict(X_final)

y_pred_random = np.random.randint(0, K, X.shape[0])

def count_errors(y_true, y_pred):
    error_count = 0
    number_of_pairs = 0
    # Compare all pairs of datapoints
    for i, j in combinations(range(len(y_true)), 2):  # All pairs
        if y_true[i] == y_true[j]:  # Same true class
            number_of_pairs += 1
            if y_pred[i] != y_pred[j]:  # Different clusters
                error_count += 1
    return (error_count, number_of_pairs)


error_kmeans = count_errors(y_true, y_pred)
print("Error count - k-means: ", error_kmeans[0])
print("Total pairs: ", error_kmeans[1])
print("Procentage error K-means: ", error_kmeans[0]/error_kmeans[1])

error_gmm = count_errors(y_true, y_pred_gmm)
print("Error count - GMM:", error_gmm[0])
print("Total pairs: ", error_gmm[1])
print("Procentage error GMM: ", error_gmm[0]/error_gmm[1])

error_random_baseline = count_errors(y_pred_random, y_true)
print("Error count - random baseline:", error_random_baseline[0])
print("Total pairs: ", error_random_baseline[1])
print("Procentage error random baseline: ", error_random_baseline[0]/error_random_baseline[1])
