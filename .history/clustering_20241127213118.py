import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.preprocessing import StandardScaler,  RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

K = 7
N = 10000
X = XFull[:N]
y_true = yFull[:N]

print("Shape of X: ", X.shape)

# scaler = StandardScaler().fit(X)
# X_scaled = scaler.transform(X)

pca = PCA(n_components='mle')

X_pca = pca.fit_transform(X)
print(pca.n_components_)

# Standard scaler without mean - 50%, gmm better around 30 (i think it was with random)

scaler = StandardScaler(with_mean=False).fit(X_pca)
X_final = scaler.transform(X_pca)
# without std the results get worse by 7%. Without mean around 3 maybe 
# not even that. There are probably some outliers

kmeans = KMeans(K, init='k-means++', n_init=10).fit(X_final)
y_pred = kmeans.predict(X_final)

# kmeans is the best initialisation method. Other give poor result. This is ran 
# on not normalised dimension pca reduced data. Best us 51 and i think it was with the parameters mentioned
# above
gmm = GaussianMixture(n_components=K, n_init=10, covariance_type='tied').fit(X_final)
y_pred_gmm = gmm.predict(X_final)

y_pred_random = np.random.randint(0, K, X.shape[0])

def count_errors(y_true, y_pred):
    error_count = 0
    number_of_pairs = 0
    for i, j in combinations(range(len(y_true)), 2): 
        if y_true[i] == y_true[j]:  
            number_of_pairs += 1
            if y_pred[i] != y_pred[j]: 
                error_count += 1
    return (error_count, number_of_pairs)

# silhouette_avg = silhouette_score(X, y_pred)
# print(f"Silhouette Score: {silhouette_avg}")

# ch_score = calinski_harabasz_score(X, y_pred)
# print(f"Calinski-Harabasz Index: {ch_score}")

# db_score = davies_bouldin_score(X, y_pred)
# print(f"Davies-Bouldin Index: {db_score}")

# Access the cluster labels
labels = kmeans.labels_

# Print the labels
print("Cluster Labels:", labels)

# Calculate proportions
unique, counts = np.unique(labels, return_counts=True)
proportions = counts / len(labels)

# Display results
for label, proportion in zip(unique, proportions):
    print(f"Label {label}: {proportion:.2f} (Proportion)")

# # Calculate proportions
# uniqueGMM, countsGMM = np.unique(y_pred_gmm, return_counts=True)
# proportionsGMM = counts / len(y_pred_gmm)

# # Display results
# for y_pred_gmm, proportion in zip(unique, proportionsGMM):
#     print(f"Label {y_pred_gmm}: {proportion:.2f} (Proportion)")

error_kmeans = count_errors(y_true, y_pred)
print("Error count - k-means: ", error_kmeans[0])
print("Total pairs: ", error_kmeans[1])
print("Procentage error K-means: ", error_kmeans[0]/error_kmeans[1])

error_gmm = count_errors(y_true, y_pred_gmm)
print("Error count - GMM:", error_gmm[0])
print("Total pairs: ", error_gmm[1])
print("Procentage error GMM: ", error_gmm[0]/error_gmm[1])

# error_random_baseline = count_errors(y_true, y_pred_random)
# print("Error count - random baseline:", error_random_baseline[0])
# print("Total pairs: ", error_random_baseline[1])
# print("Procentage error random baseline: ", error_random_baseline[0]/error_random_baseline[1])
