import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA

XFull, yFull = fetch_covtype(return_X_y=True, download_if_missing=True)

K = 7
N = 10000
X = XFull[:N]
y_true = yFull[:N]

scaler = StandardScaler(with_mean=False, with_std=False).fit(X)
X_scaled = scaler.transform(X)

min_max_scaler = MinMaxScaler().fit(X)
X_scaled_MM = min_max_scaler.transform(X)

pca = PCA(n_components=49)

X_pca = pca.fit_transform(X_scaled)
print(pca.n_components_)

X_pca_gmm = pca.fit_transform(X)
print(pca.n_components_)

power_scaler = PowerTransformer().fit(X_pca)
X_scaled_power = power_scaler.transform(X_pca)

scaler = StandardScaler().fit(X_pca)
X_final = scaler.transform(X_pca)
# without std the results get worse by 7%. Without mean around 3 maybe 
# not even that. There are probably some outliers

feature_names = fetch_covtype(download_if_missing=True).feature_names
# print(X.shape)
# print(X_scaled[:5, :10])
# print(feature_names)
initial_centroids = np.array([
    [8.69663962e-01, 4.10652184e-01, -3.78985290e-02, 6.38204773e-01, 3.41260297e-01, -2.83594180e-01, -4.79304759e-01, -2.02185763e-01, -2.28009756e-01, -1.82891628e-01, -1.31541843e-01, 7.38313935e-02, -1.90529122e-01, 3.70405873e-01, 7.44154259e-02, 1.45213886e-01, 8.72765955e-02, 1.91387928e-01, -3.49010072e-01, 2.50215859e-02, 1.31151589e-01, 5.00744142e-01, -6.47174558e-02, -1.76845037e-02, -4.56334738e-02, 2.82145277e-02, -1.40548762e-01, 3.70007001e-02, -1.43680405e-01, -2.30872243e-01, 6.29312555e-03, 1.26075329e-02, 4.10284113e-02, -7.92078302e-02, -9.27813047e-02, -1.76441735e-02, 1.05778152e-01, 4.18318077e-01, 3.18613182e-01, -3.83109577e-02, -4.44438471e-01, 1.20177713e-01, 2.30980041e-02, -7.73362745e-02, 5.14933181e-02, -2.19452026e-02, -1.97223922e-01, -1.14938578e-01, 3.09100671e-04],
    [-3.68767423e-01, -5.20750056e-02, 9.67395489e-03, -7.37419736e-02, -1.02789495e-01, 6.64251489e-02, 1.08637702e-01, 1.77616154e-02, 2.05091369e-01, 1.63771321e-01, -4.78232181e-03, -2.11265583e-01, 2.27927657e-01, -4.71677330e-03, -2.23157701e-01, -1.71934112e-02, -5.15504148e-02, -4.88590961e-02, 2.13650385e-01, 3.28979651e-02, -6.09567732e-02, -1.23939338e-01, 1.21226558e-01, 1.13579003e-03, 1.65011655e-02, -9.10071781e-02, -2.62453851e-02, -4.87432313e-02, 5.76027822e-02, 8.73225774e-02, -3.39214740e-02, 3.75632378e-02, -3.26771891e-02, 3.78954615e-02, 3.71201526e-02, 5.73478881e-03, -1.20854762e-02, -8.13155695e-02, -6.40820070e-02, -6.65713542e-02, 1.60861044e-01, -1.40546118e-03, -2.28922240e-03, -2.83148414e-02, -3.43718353e-02, -1.11574528e-02, 6.41165578e-02, 4.41316140e-02, -6.88042490e-05],
    [9.48538953e-01, -5.57800322e-04, 1.33575005e-01, -9.18921348e-01, -6.77491700e-01, 1.31052315e-01, -1.64997960e+00, -1.62117357e+00, -7.86526726e-01, 2.54173891e-01, -3.66519333e-01, -6.68042324e-01, -2.04056830e-01, -1.58421067e-01, -1.02567245e-01, 3.67710929e-02, -4.82617263e-01, -6.28653663e-01, 1.74140529e+00, -2.37926210e-01, -2.31646092e+00, -2.21965915e+00, -2.11043357e+00, -9.49252952e-02, -1.38339748e+00, 7.22076686e+00, 3.44409936e+00, 6.49436613e+00, -6.67012574e-01, 7.88361832e-01, -8.33331398e-01, -1.80877817e+00, 5.07035331e-01, -6.26937047e-02, 4.13051663e-02, 1.26468620e-01, -1.14796419e+00, 7.35864378e-01, -7.11297983e-01, 6.18432227e-01, 6.00923231e-02, 5.36796902e-02, 3.65968039e-01, 4.01384753e-01, -7.36100257e-02, -2.06691452e-02, -6.80315516e-02, -2.45735010e-01, -3.36760560e-05],
    [8.49594321e-01, -4.15173911e-01, 4.87600934e-01, -1.23226117e+00, 2.53577603e-01, -1.20876197e+00, -1.87933343e-01, 1.54950948e+00, -2.58922032e-01, 3.58645403e-02, 8.40820651e-01, -2.35601811e-01, 2.53462663e-01, 3.42359738e-01, -1.56610949e-01, 2.42386367e-01, 1.32164672e-01, 2.28256688e-02, 4.86157151e-01, 2.15385416e-01, 9.39393822e-01, -1.79712722e+00, -6.57876357e-01, -5.26070679e-03, -3.61351273e-01, -1.38761694e+00, 1.15888905e+01, -5.19774924e+00, -1.67028784e+00, -3.92647609e+00, 6.06194302e+00, -1.50478349e+00, 2.44770020e+00, -7.06352155e-01, -2.80671431e-01, -4.22308331e-01, -8.34780499e-01, 6.97719585e-01, -4.87768643e-02, 6.93139303e-01, -7.31477839e-01, -6.48572210e-01, 3.45346079e-02, 5.76863818e-01, 2.02680294e-01, -1.01926843e-01, 1.44901791e-03, 2.32048197e-02, -1.20068954e-03],
    [7.81213839e-01, -6.50429331e-01, 7.94656118e-01, -1.04948804e+00, 1.10367222e-01, 2.87185553e-01, 1.03260375e+00, -9.77775560e-02, -1.02176117e+00, -1.19898796e+00, 1.39431273e+00, -2.88160807e-01, -1.76390064e+00, -1.69010070e+00, 5.01365632e-01, -8.31741998e-01, 1.04232527e+00, -8.68127005e-01, -2.47057384e-01, -2.11799010e-01, -2.59617343e-01, 2.92012646e-01, 1.68155996e-01, -1.08757888e-01, 1.22956264e-01, 2.36029416e-02, -3.38874432e-01, -1.05620431e-01, 8.10676784e-02, 1.70263213e-02, 6.01597613e-02, -1.87211481e-01, -6.90498591e-02, -2.77510635e-02, 2.31279722e-03, 2.84824330e-02, -3.55080981e-01, -3.99376694e-01, 4.14372063e-01, 6.12855951e-01, 5.32661102e-02, -4.39321283e-01, 2.09134068e-02, 5.32045093e-01, 1.48717964e-01, 9.82378179e-02, -7.41660708e-03, -8.55037616e-02, -4.90769316e-04],
    [-5.84060440e-01, -5.65449467e-01, -1.13544988e+00, -1.04827912e+00, -2.44844938e-02, -6.29430819e-01, -2.82391985e-01, 4.67731082e-02, -1.74676710e-01, 1.57177734e-01, -5.60510794e-01, 2.58758689e+00, 2.55157235e-03, -6.13354676e-01, 2.59061189e+00, -1.85416802e-01, -6.08252165e-01, 4.13443733e-01, -7.02437269e-01, 3.38436767e-01, 5.80501997e-01, -4.76152260e-01, -5.62856447e-01, -4.20378514e-01, -3.71482078e-01, -1.73806375e-01, -1.73323105e-01, 5.90723747e-02, -6.71277945e-02, 1.61549220e-01, -9.58093613e-03, -6.67214060e-02, 9.87625833e-03, -3.53990925e-03, -1.97368906e-02, -3.23827877e-02, 3.81494637e-01, 3.52920475e-01, -1.12304276e+00, 6.32484155e-01, 1.60869814e-01, 5.64434760e-03, -1.05316019e-01, -2.53581008e-02, 2.12899719e-02, 1.52513277e-01, 1.78695557e-01, 1.34609370e-01, -4.92412671e-04],
    [5.60336976e-01, -3.45820926e-01, 6.22790692e-01, 3.76211402e-01, -1.04661222e+00, 2.81623420e+00, 1.10640197e+00, 3.01922944e+00, -9.10461866e-01, -7.65294101e-01, -8.30830813e-01, 1.86200200e+00, -9.32267048e-01, 1.77153342e+00, -6.19061848e-01, 1.71072543e+00, -7.13569421e-01, 1.09357527e+00, -1.82135045e+00, -2.08164100e+00, 5.71355920e-01, -6.79820400e-01, -2.03611787e+00, 1.89545768e+00, 1.44241584e+00, 9.71580638e-01, 2.24533203e-01, 1.30092853e-01, 4.89795034e-01, 7.98113246e-02, -8.96849684e-02, 2.50959531e-01, 5.81326048e-02, -5.67426307e-02, -2.61656891e-02, 8.97311673e-02, -3.16076746e-01, -3.40834710e+00, -8.42385209e-02, -8.87743204e-01, -3.81924991e-01, -2.43848538e-01, -2.16447062e-01, 4.08517796e-01, 9.02695972e-02, 1.00800568e-01, -1.05985580e-01, -8.19309258e-02, 1.51365260e-03]
])


kmeans = KMeans(K, init=initial_centroids, n_init=10).fit(X_final)
y_pred = kmeans.predict(X_final)

# print(kmeans.cluster_centers_)

# 60%(53% best) - 7 cluster (35, 40 with normalisation after pca)
# 63% - 6 clusters
# 57% - 5 clusters

# kmeans is the best initialisation method. Other give poor result. This is ran 
# on not normalised dimension pca reduced data. Best us 51 and i think it was with the parameters mentioned
# above
gmm = GaussianMixture(n_components=K, n_init=10, covariance_type='tied').fit(X_final)
y_pred_gmm = gmm.predict(X_final)

# np.random.seed(0)
# y_pred_random = np.random.randint(0, K, X_scaled.shape[0])

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
# print(error_kmeans[0])
print("Procentage error K-means: ", error_kmeans[0]/error_kmeans[1])

error_gmm = count_errors(y_true, y_pred_gmm)
# print(error_kmeans[0])
print("Procentage error GMM: ", error_gmm[0]/error_gmm[1])

# error_random_baseline = count_errors(y_pred_random, y_true)
# print("Procentage error random baseline: ", error_random_baseline[0]/error_random_baseline[1])
