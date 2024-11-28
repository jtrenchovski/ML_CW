import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pymc as pm
import arviz as az

data_train = np.loadtxt("regression_train.txt")
data_test = np.loadtxt("regression_test.txt")

X_train = data_train[:,0].reshape(-1, 1)
y_train = data_train[:,1] 

x_test = data_test[:,0].reshape(-1, 1)
y_test = data_test[:,1] 

num_samples = 1000
model = pm.Model()

with model:
    # Defining our priors
    w0 = pm.Normal('w0', mu=0, sigma=20)
    w1 = pm.Normal('w1', mu=0, sigma=20)
    w2 = pm.Normal('w2', mu=0, sigma=20)
    w3 = pm.Normal('w3', mu=0, sigma=20)
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    y_est = w0 + w1*X_train + w2*pow(X_train, 2) + w3*pow(X_train,3) # auxiliary variables

    likelihood = pm.Normal('y_train', mu=y_est, sigma=sigma, observed=y_train)
    
    # inference
    sampler = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler 
    # or alternatively
    # sampler = pm.Metropolis()
    
    idata = pm.sample(num_samples, step=sampler, progressbar=True)


# Generate posterior predictive samples
with model:
    posterior_predictive = pm.sample_posterior_predictive(idata, progressbar=True)

# Visualize results
plt.figure(figsize=(10, 6))

# Plot true training data
plt.scatter(X_train, y_train, label="True Data", color="blue", alpha=0.6)

# Plot posterior predictive mean
posterior_mean = posterior_predictive["y_train"].mean(axis=0)
plt.plot(X_train, posterior_mean, label="Mean Prediction", color="red")

