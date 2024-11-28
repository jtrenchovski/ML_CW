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

