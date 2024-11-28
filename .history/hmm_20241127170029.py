import numpy as np
from hmmlearn import hmm

# Read the observed rewards from the file
observations = np.loadtxt('rewards.txt', dtype=int).reshape(-1, 1)

# Reshape observations for hmmlearn (n_samples, n_features)
observations = observations