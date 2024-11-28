import numpy as np
from hmmlearn import hmm

observations = np.loadtxt('rewards.txt', dtype=int).reshape(-1, 1)

n_hidd_states = 9  # 3x3 grid
n_emmisions = 3  # rewards 0,1,2

def get_neighbors(state):
    x, y = divmod(state, 3)  # Convert state index to x, y
    neighbors = []
    if x > 0:
        neighbors.append((x - 1) * 3 + y)  # Up
    if x < 2:
        neighbors.append((x + 1) * 3 + y)  # Down
    if y > 0:
        neighbors.append(x * 3 + y - 1)  # Left
    if y < 2:
        neighbors.append(x * 3 + y + 1)  # Right
    return neighbors

# Build the true transition matrix
transmat_true = np.zeros((n_hidd_states, n_hidd_states))

# Transition probabilites 
for s in range(n_hidd_states):
    neighbors = get_neighbors(s)
    n = len(neighbors)
    for neighbor in neighbors:
        transmat_true[s, neighbor] = 1 / n

# Initialize the HMM
model = hmm.CategoricalHMM(n_components=n_hidd_states, n_iter=100, random_state=42)

# Fit the model to the observations
model.fit(observations)

# Get the estimated parameters
estimated_startprob = model.startprob_
estimated_transmat = model.transmat_
estimated_emissionprob = model.emissionprob_

print(estimated_startprob)
print(estimated_transmat)
print(estimated_emissionprob)

# Compare estimated transition matrix with the true transition matrix
transmat_diff = np.abs(estimated_transmat - transmat_true)
average_trans_diff = np.mean(transmat_diff)

print("Average absolute difference between estimated and true transition probabilities:")
print(average_trans_diff)

# Initialize the HMM model with known transition probabilities
model_known_trans = hmm.MultinomialHMM(
    n_components=n_hidd_states, n_iter=100, random_state=42,
    params='se', init_params='se'
)
model_known_trans.transmat_ = transmat_true

# Fit the model to the observations
model_known_trans.fit(observations)

# Extract the estimated parameters
estimated_startprob_known_trans = model_known_trans.startprob_
estimated_transmat_known = model_known_trans.transmat_
estimated_emissionprob_known_trans = model_known_trans.emissionprob_

print(estimated_startprob)
print(estimated_transmat_known)
print(estimated_emissionprob_known_trans)

