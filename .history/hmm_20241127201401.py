import numpy as np
from hmmlearn import hmm

observations = np.loadtxt('rewards.txt', dtype=int).reshape(-1, 1)

n_hidd_states = 9  
n_emmisions = 3  # number of rewards

def get_neighbors(state):
    x, y = state // 3, state % 3  # Convert state index to x, y
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
model = hmm.MultinomialHMM(n_components=n_hidd_states, n_iter=100, random_state=42)

# Fit the model to the observations
model.fit(observations)

print("Start probabilities: ", model.startprob_)
print("Transition distribution: ", model.transmat_)
print("Emission distribution: ", model.emissionprob_)

transmat_diff = np.abs(model.transmat_ - transmat_true)
average_trans_diff = np.mean(transmat_diff)

print("Average absolute difference between estimated and true transition probabilities: ", average_trans_diff)

model_known_trans = hmm.MultinomialHMM(
    n_components=n_hidd_states, n_iter=100, random_state=42,
    params='se', init_params='se'
)
model_known_trans.transmat_ = transmat_true

# Fit the model to the observations
model_known_trans.fit(observations)

print(model_known_trans.startprob_)
print(model_known_trans.transmat_)
print(model_known_trans.emissionprob_)

