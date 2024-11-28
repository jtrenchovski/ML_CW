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

for s in range(n_hidd_states):
    neighbors = get_neighbors(s)
    n = len(neighbors)
    for neighbor in neighbors:
        transmat_true[s, neighbor] = 1 / n