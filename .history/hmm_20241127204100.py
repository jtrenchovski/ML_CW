import numpy as np
from hmmlearn import hmm

observations = np.loadtxt('rewards.txt', dtype=int).reshape(-1, 1)

n_hidd_states = 9  
n_emmisions = 3  # number of rewards

def get_neighbours(state):
    x, y = state // 3, state % 3  
    neighbours = []
    # Up neighbour
    if x > 0:
        neighbours.append((x-1)*3 + y)  
    # Lower
    if x < 2:
        neighbours.append((x+1)*3 +y)  
    # Left
    if y > 0:
        neighbours.append(x*3 + y - 1)
    # Right 
    if y < 2:
        neighbours.append(x*3 + y +1)  
    return neighbours

transmat_true = np.zeros((n_hidd_states, n_hidd_states))
for state in range(n_hidd_states):
    neighbours = get_neighbours(state)
    for neighbour in neighbours:
        transmat_true[state, neighbour] = 1/len(neighbours)

model = hmm.CategoricalHMM(n_components=n_hidd_states,n_features=3, n_iter=100, random_state=42)
model.fit(observations)

print("Start probabilities: ", model.startprob_)
print("Transition distribution: ", model.transmat_)
print("Emission distribution: ", model.emissionprob_)

transmat_diff = np.abs(model.transmat_-transmat_true)
average_trans_diff = np.mean(transmat_diff)

print("Mean difference between estimated and true probabilities: ", average_trans_diff)

model_known_trans = hmm.CategoricalHMM(n_components=n_hidd_states, n_features=3, n_iter=100, random_state=42, 
    params='se', init_params='se')
model_known_trans.transmat_ = transmat_true

model_known_trans.fit(observations)

print(model_known_trans.startprob_)
print(model_known_trans.transmat_)
print(model_known_trans.emissionprob_)

