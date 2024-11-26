import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor


data_train = np.loadtxt("regression_train.txt")
data_test = np.loadtxt("regression_test.txt")

X_train = data_train[:,0].reshape(-1, 1)
y_train = data_train[:,1] 

x_test = data_test[:,0].reshape(-1, 1)
y_test = data_test[:,1] 

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

loss_func = nn.MSELoss()

learning_rate = 0.01

optimizer = optim.SGD(model.parameters(), lr=learning_rate)