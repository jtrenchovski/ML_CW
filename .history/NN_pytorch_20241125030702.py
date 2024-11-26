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

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(x_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
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

loss_fn = nn.MSELoss()

learning_rate = 0.001
epochs = 100

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(X, y, epochs, model, loss_fn, optimizer):
    for epoch in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    predictions = model(X_test)
    mse = loss_fn(predictions, y_test)
    print(f'\nTest Mean Squared Error: {mse.item():.4f}')