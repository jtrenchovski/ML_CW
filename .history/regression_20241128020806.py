import numpy as np
import torch
import pymc as pm
import arviz as az

from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch import nn
from torch import optim

data_train = np.loadtxt("regression_train.txt")
data_test = np.loadtxt("regression_test.txt")

X_train = data_train[:,0].reshape(-1, 1)
y_train = data_train[:,1] 

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_train)

x_test = data_test[:,0].reshape(-1, 1)
y_test = data_test[:,1] 

poly2 = PolynomialFeatures(degree=3)
X_poly_test = poly2.fit_transform(x_test)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y_train)

RMSE_train = np.sqrt(mean_squared_error(y_train, lin_reg.predict(X_poly)))
print('Linear Regression MSE (train data): ', round(RMSE_train,2))
RMSE_test = np.sqrt(mean_squared_error(y_test, lin_reg.predict(X_poly_test)))
print('Linear Regression MSE (test data): ', round(RMSE_test,2))

X_train = data_train[:,0].reshape(-1, 1) 
y_train = data_train[:,1].reshape(-1, 1)

X_test = data_test[:,0].reshape(-1, 1) 
y_test = data_test[:,1].reshape(-1, 1)

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(X_train)

poly2 = PolynomialFeatures(degree=3)
x_poly_test = poly2.fit_transform(X_test)

X_train = torch.from_numpy(x_poly).float()
X_test = torch.from_numpy(x_poly_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
loss_fn = nn.MSELoss()
learning_rate = 0.0001
optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)

epochs = 500
losses = []

def train_loop(X_train, y_train, epochs, model, loss_fn, optimizer):
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()  
            predictions = model(X_train)  
            loss = loss_fn(predictions, y_train)  
            loss.backward()  
            print("Epoch: ", epoch, ", Loss: ", loss.item())
            losses.append(loss.item())
            return loss
        optimizer.step(closure)
        optimizer.zero_grad()
        
def test_model(X_test, y_test, model, loss_fn):
    with torch.no_grad():
        predictions = model(X_test)
        mse = loss_fn(predictions, y_test)
        print("Neural Network MSE (test data): ", mse.item())

train_loop(X_train, y_train, epochs, model, loss_fn, optimizer)
test_model(X_test, y_test, model, loss_fn)

n_samples = 1100
model = pm.Model()

with model:
    w0 = pm.Normal('w0', mu=0, sigma=20)
    w1 = pm.Normal('w1', mu=0, sigma=20)
    w2 = pm.Normal('w2', mu=0, sigma=20)
    w3 = pm.Normal('w3', mu=0, sigma=20)
    sigma = pm.Uniform('sigma', lower=0, upper=20)

    y_est = w0 + w1*X_train + w2*(X_train**2) + w3*(X_train**3) 

    likelihood = pm.Normal('y_train', mu=y_est, sigma=sigma, observed=y_train)
    sampler = pm.NUTS() 
    idata = pm.sample(n_samples, step=sampler, progressbar=True)

n_burn_in = 100
# Delete the burn-in
idata.sel(draw=slice(n_burn_in, None))

# Print summary statistics of the posterior distributions
summary = pm.summary(posterior)
print(summary)
