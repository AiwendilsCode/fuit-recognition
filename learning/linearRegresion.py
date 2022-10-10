# 1) design model
# 2) construct loss and optimizer
# 3) training loop
#   - forward pass
#   - backward pass
#   - compute gradients
#   - update weights

import torch
import torch.nn as nn 
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare data

Xnumpy, Ynumpy = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1)

x = torch.from_numpy(Xnumpy.astype(np.float32))
y = torch.from_numpy(Ynumpy.astype(np.float32))

y = y.view(y.shape[0], 1)

nSamples, nFeatures = x.shape

# model
inputSize = nFeatures
outputSize = 1

class Model(nn.Module):
    def __init__(self, inputSize, outputSize) -> None:
        super(Model, self).__init__()
        self.fc1 = nn.Linear(inputSize, 1)
        self.fc2 = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.fc2(self.fc1(x))

model = Model(inputSize, outputSize)
# loss and optimizer

learnRate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learnRate)

# train loop

numEpochs = 100

for epoch in range(numEpochs):
    yPred = model.forward(x)

    loss = criterion(yPred, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    print(F"Epoch {epoch + 1}. loss = {loss.item():.4f}")

# plot
predicted = model.forward(x).detach().numpy()
plt.plot(Xnumpy, Ynumpy, 'ro')
plt.plot(Xnumpy, predicted, 'b')
plt.show()



