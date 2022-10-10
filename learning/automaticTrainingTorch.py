# 1) design model
# 2) construct loss and optimizer
# 3) training loop
#   - forward pass
#   - backward pass
#   - compute gradients
#   - update weights

import torch
import torch.nn as nn

x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8], [10]], dtype=torch.float32)

test = torch.tensor([6], dtype = torch.float32)

# model = nn.Linear(1, 1)

class LinearRegression(nn.Module):
    def __init__(self, inFeatures, outFeatures) -> None:
        super(LinearRegression, self).__init__()

        self.lin = nn.Linear(inFeatures, outFeatures)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(1, 1)

learnRate = 0.01
numEpochs = 100

# the loss of MSE
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learnRate)

for epoch in range(numEpochs):
    yPred = model(x)

    l = loss(y, yPred)

    l.backward()

    # update weights
    optimizer.step()

    # zero grad
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}. loss: {l:.3f}")

print(f"test: {model(test)}")
