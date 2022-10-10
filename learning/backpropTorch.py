import torch

x = torch.tensor([1, 2, 3, 4, 5], dtype = torch.float32)
y = torch.tensor([2, 4, 6, 8, 10], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)


def forward(x):
    return x * w

# the loss of MSE
def loss(y, Ypred):
    return ((Ypred - y) ** 2).mean()

learnRate = 0.01
numEpochs = 100

for epoch in range(numEpochs):
    yPred = forward(x)

    l = loss(y, yPred)

    l.backward()

    # update weights
    with torch.no_grad():
        w -= learnRate * w.grad

    # zero grad
    w.grad.zero_()

    print(f"Epoch {epoch + 1}. loss: {l:.3f} weight: {w:.4f}")
