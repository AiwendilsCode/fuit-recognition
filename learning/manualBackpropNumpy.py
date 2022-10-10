import numpy as np

# f = w * x
# f = 2 * x - 2 is value which we want to get

x = np.array([1, 2, 3, 4], dtype = np.float32) # input
y = np.array([2, 4, 6, 8], dtype = np.float32) # desired output

w = 0.0

def forward(x):
    return x * w

# the loss of MSE
def loss(y, Ypred):
    return ((Ypred - y) ** 2).mean()

# MSE = 1/N * (w * x - y) ** 2
# derivative of MSE = 1/N * 2 * x * (w * x - y)

def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y)

print(f"output before training: f(5) {forward(5):.3f}")

# learning

learnRate = 0.01
numEpochs = 20

for epoch in range(numEpochs):
    yPred = forward(x)

    l = loss(y, yPred)

    grad = gradient(x, y, yPred)

    # update weights
    w -= learnRate * grad

    print(f"Epoch {epoch + 1}. loss: {l:.3f} weight: {w:.4f}")

print(f"output after training: f(5) {forward(5):.3f}")
