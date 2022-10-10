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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# prepare data
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

nSamples, nFeatures = x.shape

trainData, testData, trainLabels, testLabels = train_test_split(x, y, test_size = 0.2, random_state = 1234)

# scale
sc = StandardScaler()
trainData = sc.fit_transform(trainData)
testData = sc.fit_transform(testData)

trainData = torch.from_numpy(trainData.astype(np.float32))
testData = torch.from_numpy(testData.astype(np.float32))

trainLabels = torch.from_numpy(trainLabels.astype(np.float32))
testLabels = torch.from_numpy(testLabels.astype(np.float32))

trainLabels = trainLabels.view(trainLabels.shape[0], 1)
testLabels = testLabels.view(testLabels.shape[0], 1)

# model

class LogisticRegression(nn.Module):
    def __init__(self, numInput):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(numInput, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegression(nFeatures)

# loss and optimizer

learnRate = 0.1

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learnRate)

# train loop
nEpochs = 100

for epoch in range(nEpochs):

    yPred = model(trainData)
    loss = criterion(yPred, trainLabels)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}. loss = {loss}")

# evaluation

with torch.no_grad():
    yPred = model(testData)
    yPred.round_()

    acc = (yPred.eq(testLabels).sum() / (nSamples * 0.2)) * 100

    print(f"test accuracy: {acc:.4f} %")