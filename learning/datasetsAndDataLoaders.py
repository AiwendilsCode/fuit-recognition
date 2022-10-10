# 1) design model
# 2) construct loss and optimizer
# 3) training loop
#   - forward pass
#   - backward pass
#   - compute gradients
#   - update weights

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class CustomDataset(Dataset):

    def __init__(self, fileName):
        rawData = np.loadtxt(fileName, delimiter = ',', dtype = np.float32, skiprows = 1)
        self.data = torch.from_numpy(rawData[:, 1:])
        self.labels = torch.from_numpy(rawData[:, [0]])
        self.nSamples = self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.nSamples

dataset = CustomDataset("./learning/data/wine.csv")
firstItem = dataset[0]
data, label = firstItem
print(data, label)

batchSize = 5

dataloader = DataLoader(dataset = dataset, batch_size = batchSize, shuffle = True, num_workers = 0)

dataIter = iter(dataloader)
examples = dataIter.next()
data, labels = examples

print(data, labels)

# train loop

nEpochs = 2
totalSamples = len(dataset)
nIterations = math.ceil(totalSamples / batchSize)

for epoch in range(nEpochs):

    for i, (data, labels) in enumerate(dataloader):

        if ((i + 1) % 5) == 0:
            print(f"Epoch {epoch + 1}/{nEpochs}, step {i + 1}/{nIterations}, data {data.shape} ")