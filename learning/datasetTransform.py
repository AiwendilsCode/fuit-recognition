import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class CustomDataset(Dataset):

    def __init__(self, fileName, transform = None):
        rawData = np.loadtxt(fileName, delimiter=',',
                             dtype=np.float32, skiprows=1)
        self.data = rawData[:, 1:]
        self.labels = rawData[:, [0]]
        self.nSamples = self.data.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.data[index], self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.nSamples


class ToTensor:
    def __call__(self, sample):
        data, labels = sample
        return torch.from_numpy(data), torch.from_numpy(labels)

dataset = CustomDataset("./learning/data/wine.csv", transform = ToTensor())
raw = dataset[0]
data, labels = raw
print(type(data), type(labels))
