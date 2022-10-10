import torch
import numpy as np
import torch.nn as nn 
def softmax(x, axis):
    return np.exp(x) / np.sum(np.exp(x), axis = axis)

x = np.array([1.0, 1.0, 0.1])
outputs = softmax(x, 0)

print(outputs)

tensor = torch.tensor([0.7, 0.2, 0.1])
outputs = tensor.softmax(dim = 0)

print(outputs)

# cross entropy
# no softmax when using cross entropy

def crossEntropy(label, prediction):
    return -np.sum(label * np.log(prediction))