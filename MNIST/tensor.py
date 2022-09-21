import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

arangeTensor = torch.arange(0, 10, 2)
linspaceTensor = torch.linspace(0, 10, 2)
emptyNormalTensor = torch.empty(size = (1, 5)).normal_(mean = 1, std = 2)
emptyUniformTensor = torch.empty(size = (1, 5)).uniform_(0, 2)
diagTensor = torch.diag(torch.ones(4).uniform_(0, 1))

justTensor = torch.from_numpy(np.ones((4, 5)))

print(justTensor)


# tensor math and comparision

x = torch.tensor([1, 2, 3], dtype = torch.int16)
y = torch.tensor([9, 8, 7], dtype = torch.int16)

# addition
z = torch.add(x, y)
torch.add(x, y, out = z)
z = x.add(y)
z = x + y

# subtract
z = x - y

# division
z = torch.true_divide(x, y)

# inplace operations
# '_' mean inplace
x.add_(y)
x += y # not the same as x = x + y (result is same but approach is different in second case is one another copying)

# exponentiation
z = x.pow(2)
z = x ** 2

# simple comparision
z = x > 0 # bool tensor

# matrix multiplication
x2 = torch.rand((2, 5))
y2 = torch.rand((5, 3))
z = torch.mm(x2, y2)
z = x2.mm(y2)

# matrix exponentiation
x3 = torch.rand(5, 5)
x3.matrix_power(2) # multiply himself with himself

# element wise multiplication
z = x * y

# dot product
z = torch.dot(x, y)

# batch matrix multiplication
x4 = torch.rand((32, 10, 20))
y3 = torch.rand((32, 20, 30))
zBMM = torch.bmm(x4, y3) # (32, 10, 30)

# broadcasting
x1 = torch.rand((5, 5))
y1 = torch.rand((1, 5))
z = x1 - y1 # mathematically impossible but in pytorch every row is subtracted with y1
z = x1 ** y1

# other useful tensor operations
sumX = torch.sum(x)
value, indexes = torch.max(x, dim = 0)
values, indexes = torch.min(x, dim = 0)
absX = torch.abs(x)
argMaxX = torch.argmax(x, dim = 0) # only returns index
argMinX = torch.argmin(x, dim = 0)
meanX = torch.mean(x.float(), dim = 0)
z = torch.eq(x, y) # bool tensor true if elements are same
sortedY, swappedIndexes = torch.sort(y, dim = 0, descending = False)

z = torch.clamp(y, min = 0) # every element which is less than zero set to zero

x5 = torch.tensor([1, 0, 1, 1, 1, 0], dtype = torch.bool)
z = torch.any(x, dim = 0) # return true if any of the values are true
z = torch.all(x, dim = 0) # return true if every value is true


#
# tensor indexing
#

batchSize = 10
features = 25

x = torch.rand((batchSize, features))

print(x[0].shape) # x[0, :]
print(x[:, 0].shape)

print(x[2, 0 : 10]) # returns list of thrid example

# fancy indexing
x = torch.arange(10)
indexes = [2, 5, 8]
print(x[indexes])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)

# more advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[(x % 2 == 0)])

# useful operations

print(torch.where(x > 5, x, x * 2)) # if condition is not satisfied we move to another statement in this example x * 2
print(torch.where(x > 5, x, x * 2).unique())

print(x.ndimension()) # return number of dimensions
print(x.numel()) # count of elements

#
# tensor reshape
#

x = torch.arange(9)

x3X3 = x.view(3, 3)
print(x3X3)
x3X3 = x.reshape(3, 3) # not the same

xT3X3 = x3X3.t() # transpose
print(xT3X3.contiguous().view(9))

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))

print(torch.cat((x1, x2), dim = 0).shape) # concatenate
print(torch.cat((x1, x2), dim = 1).shape)

z = x1.view(-1)
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1, 3)
print(x.shape)

x = torch.arange(10)
print(x.unsqueeze().shape)
