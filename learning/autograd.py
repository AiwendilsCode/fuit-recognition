import torch

x = torch.ones(3, requires_grad = True) # requires_grad is necessary to train
print(x)

y = x * 2
print(y)


z = y * y + 2
z = z.sum()
print(z)

z.backward()
print(x.grad)

# three ways how to disable grads

x.requires_grad_(False) # first way

y = x.detach() # second way (detach() makes new tensor)

with torch.no_grad(): # third way
    x = x * x

# gradients are accumulating

weights = torch.ones(4, requires_grad = True)

for epoch in range(5):
    modelOutput = (weights * 2).sum()

    modelOutput.backward()
    print(weights.grad)
    
    # weights.grad.zero_() # clear the gradients