import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad = True)

y2 = w * x # simple linear function without bias
loss = (y2 - y) ** 2 # loss is only difference between predicted and desired squared

# gradient for w should be -2.0

loss.backward()

print(w.grad)