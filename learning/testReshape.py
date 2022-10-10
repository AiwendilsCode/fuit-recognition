import torch

tensor = torch.tensor([[1,2],[3,4]])

tensor = tensor.reshape(1, -1)

print(tensor)