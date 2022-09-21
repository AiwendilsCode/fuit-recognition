import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        # if stride is not zero new size of input is [same, old / stride, old / stride]

        # convolutional part (input dims [1, 28, 28])
        self.conv1 = nn.Conv2d(1, 5, kernel_size = 3) # params (in_channels, out_channels) 
        self.conv2 = nn.Conv2d(5, 10, kernel_size = 3)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)

        # fully connected part
        self.lin1 = nn.Linear(10 * 7 * 7, 15) # params (in_features, out_features)
        self.lin2 = nn.Linear(15, 10)

    def forward(self, x):
        # convolutional part
        x = self.pool(nn.ReLU(self.conv1(x)))
        x = self.pool(nn.ReLU(self.conv2(x)))

        # convert input to dimension [1, size of input to fully connected layer]
        x = x.view(-1, 10 * 7 * 7) # size -1 is inferred from other dimension

        # fully connected part
        x = nn.Tanh(self.lin1(x))
        x = nn.Softmax(self.lin2(x))

        return x
