import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import customDataset as myData
import torch.onnx as onnx
import cv2 as cv

# fully connected network
class NN(nn.Module):
    def __init__(self, inputSize, classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(inputSize, 50)
        self.fc2 = nn.Linear(50, 15)
        self.fc3 = nn.Linear(15, classes)

    def forward(self, x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        return x

testAccuracy = 0.0

def checkAccuracy(loader, model):
    numCorrect = 0
    numSamples = 0

    # always before test
    model.eval()
    if loader.dataset.train:
        print("Checking accuracy on training data.")
    else:
        print("Checking accuracy on testing data.")

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)

            data = data.reshape(data.shape[0], -1)

            scores = model(data)

            predictions = scores.argmax(dim=1)

            numCorrect += (predictions == labels).sum()
            numSamples += predictions.size(0)

        testAccuracy = numCorrect / numSamples * 100
        print(f"{numCorrect} | {numSamples} with accuracy: {testAccuracy} %")

    model.train()
    return testAccuracy


def Convert_ONNX(model, input_size):
    print("Converting to ONNX.")

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, input_size, requires_grad=True)

    # Export the model
    onnx.export(model,         # model being run
                      # model input (or a tuple for multiple inputs)
                      dummy_input,
                      "MNIST.onnx",       # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],   # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},    # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# global variables
inputSize = 784
numClasses = 10
learnRate = 0.0005
batchSize = 64
numEpochs = 4

# datasets and data loaders
# trainDataset = datasets.MNIST(root = 'MNIST/dataset/', train = True, transform = transforms.ToTensor(), download = True)
# trainLoader = DataLoader(trainDataset, batchSize, shuffle = True)

# testDataset = datasets.MNIST('MNIST/dataset/', train = False, transform = transforms.ToTensor(), download = True)
# testLoader = DataLoader(testDataset, batchSize, shuffle = True)

trainDataset = myData.CustomDataset("MNIST/csvData/mnist_train.csv", train = True)
trainLoader = DataLoader(trainDataset, batchSize, shuffle = True)

testDataset = myData.CustomDataset("MNIST/csvData/mnist_test.csv", train = False)
testLoader = DataLoader(testDataset, batchSize, shuffle = True)

# init model
model = NN(inputSize, numClasses).to(device)

# loss and optimizer
criterium = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learnRate)

# train network
for epoch in range(numEpochs):
    print(f"{epoch + 1}. epoch")

    for batchIdx, (data, targets) in enumerate(trainLoader):

        # send to device
        data = data.to(device)
        targets = targets.to(device).to(torch.long)

        # get to correct shape
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterium(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    
    checkAccuracy(trainLoader, model)
    testAccuracy = checkAccuracy(testLoader, model)

if testAccuracy.item() >= 95:
    Convert_ONNX(model, 784)