import torch
from torchvision import transforms
import torchvision.models as models
import cv2 as cv
import matplotlib.pyplot as plt
from myDataset import MyDataset
from torch.utils.data import DataLoader
import torch.onnx as onnx

def train(model, numEpochs, DataLoader, optimizer, lossFun, device):
    model.requires_grad_(True)
    for epoch in range(numEpochs):
        for i, (images, labels) in enumerate(DataLoader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = lossFun(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                    f'Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item():.4f}')

def test(model, dataLoader):
    model.eval()
    model.requires_grad_(False)
    numCorrect = 0
    numSamples = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataLoader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            predictions = torch.argmax(outputs, 1)

            numCorrect += (predictions == labels).sum()
            numSamples += predictions.size(0)

    testAccuracy = numCorrect / numSamples
    print(f"{numCorrect} | {numSamples} with accuracy: {testAccuracy} %")

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

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load mobilenet different versions
mobileNet = models.mobilenet_v2(pretrained = False)
#mobileNet = models.mobilenet_v3_small(pretrained = False)
#mobileNet = models.mobilenet_v3_large(pretrained = False)

mobileNet.requires_grad_(True)

# main variables
numEpochs = 1
batchSize = 15
learnRate = 0.001

# optimizers
optimizer = torch.optim.Adam(mobileNet.parameters(), learnRate)
#optimizer = torch.optim.SGD(mobileNet.parameters(), learnRate)

# loss
#criterion = torch.nn.BCELoss()
criterion = torch.nn.CrossEntropyLoss()

# datasets and dataloaders
trainDataset = MyDataset("D:\\datasets\\rotten fruits\\fruits for brano\\freshImagesOriginalTrain\\",
                         "D:\\datasets\\rotten fruits\\fruits for brano\\rottenImagesOriginalTrain\\", transform = preprocess)
testDataset = MyDataset(
    "D:\\datasets\\rotten fruits\\fruits for brano\\freshImagesOriginalTest\\",
    "D:\\datasets\\rotten fruits\\fruits for brano\\rottenImagesOriginalTest\\", transform = preprocess)

trainDataLoader = DataLoader(trainDataset, batchSize, shuffle = True)
testDataLoader = DataLoader(testDataset, batchSize, shuffle = True)

train(mobileNet, numEpochs, trainDataLoader, optimizer, criterion, device)
test(mobileNet, testDataLoader)

