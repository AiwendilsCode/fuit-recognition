from pyparsing import string
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import csv

class CustomDataset(Dataset):
    def __init__(self, csvFile: string, train: bool):
        imagesRaw = self.readCsvToArray(csvFile)
        self.labels, self.images = self.getLabels(imagesRaw)
        self.train = train

    def __getitem__(self, index: int):
        tensorImage = torch.tensor(self.images[index], dtype = torch.float32)
        return ( tensorImage.reshape((1, 28, 28)), self.labels[index] )

    def __len__(self):
        return len(self.labels)

    def readCsvToArray(self, file: string):
        """
        params:
        file (string) = csv file with images and labels
        return content of file in list
        """
        images = []

        with open(file, mode='r') as file:
            csvFile = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
            for row in csvFile:
                images.append(row)

        return images

    def getLabels(self, raw: list):
        """
        params:
        raw (list of lists) = raw content of csv file with labels on first position
        first return labels then images
        """
        labels = []

        for i in range(0, len(raw)):
            labels.append(raw[i][0])
            raw[i].pop(0)

        return labels, raw