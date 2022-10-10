from tokenize import String
from markupsafe import string
from torch.utils.data import Dataset
import torch
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, freshDirectory: string, rottenDirectory: string, transform = None):
        self.rottenImages = os.listdir(rottenDirectory)
        self.freshImages = os.listdir(freshDirectory)
        self.transform = transform
        self.size = len(self.rottenImages) + len(self.freshImages)

        self.freshDirectory = freshDirectory
        self.rottenDirectory = rottenDirectory
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= len(self.rottenImages):
            image = Image.open(self.freshDirectory +
                              self.freshImages[index - len(self.rottenImages)])
            item = self.transform(image)
            return item, 1
        else:
            image = Image.open(self.rottenDirectory + self.rottenImages[index])
            item = self.transform(image)
            return item, 0