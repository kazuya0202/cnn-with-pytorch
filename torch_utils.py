import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_num, transform=None):
        self.transform = transform

        self.data_num = data_num
        self.data = []
        self.label = []

        for x in range(self.data_num):
            self.data.append(x)
            self.label.append(x)

    def __getitem__(self, idx):
        # img = None
        # label = None

        img = self.data[idx]
        label = self.label[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return self.data_num
