import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import config  # Import from config.py

class MNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape(config.resize_y, config.resize_x)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0  # Basic transform if none provided

        label = self.data.iloc[idx, 0]
        return img, label

#Anomalous dataset
class AnomalousDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx, :].values.astype(np.uint8).reshape(config.resize_y, config.resize_x)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0  # Basic transform if none provided

        return img
