import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import train_csv, test_csv

class MNISTTrainDataset(Dataset):
    """Loads unlabelled MNIST training data from CSV and normalizes to [0,1]."""
    def __init__(self, csv_file=None):
        super().__init__()
        try:
            self.df = pd.read_csv(csv_file or train_csv, index_col=False)
            print(f"Loaded training dataset with {len(self.df)} samples")
        except FileNotFoundError:
            raise FileNotFoundError(f"Training data file not found at {csv_file or train_csv}. Please check the path.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Drop label if present, normalize pixel values
        row = self.df.iloc[idx]
        if 'label' in row.index:
            row = row.drop('label')
        pixels = row.values.astype('float32') / 255.0
        return torch.from_numpy(pixels)

class MNISTAnomalyDataset(Dataset):
    """Loads labelled anomaly test set from CSV for evaluation."""
    def __init__(self, csv_file=None):
        super().__init__()
        try:
            self.df = pd.read_csv(csv_file or test_csv, index_col=0)
            print(f"Loaded anomaly dataset with {len(self.df)} samples")
        except FileNotFoundError:
            raise FileNotFoundError(f"Anomaly data file not found at {csv_file or test_csv}. Please check the path.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        pixels = row.drop('label').values.astype('float32') / 255.0
        return torch.from_numpy(pixels), label

# Convenience functions for DataLoader creation

def get_train_dataloader(batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True):
    dataset = MNISTTrainDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


def get_anomaly_dataloader(batch_size, shuffle=False, num_workers=4, pin_memory=True):
    dataset = MNISTAnomalyDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
