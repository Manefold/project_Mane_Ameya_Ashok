# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Tuple, List, Union
import os
from config import config
from logger import logger

class MNISTDataset(Dataset):
    """
    Dataset class for MNIST or similar data loaded from CSV.
    
    Attributes:
        data: Pandas DataFrame containing the dataset
        transform: PyTorch transforms to apply to images
        cache: Dictionary to cache processed images
        use_cache: Whether to use image caching
    """
    def __init__(self, 
                 csv_file: str, 
                 transform: Optional[transforms.Compose] = None,
                 use_cache: bool = True):
        """
        Initialize MNIST dataset from CSV file.
        
        Args:
            csv_file: Path to the CSV file containing the dataset
            transform: PyTorch transforms to apply to images
            use_cache: Whether to use image caching
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        logger.info(f"Loading dataset from {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
        logger.info(f"Dataset loaded with {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label at the given index.
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            Tuple of (image tensor, label)
        """
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        # Get image data and reshape
        img_data = self.data.iloc[idx, 1:].values.astype(np.uint8)
        img = img_data.reshape(config.resize_y, config.resize_x)
        img = Image.fromarray(img)
        
        # Transform image
        img_tensor = self.transform(img)
        
        # Get label
        label = int(self.data.iloc[idx, 0])
        
        result = (img_tensor, label)
        
        # Cache result if caching is enabled
        if self.use_cache:
            self.cache[idx] = result
            
        return result
    
    def add_noise(self, 
                 idx: int, 
                 noise_level: float = 0.1, 
                 noise_type: str = 'gaussian') -> torch.Tensor:
        """
        Add noise to an image at the specified index.
        
        Args:
            idx: Index of the image to modify
            noise_level: Amount of noise to add (0.0 to 1.0)
            noise_type: Type of noise ('gaussian' or 'salt_pepper')
            
        Returns:
            Noisy image tensor
        """
        img, label = self[idx]
        
        # Convert to numpy for easier manipulation
        img_np = img.clone().numpy()
        
        if noise_type == 'gaussian':
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, img_np.shape)
            img_np = img_np + noise
            img_np = np.clip(img_np, 0, 1)  # Clip values to valid range
        elif noise_type == 'salt_pepper':
            # Add salt and pepper noise
            mask = np.random.random(img_np.shape) < noise_level
            img_np[mask] = 1.0  # Salt
            
            mask = np.random.random(img_np.shape) < noise_level
            img_np[mask] = 0.0  # Pepper
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
        return torch.from_numpy(img_np).float()

class AnomalousDataset(Dataset):
    """
    Dataset class for anomalous data loaded from CSV.
    
    Attributes:
        data: Pandas DataFrame containing the dataset
        transform: PyTorch transforms to apply to images
        cache: Dictionary to cache processed images
        use_cache: Whether to use image caching
    """
    def __init__(self, 
                 csv_file: str, 
                 transform: Optional[transforms.Compose] = None,
                 use_cache: bool = True):
        """
        Initialize anomalous dataset from CSV file.
        
        Args:
            csv_file: Path to the CSV file containing the dataset
            transform: PyTorch transforms to apply to images
            use_cache: Whether to use image caching
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        logger.info(f"Loading anomalous dataset from {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        
        logger.info(f"Anomalous dataset loaded with {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get image at the given index.
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            Image tensor
        """
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        # Assuming no labels column in anomalous dataset
        img_data = self.data.iloc[idx, :].values.astype(np.uint8)
        img = img_data.reshape(config.resize_y, config.resize_x)
        img = Image.fromarray(img)
        
        # Transform image
        img_tensor = self.transform(img)
        
        # Cache result if caching is enabled
        if self.use_cache:
            self.cache[idx] = img_tensor
            
        return img_tensor

def get_dataloaders(
    train_csv: str,
    test_csv: Optional[str] = None,
    anomaly_csv: Optional[str] = None,
    batch_size: int = config.batch_size,
    val_split: float = 0.2,
    transform: Optional[transforms.Compose] = None
) -> dict:
    """
    Create DataLoaders for training, validation, testing, and anomaly detection.
    
    Args:
        train_csv: Path to training data CSV
        test_csv: Path to test data CSV (optional)
        anomaly_csv: Path to anomaly data CSV (optional)
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation
        transform: PyTorch transforms to apply
        
    Returns:
        Dictionary containing DataLoaders
    """
    result = {}
    
    # Create datasets
    train_dataset = MNISTDataset(csv_file=train_csv, transform=transform)
    
    # Split into train and validation
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        result['val_loader'] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
    
    # Create training loader
    result['train_loader'] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    # Create test loader if test CSV provided
    if test_csv:
        test_dataset = MNISTDataset(csv_file=test_csv, transform=transform)
        result['test_loader'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
    
    # Create anomaly loader if anomaly CSV provided
    if anomaly_csv:
        anomaly_dataset = AnomalousDataset(csv_file=anomaly_csv, transform=transform)
        result['anomaly_loader'] = DataLoader(
            anomaly_dataset, batch_size=batch_size, shuffle=False
        )
    
    return result
