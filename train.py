# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import os
from config import config
from logger import logger
from model import ConvAutoencoder, VariationalAutoencoder

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Attributes:
        patience: Number of epochs with no improvement after which training stops
        min_delta: Minimum change in monitored value to qualify as improvement
        counter: Counter for epochs with no improvement
        best_score: Best validation score observed
        early_stop: Whether to stop training
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change in monitored value to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should be stopped, False otherwise
        """
        score = -val_loss  # Higher score is better
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop

def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                      logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    VAE loss function combining reconstruction and KL divergence.
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight of KL divergence term
        
    Returns:
        Total loss
    """
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = config.num_epochs,
    learning_rate: float = config.learning_rate,
    weight_decay: float = config.weight_decay,
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = config.patience,
    early_stopping_delta: float = config.min_delta,
    checkpoints_dir: str = "checkpoints",
    device: str = config.device
) -> Dict[str, List[float]]:
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        scheduler_patience: Patience for learning rate scheduler
        scheduler_factor: Factor for learning rate scheduler
        early_stopping_patience: Patience for early stopping
        early_stopping_delta: Minimum delta for early stopping
        checkpoints_dir: Directory to save checkpoints
        device: Device to train on
        
    Returns:
        Dictionary containing training metrics
    """
    logger.info(f"Training model on {device}")
    model = model.to(device)
    model.train()
    
    # Initialize metrics dictionary
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    # Create directory for checkpoints
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
    )
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, min_delta=early_stopping_delta
    )
    
    # Set up loss function based on model type
    is_vae = isinstance(model, VariationalAutoencoder)
    
    if not is_vae:
        loss_fn = nn.MSELoss(reduction='mean')
    
    # Start training timer
    total_start = time.time()
    
    # Train for specified number of epochs
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # Get data and move to device
            if len(batch) == 2:  # (data, label)
                data, _ = batch
            else:  # Just data
                data = batch
                
            data = data.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            if is_vae:
                recon, mu, logvar = model(data)
                loss = vae_loss_function(recon, data, mu, logvar)
