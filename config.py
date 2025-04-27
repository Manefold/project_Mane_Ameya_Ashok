# config.py
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class Config:
    # Data parameters
    resize_x: int = 28
    resize_y: int = 28
    input_channels: int = 1  # For MNIST (grayscale)
    
    # Model parameters
    latent_dim: int = 16
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 20
    weight_decay: float = 1e-5
    
    # Early stopping parameters
    patience: int = 5
    min_delta: float = 0.001
    
    # Noise analysis parameters
    noise_levels: list[float] = None
    
    # Anomaly detection parameters
    anomaly_threshold: Optional[float] = None
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Create default configuration
config = Config()
