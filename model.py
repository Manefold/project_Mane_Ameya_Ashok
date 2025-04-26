# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from config import config
import os

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for image reconstruction and anomaly detection.
    
    Attributes:
        encoder: Encoder network that compresses input to latent representation
        decoder: Decoder network that reconstructs input from latent representation
    """
    def __init__(self, latent_dim: int = config.latent_dim):
        """
        Initialize the autoencoder with encoder and decoder networks.
        
        Args:
            latent_dim: Dimension of the latent space
        """
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(config.input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            # Second convolutional layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            
            # Third convolutional layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Flatten and reduce to latent dimension
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Expand from latent dimension
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            
            # Reshape to feature maps
            nn.Unflatten(1, (128, 7, 7)),
            
            # First transposed convolution
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Second transposed convolution
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 14x14 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Final convolution to generate output
            nn.Conv2d(32, config.input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output pixels in [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
              or [batch_size, height*width] which will be reshaped
              
        Returns:
            Reconstructed tensor of shape [batch_size, channels, height, width]
        """
        # Check if input is flattened and reshape if needed
        if x.dim() == 2:
            x = x.view(-1, config.input_channels, config.resize_y, config.resize_x)
            
        # Pass through encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the latent representation of input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation tensor
        """
        # Check if input is flattened and reshape if needed
        if x.dim() == 2:
            x = x.view(-1, config.input_channels, config.resize_y, config.resize_x)
            
        return self.encoder(x)
        
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.state_dict(),
            'latent_dim': config.latent_dim,
            'config': {
                'resize_x': config.resize_x,
                'resize_y': config.resize_y,
                'input_channels': config.input_channels
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = config.device) -> 'ConvAutoencoder':
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Create model instance
        model = cls(latent_dim=checkpoint.get('latent_dim', config.latent_dim))
        
        # Load state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        
        return model

class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for improved latent space representation.
    
    Attributes:
        encoder: Encoder network that maps input to mean and variance of latent space
        decoder: Decoder network that reconstructs input from latent sample
    """
    def __init__(self, latent_dim: int = config.latent_dim):
        """
        Initialize the VAE with encoder and decoder networks.
        
        Args:
            latent_dim: Dimension of the latent space
        """
        super(VariationalAutoencoder, self).__init__()
        
        # Shared encoder layers
        self.encoder_layers = nn.Sequential(
            nn.Conv2d(config.input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 64 * 7 * 7),
            nn.BatchNorm1d(64 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 7, 7)),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 7x7 -> 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 14x14 -> 28x28
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, config.input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of mean and log variance tensors
        """
        # Check if input is flattened and reshape if needed
        if x.dim() == 2:
            x = x.view(-1, config.input_channels, config.resize_y, config.resize_x)
            
        h = self.encoder_layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu: Mean tensor
            logvar: Log variance tensor
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed tensor
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, mean, log variance)
        """
        # Check if input is flattened and reshape if needed
        if x.dim() == 2:
            x = x.view(-1, config.input_channels, config.resize_y, config.resize_x)
            
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'config': {
                'resize_x': config.resize_x,
                'resize_y': config.resize_y,
                'input_channels': config.input_channels
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = config.device) -> 'VariationalAutoencoder':
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Create model instance
        model = cls(latent_dim=checkpoint.get('latent_dim', config.latent_dim))
        
        # Load state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        
        return model

def get_model(model_type: str = "autoencoder", latent_dim: int = config.latent_dim) -> nn.Module:
    """
    Factory function to get the appropriate model.
    
    Args:
        model_type: Type of model to return ('autoencoder' or 'vae')
        latent_dim: Dimension of the latent space
        
    Returns:
        Model instance
    """
    if model_type.lower() == "autoencoder":
        return ConvAutoencoder(latent_dim)
    elif model_type.lower() in ["vae", "variational"]:
        return VariationalAutoencoder(latent_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
