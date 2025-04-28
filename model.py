import torch.nn as nn

class AE(nn.Module):
    """Simple fully-connected autoencoder compressing 784-dim input to 16-dim latent."""
    def __init__(self, input_dim=784, latent_dim=16):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)
        
    def encode(self, x):
        """Get the latent space representation."""
        return self.enc(x)
