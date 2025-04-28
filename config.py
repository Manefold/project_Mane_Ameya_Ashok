# config.py
# File paths
train_csv     = 'data/mnist_train.csv'
test_csv      = 'data/anom.csv'

# Model parameters
input_dim     = 784
latent_dim    = 16

# Training hyperparameters
batch_size    = 32
lr            = 1e-3  # Changed from 1e-2 to 1e-3 for potentially better convergence
weight_decay  = 1e-5
momentum      = 0.9
epochs        = 15

# Threshold for anomaly decision
threshold     = 0.3

# DataLoader settings - These will be conditionally set in interface.py
torch_num_workers = 4
pin_memory         = True
