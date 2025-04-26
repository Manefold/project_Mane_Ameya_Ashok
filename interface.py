# interface.py
from model import get_model
from train import train_model
from predict import classify_anomalies, classify_frogs
from dataset import MNISTDataset, AnomalousDataset

# You may need to adjust these based on your specific needs
def load_dataset(csv_file, transform=None):
    return MNISTDataset(csv_file=csv_file, transform=transform)

def load_anomalous_dataset(csv_file, transform=None):
    return AnomalousDataset(csv_file=csv_file, transform=transform)

def create_model(latent_dim):
    return get_model(latent_dim)

def run_training(model, num_epochs, train_loader, loss_fn, optimizer, device):
    return train_model(model=model, num_epochs=num_epochs,
                       train_loader=train_loader, loss_fn=loss_fn,
                       optimizer=optimizer, device=device)

def run_anomaly_classification(model, data_loader, threshold, device):
    return classify_anomalies(model=model, data_loader=data_loader,
                             threshold=threshold, device=device)

def run_frog_classification(model, list_of_frog_img_paths, device):
    return classify_frogs(model, list_of_frog_img_paths, device)
