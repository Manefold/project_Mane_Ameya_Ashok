import torch
from torch.utils.data import DataLoader
from dataset import MNISTDataset
import config
from PIL import Image
import numpy as np

def classify_anomalies(model, data_loader, threshold, device):
    """
    Classifies images as anomalous or normal based on reconstruction error.

    Args:
        model (nn.Module): Trained autoencoder model.
        data_loader (DataLoader): DataLoader for the dataset to classify.
        threshold (float): Threshold for the reconstruction error to
                         distinguish anomalies.
        device (str): 'cuda' or 'cpu'.

    Returns:
        tuple: (true_labels, predicted_labels, reconstruction_errors)
               where:
                   true_labels:  List of ground truth labels (0 for normal, 1 for anomaly).
                   predicted_labels: List of predicted labels (0 or 1).
                   reconstruction_errors: List of reconstruction errors for each image.
    """
    model.eval()  # Set model to evaluation mode
    true_labels = []
    predicted_labels = []
    reconstruction_errors = []
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 2:
              images, labels = batch
            else:
              images = batch
              labels = [0]*len(batch) # Assign dummy labels if none provided (e.g., anomaly data)
            images = images.to(device)
            images = images.view(-1, 784) # Flatten
            reconstructed = model(images)
            # Calculate reconstruction error (MSE)
            error = torch.mean((reconstructed.view(-1, 784) - images) ** 2, dim=1)
            reconstruction_errors.extend(error.cpu().numpy())

            # Predict labels based on threshold
            predictions = (error > threshold).long().cpu().numpy()
            predicted_labels.extend(predictions)
            true_labels.extend(labels.cpu().numpy())

    return true_labels, predicted_labels, reconstruction_errors

def inferloader(image_paths, transform=None):
  images = []
  for path in image_paths:
    img = Image.open(path).convert('L')  # Open in grayscale
    img = img.resize((config.resize_x, config.resize_y))
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0) / 255.0
    images.append(img_tensor)
  return torch.stack(images)

def classify_frogs(model, list_of_frog_img_paths, device):
    """
    Dummy function to satisfy the interface.
    This needs to be replaced with your actual frog classification logic.
    """

    model.eval()
    frog_batch = inferloader(list_of_frog_img_paths, transform=None)
    frog_batch = frog_batch.to(device)
    with torch.no_grad():
        logits = model(frog_batch) # Assumes your model outputs logits
        # Replace this with your actual classification
        labels = ["kiss" if l[0] > 0.5 else "throw in pond" for l in logits]
    return labels
