from model import AE as TheModel
from train import train_model as the_trainer
from predict import detect_anomalies as the_predictor
from dataset import MNISTTrainDataset as TheDataset, get_train_dataloader as the_dataloader, get_anomaly_dataloader as the_anomaly_dataloader
from config import batch_size as the_batch_size, epochs as total_epochs, lr, momentum, weight_decay, input_dim, latent_dim, threshold, torch_num_workers

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set pin_memory based on CUDA availability to avoid warnings
    pin_memory = torch.cuda.is_available()

    # Data loaders
    try:
        train_loader = the_dataloader(the_batch_size, num_workers=torch_num_workers, pin_memory=pin_memory)
        anomaly_loader = the_anomaly_dataloader(the_batch_size, num_workers=torch_num_workers, pin_memory=pin_memory)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Model, loss, optimizer
    model = TheModel(input_dim=input_dim, latent_dim=latent_dim)
    loss_fn = nn.MSELoss()
    
    # Using Adam optimizer instead of SGD for better convergence
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Update this line in interface.py
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Train
    print("Starting training...")
    try:
        train_metrics = the_trainer(model, total_epochs, train_loader, loss_fn, optimizer, device, scheduler)
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_metrics['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/training_loss.png')
        print("Training loss plot saved to plots/training_loss.png")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Evaluate anomalies
    print("\nEvaluating anomalies...")
    try:
        scores, true_labels, preds = the_predictor(model, anomaly_loader, nn.MSELoss(reduction='none'), device, threshold)
    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        return

    # Compute metrics
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        acc = accuracy_score(true_labels, preds)
        precision = precision_score(true_labels, preds, zero_division=0)
        recall = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)
        auc = roc_auc_score(true_labels, scores)
        cm = confusion_matrix(true_labels, preds)
        
        print("\nAnomaly Detection Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
        
    except ImportError:
        print("scikit-learn not installed: install it to compute accuracy and AUC metrics.")
    except Exception as e:
        print(f"Error computing metrics: {e}")

if __name__ == "__main__":
    main()
