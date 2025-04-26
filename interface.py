# interface.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# replace MyCustomModel with the name of your model
from model import ConvAutoencoder as TheModel
# change my_descriptively_named_train_function to
# the function inside train.py that runs the training loop.
from train import train_model as the_trainer
# change cryptic_inf_f to the function inside predict.py that
# can be called to generate inference on a single image/batch.
from predict import classify_anomalies as the_predictor
# change UnicornImgDataset to your custom Dataset class.
from dataset import MNISTDataset as TheDataset
# change unicornLoader to your custom dataloader
from dataset import get_dataloaders as the_dataloader
# change batchsize, epochs to whatever your names are for these
# variables inside the config.py file
from config import config
# Using config object properties
the_batch_size = config.batch_size
total_epochs = config.num_epochs

def main():
    """Main function to run the entire pipeline."""
    # Set device
    device = config.device
    print(f"Using device: {device}")
    
    # Initialize model
    model = TheModel(latent_dim=config.latent_dim).to(device)
    print(f"Model initialized with latent dimension: {config.latent_dim}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Prepare data
    print("Preparing datasets...")
    train_csv = "data/mnist_train.csv"  # Adjust path as needed
    test_csv = "data/mnist_test.csv"    # Adjust path as needed
    
    # Check if files exist
    if not os.path.exists(train_csv):
        print(f"Warning: Training data file not found at {train_csv}")
        print("Using dummy data for demonstration...")
        # Create dummy data for demonstration if files don't exist
        # In a real scenario, you would need actual data
    
    try:
        dataloaders = the_dataloader(
            train_csv=train_csv,
            test_csv=test_csv,
            batch_size=the_batch_size
        )
        
        train_loader = dataloaders.get('train_loader')
        val_loader = dataloaders.get('val_loader')
        test_loader = dataloaders.get('test_loader')
        
        # Train model
        print(f"Starting model training for {total_epochs} epochs...")
        training_metrics = the_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=total_epochs,
            learning_rate=config.learning_rate,
            checkpoints_dir="checkpoints",
            device=device
        )
        
        # Save final model
        final_model_path = os.path.join("checkpoints", "final_weights.pth")
        model.save(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        # Plot training metrics
        plt.figure(figsize=(10, 5))
        plt.plot(training_metrics['train_loss'], label='Training Loss')
        if 'val_loss' in training_metrics and training_metrics['val_loss']:
            plt.plot(training_metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join("results", "training_loss.png"))
        
        # Test model if test data is available
        if test_loader:
            print("Testing model on test dataset...")
            # Define anomaly threshold (this could be determined more systematically)
            anomaly_threshold = 0.1
            
            true_labels, predicted_labels, recon_errors = the_predictor(
                model=model,
                data_loader=test_loader,
                threshold=anomaly_threshold,
                device=device
            )
            
            # Calculate accuracy
            accuracy = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p) / len(true_labels)
            print(f"Test accuracy: {accuracy:.4f}")
            
            # Save reconstruction errors
            plt.figure(figsize=(10, 5))
            plt.hist(recon_errors, bins=50)
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Count')
            plt.title('Distribution of Reconstruction Errors')
            plt.savefig(os.path.join("results", "recon_errors.png"))
            
            # Save results to text file
            with open(os.path.join("results", "test_results.txt"), 'w') as f:
                f.write(f"Test accuracy: {accuracy:.4f}\n")
                f.write(f"Mean reconstruction error: {np.mean(recon_errors):.4f}\n")
                f.write(f"Std of reconstruction error: {np.std(recon_errors):.4f}\n")
                f.write(f"Used anomaly threshold: {anomaly_threshold}\n")
        
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
