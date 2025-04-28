import torch
import time
import os
from datetime import timedelta


def train_model(model, num_epochs, train_loader, loss_fn, optimizer, device, scheduler=None):
    """Runs training loop and saves final_weights.pth in checkpoints/."""
    model.to(device)
    metrics = {'train_loss': []}
    start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        last_batch_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            # Forward pass
            recon = model(batch)
            loss = loss_fn(recon, batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Stats tracking
            running_loss += loss.item()
            last_batch_loss = loss.item()
            batch_count += 1
            
            # Print occasional batch-level loss for more detailed monitoring
            if batch_count % 50 == 0:
                print(f"  Batch {batch_count}: loss={last_batch_loss:.6f}")

        # Compute average epoch loss
        epoch_loss = running_loss / len(train_loader)
        metrics['train_loss'].append(epoch_loss)
        
        # Print epoch summary
        print(f"[EPOCH {epoch+1}/{num_epochs}] LOSS: {epoch_loss:.6f} LAST_BATCH: {last_batch_loss:.6f} TIME: {timedelta(seconds=time.time()-epoch_start)}")
        
        # Step the scheduler if provided
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"  Learning rate adjusted: {old_lr:.6f} → {new_lr:.6f}")

    print(f"Training complete in {timedelta(seconds=time.time()-start)}")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Save the model
    model_path = 'checkpoints/final_weights.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return metrics
