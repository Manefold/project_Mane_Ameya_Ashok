import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import time
from datetime import timedelta

def train_model(model, num_epochs, train_loader, loss_fn, optimizer, device):
    model.train()
    total_start = time.time()
    metrics = {'train_loss': []}

    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader): # Changed to get data only
            data = data.to(device)
            data = data.view(-1, 784)  # Flatten the image
            # forward
            output = model(data)
            loss = loss_fn(output, data.view(-1, 1, 28, 28))
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        metrics['train_loss'].append(epoch_loss)
        epoch_end = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
              f"Time: {timedelta(seconds=int(epoch_end - epoch_start))}")

    total_end = time.time()
    print(f"Training finished in: {timedelta(seconds=int(total_end - total_start))}")
    return metrics
