import torch
import numpy as np


def detect_anomalies(model, anomaly_loader, loss_fn, device, threshold):
    """Generates reconstruction loss scores and binary predictions for anomalies."""
    model.to(device)
    model.eval()
    scores, true_labels, preds = [], [], []

    with torch.no_grad():
        for features, label in anomaly_loader:
            features = features.to(device)
            recon = model(features)
            # MSE over features per sample
            loss = loss_fn(recon, features)
            per_sample = loss.mean(dim=1).cpu().numpy()
            scores.extend(per_sample.tolist())
            true_labels.extend(label.numpy().tolist())
            
            # For monitoring, print some reconstruction losses
            if len(scores) <= 10:  # Just for the first batch
                for i, (score, lbl) in enumerate(zip(per_sample.tolist(), label.numpy().tolist())):
                    print(f"Sample {i}: loss={score:.6f}, label={lbl}")

    preds = [1 if s >= threshold else 0 for s in scores]
    
    # Print histogram of scores
    anomaly_scores = [s for s, l in zip(scores, true_labels) if l == 1]
    normal_scores = [s for s, l in zip(scores, true_labels) if l == 0]
    
    print(f"\nAnomaly score statistics:")
    print(f"  Normal samples (n={len(normal_scores)}): min={min(normal_scores):.6f}, max={max(normal_scores):.6f}, avg={np.mean(normal_scores):.6f}")
    print(f"  Anomaly samples (n={len(anomaly_scores)}): min={min(anomaly_scores):.6f}, max={max(anomaly_scores):.6f}, avg={np.mean(anomaly_scores):.6f}")
    print(f"  Threshold: {threshold}")
    
    return scores, true_labels, preds
