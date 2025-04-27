# project_Mane_Ameya_Ashok
Roll No.20211017 
# Anomaly Detection Using Autoencoders on MNIST

## Project Overview
This project implements anomaly detection using convolutional autoencoders and variational autoencoders (VAEs) on the MNIST dataset. The system is designed to detect anomalies by learning the distribution of normal data and identifying samples that deviate significantly from this learned distribution.

## Model Architecture
The project implements two deep learning models:
- **Convolutional Autoencoder**: A traditional autoencoder that encodes input images to a lower-dimensional latent space and then reconstructs them.
- **Variational Autoencoder (VAE)**: A probabilistic autoencoder that encodes inputs to a distribution in latent space, providing better generalization and representation learning.

Both models use the reconstruction error as the anomaly score, where higher reconstruction error indicates a higher likelihood of anomaly.

## Dataset
The project uses the MNIST dataset in CSV format:
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv)
- Extract the downloaded files to the project directory
- The main training file (`mnist_train.csv`) needs to be unzipped before use

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Project Structure
- `config.py`: Configuration settings for the model and training
- `dataset.py`: Dataset classes and data loading utilities
- `model.py`: Implementations of autoencoder and VAE models
- `train.py`: Training loop and procedures
- `predict.py`: Inference and anomaly classification
- `logger.py`: Logging utilities
- `interface.py`: User interface for running the complete pipeline

## Usage

### Training
```bash
python interface.py --input path/to/mnist_train.csv --output path/to/save/results
```

### Inference Only
```bash
python interface.py --input path/to/test_data.csv --only_inference --model path/to/saved/model
```

### Additional Options
- `--skip_training`: Skip the training phase
- `--device`: Specify computing device (cuda/cpu)
- `--batch_size`: Set batch size for training and inference
- `--debug`: Enable debug mode for verbose output

## Anomaly Detection Process
1. The autoencoder learns to reconstruct normal (non-anomalous) samples during training
2. During inference, samples with high reconstruction error are classified as anomalies
3. A threshold parameter determines the cutoff for anomaly classification

## Notes
- `anom.csv` is an example dataset created by adding noise to the MNIST test data
- The code includes utilities for adding controlled noise to images to simulate anomalies
- All hyperparameters are configurable through the `config.py` file

## Citation
```
The MNIST dataset used in this project was obtained from:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv
```
