# MNIST Anomaly Detection with PyTorch Autoencoder

This project implements an anomaly detection system using a PyTorch autoencoder trained on the MNIST dataset. It follows the structure outlined in the [Medium tutorial by Benjamin](https://benjoe.medium.com/anomaly-detection-using-pytorch-autoencoder-and-mnist-31c5c2186329).

## 📁 Project Structure

```
project_<student_full_name>/
├── checkpoints/
│   └── final_weights.pth
├── data/
│   ├── mnist_train.csv
│   └── anom.csv
├── dataset.py
├── model.py
├── train.py
├── predict.py
├── config.py
├── interface.py
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/project_<student_full_name>.git
   cd project_<student_full_name>
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## 🧠 Usage

To train the model and perform anomaly detection:

```bash
python interface.py
```

This will:
* Train the autoencoder on the MNIST training data.
* Save the trained model weights to `checkpoints/final_weights.pth`.
* Evaluate the model on the anomaly dataset.
* Output performance metrics such as Accuracy and AUC.

## 📊 Outputs

* **Model Weights**: Saved in `checkpoints/final_weights.pth`.
* **Performance Metrics**: Displayed in the console after evaluation.

## 📝 Notes

* Ensure that the `data/` directory contains the `mnist_train.csv` and `anom.csv` files.
* The `config.py` file contains configurable parameters such as learning rate, batch size, and number of epochs.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
