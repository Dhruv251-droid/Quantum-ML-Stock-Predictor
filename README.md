# Quantum-ML-Stock-Predictor

This repository contains a stock price predictor utilizing Classical GAN, Hybrid Quantum GAN (Hybrid QGAN), and Fully Quantum GAN (FQGAN) architectures. The models are trained on historical stock data (e.g., FTSE prices) and evaluated on their prediction accuracy.

## Prerequisites

Make sure you have Python installed. Install the required dependencies using the `requirements.txt` file:

```bash 
pip install -r requirements.txt
```

The required libraries include:
- `torch`
- `pennylane`
- `numpy`
- `pandas`
- `scikit-learn`
- `statsmodels`
- `matplotlib`

## Dataset

The project expects a dataset in CSV format named `FTSE_prices.csv` in the root directory. This dataset should contain a column of stock prices. The scripts automatically process the data using `statsmodels` (hpfilter) for trend extraction and `MinMaxScaler` for normalization.

## Training the Models

You can train the models using the `train.py` script. The script allows you to train individual models or all of them consecutively. The trained weights will be saved in the `weights` directory.

**Basic Usage:**
```bash
python train.py
```
*(By default, this trains the classical model).*

**Train all models:**
```bash
python train.py --model all
```

**Train a specific model:**
```bash
python train.py --model hybrid  # options: classical, hybrid, fqgan
```

**Optional Arguments for `train.py`:**
- `--model`: Model to train (`classical`, `hybrid`, `fqgan`, `all`). Default: `classical`.
- `--past`: Number of past time steps to look at (lookback window). Default: `3`.
- `--future`: Number of future time steps to predict. Default: `1`.
- `--epochs`: Number of training epochs. Defaults vary per model.
- `--batch-size`: Batch size for training. Default: `128`.
- `--subset`: Number of samples to use for training (useful for quick testing).

## Evaluating and Testing the Models

Once the models are trained (and their weights are saved in the `weights` directory), you can evaluate them using the `evaluate.py` script. This script will compute performance metrics and generate prediction plots in the `results` directory.

**Evaluate all trained models:**
```bash
python evaluate.py --model all
```

**Evaluate a specific model:**
```bash
python evaluate.py --model hybrid
```

**Optional Arguments for `evaluate.py`:**
- `--model`: Model to evaluate (`classical`, `hybrid`, `fqgan`, `all`). Default: `all`.
- `--past`: Should match the `--past` value used during training. Default: `3`.
- `--future`: Should match the `--future` value used during training. Default: `1`.

### Outputs
The evaluation script compares actual vs. predicted values and outputs metrics to the console. Furthermore, it generates line plots of the predictions for the first 100 test samples and saves them as PNG images in the `results/` folder (e.g., `results/classical.png`, `results/hybrid.png`).
