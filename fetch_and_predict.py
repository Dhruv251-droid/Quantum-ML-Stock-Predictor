import yfinance as yf
import numpy as np
import pandas as pd
import os
import json
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.preprocessing import MinMaxScaler
import torch

def fetch_live_data(ticker="^FTSE", period="1y"):
    print(f"Fetching {ticker} data...")
    data = yf.download(ticker, period=period)
    if data.empty:
        raise ValueError("No data fetched from Yahoo Finance.")
    
    # yfinance sometimes returns MultiIndex columns. Let's handle it.
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'][ticker].dropna()
    else:
        prices = data['Close'].dropna()
        
    return prices

def process_data(prices, past=3, future=1):
    _, trend = hpfilter(prices.values, lamb=1600)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(trend.reshape(-1, 1)).flatten().astype(np.float32)
    
    X, y_actual = [], []
    for i in range(len(scaled) - past - future + 1):
        X.append(scaled[i: i + past])
        y_actual.append(scaled[i + past: i + past + future])
        
    X, y_actual = np.array(X), np.array(y_actual)
    
    # Calculate norms for inverse transform
    norms = np.ones(len(X), dtype=np.float32)
    for i, (xi, yi) in enumerate(zip(X, y_actual)):
        n = np.linalg.norm(np.concatenate([xi, yi]))
        if n > 1e-9:
            X[i] = xi / n
            norms[i] = n
            
    # We'll evaluate on the most recent 100 days
    X_test = X[-100:]
    y_test_raw = prices.values[-100:] # The actual raw prices
    test_norms = norms[-100:]
    
    return X_test, y_test_raw, test_norms, scaler

def get_predictions(model_name, X_test, test_norms, scaler, past=3, future=1):
    path = os.path.join("weights", f"{model_name}.pt")
    if not os.path.exists(path):
        print(f"Weights not found: {path}")
        return np.zeros(100).tolist()
        
    # Load model class
    models = {
        "classical": "models.classical_gan.ClassicalGAN",
        "hybrid": "models.hybrid_qgan.HybridQGAN", 
        "fqgan": "models.fqgan.FQGANTrainer"
    }
    
    parts = models[model_name].split(".")
    mod = __import__(".".join(parts[:-1]), fromlist=[parts[-1]])
    cls = getattr(mod, parts[-1])
    
    # Initialize and load weights
    m = cls(past=past, future=future)
    m.load(path)
    
    # Predict
    preds = m.predict(X_test)
    
    # Inverse transform
    y_pred = scaler.inverse_transform(preds * test_norms[:, None])
    return [int(v) for v in y_pred.flatten().tolist()]

def main():
    try:
        # Fetch data
        prices = fetch_live_data(ticker="^FTSE", period="2y")
        
        # Process data
        X_test, y_test_raw, test_norms, scaler = process_data(prices, past=3, future=1)
        
        # Generate predictions
        actual_prices = [int(v) for v in y_test_raw.tolist()]
        classical_preds = get_predictions("classical", X_test, test_norms, scaler)
        hybrid_preds = get_predictions("hybrid", X_test, test_norms, scaler)
        fqgan_preds = get_predictions("fqgan", X_test, test_norms, scaler)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        def calc_metrics(y_true, y_pred):
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            return {"rmse": rmse, "mae": mae, "r2": r2}

        # Prepare JSON structure
        data = {
            "labels": list(range(1, 101)),
            "actual": actual_prices,
            "classical": classical_preds,
            "hybrid": hybrid_preds,
            "fqgan": fqgan_preds,
            "metrics": {
                "classical": calc_metrics(actual_prices, classical_preds),
                "hybrid": calc_metrics(actual_prices, hybrid_preds),
                "fqgan": calc_metrics(actual_prices, fqgan_preds)
            }
        }
        
        # Write to JS file
        js_content = f"const PREDICTION_DATA = {json.dumps(data, indent=4)};\n"
        
        os.makedirs("web", exist_ok=True)
        with open("web/data.js", "w") as f:
            f.write(js_content)
            
        print("Successfully generated web/data.js with live predictions.")
        
    except Exception as e:
        print(f"Error fetching/predicting: {e}")

if __name__ == "__main__":
    main()
