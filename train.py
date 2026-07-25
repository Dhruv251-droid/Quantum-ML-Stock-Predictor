import argparse, os, sys, numpy as np, pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.preprocessing import MinMaxScaler
from utils.metrics import print_metrics

DATA_PATH = "FTSE_prices.csv"
WEIGHTS_DIR = "weights"

def load_data(past=3, future=1):
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).iloc[:, 0]
    _, trend = hpfilter(prices.values, lamb=1600)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(trend.reshape(-1, 1)).flatten().astype(np.float32)
    X, y = [], []
    for i in range(len(scaled) - past - future + 1):
        X.append(scaled[i: i + past]); y.append(scaled[i + past: i + past + future])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]
    # Per-sequence L2 normalisation
    def norm(Xs, ys):
        ns = np.ones(len(Xs), dtype=np.float32)
        for i, (xi, yi) in enumerate(zip(Xs, ys)):
            n = np.linalg.norm(np.concatenate([xi, yi]))
            if n > 1e-9:
                Xs[i], ys[i], ns[i] = xi / n, yi / n, n
            else:
                ns[i] = 1.0  # Already zero-ish
        return ns

    train_data = {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te, 
                  "train_norms": norm(X_tr, y_tr), "test_norms": norm(X_te, y_te), "scaler": scaler}
    
    # Final NaN check
    for k, v in train_data.items():
        if isinstance(v, np.ndarray) and np.isnan(v).any():
            print(f"[Warning] {k} contains NaNs. Filling with zeros.")
            np.nan_to_num(v, copy=False)
            
    return train_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["classical", "hybrid", "fqgan", "all"], default="classical")
    parser.add_argument("--past", type=int, default=3); parser.add_argument("--future", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None); parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args(); os.makedirs(WEIGHTS_DIR, exist_ok=True)
    data = load_data(args.past, args.future)
    
    if args.model in ["classical", "all"]:
        from models.classical_gan import ClassicalGAN
        m = ClassicalGAN(args.past, args.future, epochs=args.epochs or 50)
        X, y = (data["X_train"][:args.subset], data["y_train"][:args.subset]) if args.subset else (data["X_train"], data["y_train"])
        m.train(X, y); m.save(os.path.join(WEIGHTS_DIR, "classical.pt"))
    if args.model in ["hybrid", "all"]:
        from models.hybrid_qgan import HybridQGAN
        m = HybridQGAN(args.past, args.future, epochs=args.epochs or 1)
        X, y = (data["X_train"][:args.subset], data["y_train"][:args.subset]) if args.subset else (data["X_train"], data["y_train"])
        m.train(X, y); m.save(os.path.join(WEIGHTS_DIR, "hybrid.pt"))
    if args.model in ["fqgan", "all"]:
        from models.fqgan import FQGANTrainer
        m = FQGANTrainer(args.past, args.future, epochs=args.epochs or 1)
        X, y = (data["X_train"][:args.subset], data["y_train"][:args.subset]) if args.subset else (data["X_train"], data["y_train"])
        m.train(X, y); m.save(os.path.join(WEIGHTS_DIR, "fqgan.pt"))

if __name__ == "__main__": main()
