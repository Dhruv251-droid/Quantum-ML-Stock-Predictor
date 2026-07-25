import argparse, os, numpy as np, matplotlib.pyplot as plt
from train import load_data
from utils.metrics import print_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["classical", "hybrid", "fqgan", "all"], default="all")
    parser.add_argument("--past", type=int, default=3); parser.add_argument("--future", type=int, default=1)
    args = parser.parse_args(); data = load_data(args.past, args.future)
    results = {}; os.makedirs("results", exist_ok=True)

    models = {"classical": "models.classical_gan.ClassicalGAN", "hybrid": "models.hybrid_qgan.HybridQGAN", "fqgan": "models.fqgan.FQGANTrainer"}
    to_eval = [args.model] if args.model != "all" else list(models.keys())

    for name in to_eval:
        path = os.path.join("weights", f"{name}.pt")
        if not os.path.exists(path): continue
        parts = models[name].split("."); mod = __import__(".".join(parts[:-1]), fromlist=[parts[-1]]); cls = getattr(mod, parts[-1])
        m = cls(past=args.past, future=args.future); m.load(path)
        preds = m.predict(data["X_test"])
        y_true = data["scaler"].inverse_transform(data["y_test"] * data["test_norms"][:, None])
        y_pred = data["scaler"].inverse_transform(preds * data["test_norms"][:, None])
        results[name] = print_metrics(name, y_true, y_pred)
        
        plt.figure(figsize=(10, 5)); plt.plot(y_true.flatten()[:100], label="Actual"); plt.plot(y_pred.flatten()[:100], label="Predicted")
        plt.title(f"{name} Predictions"); plt.legend(); plt.savefig(f"results/{name}.png"); plt.close()

if __name__ == "__main__": main()
