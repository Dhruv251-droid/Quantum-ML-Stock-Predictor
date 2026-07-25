import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true.flatten(), y_pred.flatten()))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true.flatten(), y_pred.flatten()))

def print_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    print(f"\n[{label}]")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  R²   : {metrics['r2']:.4f}")
    return metrics
