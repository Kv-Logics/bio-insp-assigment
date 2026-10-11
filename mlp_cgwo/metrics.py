import numpy as np

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy in percentage."""
    return float(np.mean(y_true.flatten() == y_pred.flatten()) * 100.0)

def compute_median_iqr(values):
    """Compute median and interquartile range (IQR)."""
    med = float(np.median(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    return med, iqr
