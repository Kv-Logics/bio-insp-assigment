import numpy as np
from .model import MLPClassifier

def compute_cross_entropy_error(mlp: MLPClassifier, X: np.ndarray, y_true: np.ndarray, vector: np.ndarray) -> float:
    """
    Average Cross-Entropy Error (CEE) fitness function:
    - Binary Classification (Eq. 4):
      CEE = - (1 / Z) * sum [ y'_i * ln(y_i) + (1 - y'_i) * ln(1 - y_i) ]
    - Multi-Class Classification (Eq. 5):
      CEE = - (1 / (Z * Q)) * sum_i sum_j [ y'_{i,j} * ln(y_{i,j}) ]
    """
    y_pred = mlp.forward(X, vector)
    eps = 1e-15
    Z = X.shape[0]

    if mlp.is_binary:
        y_pred = np.clip(y_pred.flatten(), eps, 1.0 - eps)
        y_t = y_true.flatten()
        cee = - np.mean(y_t * np.log(y_pred) + (1.0 - y_t) * np.log(1.0 - y_pred))
        return float(cee)
    else:
        Q = mlp.n_classes
        y_pred = np.clip(y_pred, eps, 1.0)
        one_hot = np.zeros((Z, Q), dtype=np.float64)
        one_hot[np.arange(Z), y_true] = 1.0
        cee = - np.sum(one_hot * np.log(y_pred)) / (Z * Q)
        return float(cee)
