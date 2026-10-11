import os
import urllib.request
import zipfile
import io
import numpy as np

def min_max_normalize(X: np.ndarray, target_min: float = 0.0, target_max: float = 1.0) -> np.ndarray:
    """
    Min-max normalization as defined in Eq. (19) of the paper:
    x' = ((x - min) / (max - min)) * (max' - min') + min'
    """
    X_norm = np.zeros_like(X, dtype=np.float64)
    for col in range(X.shape[1]):
        c_min = np.min(X[:, col])
        c_max = np.max(X[:, col])
        if c_max > c_min:
            X_norm[:, col] = ((X[:, col] - c_min) / (c_max - c_min)) * (target_max - target_min) + target_min
        else:
            X_norm[:, col] = target_min
    return X_norm

def get_k_fold_indices(y: np.ndarray, k: int = 5, seed: int = 42):
    """
    Stratified K-fold cross-validation split generator.
    Returns a list of k tuples: (train_indices, test_indices).
    """
    rng = np.random.RandomState(seed)
    unique_classes = np.unique(y)
    class_indices = {c: rng.permutation(np.where(y == c)[0]) for c in unique_classes}
    
    folds = [[] for _ in range(k)]
    for c, indices in class_indices.items():
        split = np.array_split(indices, k)
        for i in range(k):
            folds[i].extend(split[i])
            
    cv_splits = []
    for i in range(k):
        test_idx = np.array(folds[i], dtype=int)
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i]).astype(int)
        cv_splits.append((train_idx, test_idx))
    return cv_splits

def load_wine(data_dir: str = "data"):
    path = os.path.join(data_dir, "wine", "wine.data")
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            cls = int(parts[0]) - 1 # 0, 1, 2
            feats = [float(x) for x in parts[1:]]
            data.append((feats, cls))
    X = np.array([x[0] for x in data], dtype=np.float64)
    y = np.array([x[1] for x in data], dtype=int)
    X = min_max_normalize(X)
    return X, y

def load_seeds(data_dir: str = "data"):
    path = os.path.join(data_dir, "seeds", "seeds_dataset.txt")
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            feats = [float(x) for x in parts[:7]]
            cls = int(parts[7]) - 1 # 0, 1, 2
            data.append((feats, cls))
    X = np.array([x[0] for x in data], dtype=np.float64)
    y = np.array([x[1] for x in data], dtype=int)
    X = min_max_normalize(X)
    return X, y

def load_breast_cancer(data_dir: str = "data"):
    path = os.path.join(data_dir, "breast_cancer", "breast-cancer-wisconsin.data")
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            feats = []
            for x in parts[1:10]:
                feats.append(np.nan if x == "?" else float(x))
            cls = 1 if parts[10] == "4" else 0 # 2 -> 0 (benign), 4 -> 1 (malignant)
            data.append((feats, cls))
    X = np.array([x[0] for x in data], dtype=np.float64)
    y = np.array([x[1] for x in data], dtype=int)
    # Impute missing values with column median
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        if np.any(mask):
            col_med = np.nanmedian(X[:, col])
            X[mask, col] = col_med
    X = min_max_normalize(X)
    return X, y

def load_zoo(data_dir: str = "data"):
    path = os.path.join(data_dir, "zoo", "zoo.data")
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            feats = [float(x) for x in parts[1:17]]
            cls = int(parts[17]) - 1 # 0..6
            data.append((feats, cls))
    X = np.array([x[0] for x in data], dtype=np.float64)
    y = np.array([x[1] for x in data], dtype=int)
    X = min_max_normalize(X)
    return X, y

def load_scale(data_dir: str = "data"):
    path = os.path.join(data_dir, "scale", "balance-scale.data")
    cls_map = {'L': 0, 'B': 1, 'R': 2}
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            cls = cls_map[parts[0]]
            feats = [float(x) for x in parts[1:]]
            data.append((feats, cls))
    X = np.array([x[0] for x in data], dtype=np.float64)
    y = np.array([x[1] for x in data], dtype=int)
    X = min_max_normalize(X)
    return X, y

DATASET_LOADERS = {
    "Wine": load_wine,
    "Seeds": load_seeds,
    "Breast cancer": load_breast_cancer,
    "Zoo": load_zoo,
    "Scale": load_scale
}
