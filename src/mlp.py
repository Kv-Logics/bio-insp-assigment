import numpy as np

def sigmoid(x):
    """Numerically stable sigmoid activation function."""
    x_clipped = np.clip(x, -25.0, 25.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def softmax(x):
    """Numerically stable softmax activation function for multi-class outputs."""
    # x shape: (N_samples, Q)
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

class MLPClassifier:
    """
    Three-layer Multi-Layer Perceptron (MLP) Classifier as defined in the paper:
    - Input nodes: I (number of attributes)
    - Hidden nodes: H = 2*I + 1 with sigmoid activation (Eq. 1)
    - Output nodes: O (1 for binary with sigmoid, Q for multi-class with softmax)
    - Weight and bias vector representation (Eq. 3):
      [W1 (I*H), W2 (H*O), theta1 (H), theta2 (O)]
      Dimension D = I*H + H*O + H + O
    """
    def __init__(self, n_features: int, n_classes: int):
        self.I = n_features
        self.H = 2 * n_features + 1
        self.n_classes = n_classes
        self.is_binary = (n_classes == 2)
        self.O = 1 if self.is_binary else n_classes
        
        # Dimensions
        self.w1_size = self.I * self.H
        self.w2_size = self.H * self.O
        self.b1_size = self.H
        self.b2_size = self.O
        self.D = self.w1_size + self.w2_size + self.b1_size + self.b2_size
        
        self.lb = -10.0
        self.ub = 10.0

    def decode_vector(self, vector: np.ndarray):
        """Decode 1D search agent vector into weight matrices and bias vectors."""
        idx = 0
        w1 = vector[idx : idx + self.w1_size].reshape(self.I, self.H)
        idx += self.w1_size
        
        w2 = vector[idx : idx + self.w2_size].reshape(self.H, self.O)
        idx += self.w2_size
        
        b1 = vector[idx : idx + self.b1_size] # (H,)
        idx += self.b1_size
        
        b2 = vector[idx : idx + self.b2_size] # (O,)
        return w1, w2, b1, b2

    def forward(self, X: np.ndarray, vector: np.ndarray):
        """
        Forward propagation through MLP:
        s_j = F1( sum_i (w1_{i,j} * x_i) - theta1_j )   (Eq. 1)
        y_j = F2( sum_i (w2_{i,j} * s_i) - theta2_j )   (Eq. 2)
        """
        w1, w2, b1, b2 = self.decode_vector(vector)
        # Hidden layer
        s = sigmoid(X @ w1 - b1)
        # Output layer
        z2 = s @ w2 - b2
        if self.is_binary:
            y = sigmoid(z2) # (N_samples, 1)
        else:
            y = softmax(z2) # (N_samples, Q)
        return y

    def compute_cee(self, X: np.ndarray, y_true: np.ndarray, vector: np.ndarray) -> float:
        """
        Average Cross-Entropy Error (CEE):
        - Binary classification (Eq. 4):
          CEE = - (1 / Z) * sum [ y'_i * log(y_i) + (1 - y'_i) * log(1 - y_i) ]
        - Multi-class classification (Eq. 5):
          CEE = - (1 / (Z * Q)) * sum_i sum_j [ y'_{i,j} * log(y_{i,j}) ]
        """
        y_pred = self.forward(X, vector)
        eps = 1e-15 # numerical safeguard for log
        Z = X.shape[0]

        if self.is_binary:
            y_pred = np.clip(y_pred.flatten(), eps, 1.0 - eps)
            y_t = y_true.flatten()
            cee = - np.mean(y_t * np.log(y_pred) + (1.0 - y_t) * np.log(1.0 - y_pred))
            return float(cee)
        else:
            Q = self.n_classes
            y_pred = np.clip(y_pred, eps, 1.0)
            # y_true is 1D class indices (0..Q-1); convert to one-hot (Z, Q)
            one_hot = np.zeros((Z, Q), dtype=np.float64)
            one_hot[np.arange(Z), y_true] = 1.0
            cee = - np.sum(one_hot * np.log(y_pred)) / (Z * Q)
            return float(cee)

    def accuracy(self, X: np.ndarray, y_true: np.ndarray, vector: np.ndarray) -> float:
        """Calculate classification accuracy in percentage."""
        y_pred = self.forward(X, vector)
        if self.is_binary:
            preds = (y_pred.flatten() >= 0.5).astype(int)
            acc = np.mean(preds == y_true.flatten()) * 100.0
        else:
            preds = np.argmax(y_pred, axis=1)
            acc = np.mean(preds == y_true.flatten()) * 100.0
        return float(acc)
