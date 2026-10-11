import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid activation function."""
    x_clipped = np.clip(x, -25.0, 25.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax activation function for multi-class classification."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

class MLPClassifier:
    """
    Feed-Forward Multi-Layer Perceptron (MLP) Classifier:
    - Input nodes: I (number of attributes)
    - Hidden nodes: H = 2*I + 1 with Sigmoid activation (Eq. 1)
    - Output nodes: O (1 for binary with Sigmoid, Q for multi-class with Softmax)
    - Continuous search agent vector: [W1, W2, theta1, theta2] (Eq. 3)
      Dimension D = (I + O + 1)*H + O
    """
    def __init__(self, n_features: int, n_classes: int, lb: float = -10.0, ub: float = 10.0):
        self.I = n_features
        self.H = 2 * n_features + 1
        self.n_classes = n_classes
        self.is_binary = (n_classes == 2)
        self.O = 1 if self.is_binary else n_classes
        
        self.w1_size = self.I * self.H
        self.w2_size = self.H * self.O
        self.b1_size = self.H
        self.b2_size = self.O
        self.D = self.w1_size + self.w2_size + self.b1_size + self.b2_size
        
        self.lb = lb
        self.ub = ub

    def unpack_vector(self, vector: np.ndarray):
        """Unpack 1D continuous search agent vector into weight matrices and bias vectors."""
        idx = 0
        w1 = vector[idx : idx + self.w1_size].reshape(self.I, self.H)
        idx += self.w1_size
        
        w2 = vector[idx : idx + self.w2_size].reshape(self.H, self.O)
        idx += self.w2_size
        
        b1 = vector[idx : idx + self.b1_size]
        idx += self.b1_size
        
        b2 = vector[idx : idx + self.b2_size]
        return w1, w2, b1, b2

    def forward(self, X: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Forward propagation through MLP:
        Hidden layer: s_j = F1( sum_i (w1_{i,j} * x_i) - theta1_j ) (Eq. 1)
        Output layer: y_j = F2( sum_i (w2_{i,j} * s_i) - theta2_j ) (Eq. 2)
        """
        w1, w2, b1, b2 = self.unpack_vector(vector)
        s = sigmoid(X @ w1 - b1)
        z2 = s @ w2 - b2
        if self.is_binary:
            y = sigmoid(z2)
        else:
            y = softmax(z2)
        return y

    def predict(self, X: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Predict discrete class labels."""
        y_prob = self.forward(X, vector)
        if self.is_binary:
            return (y_prob.flatten() >= 0.5).astype(int)
        else:
            return np.argmax(y_prob, axis=1)

    def compute_cee(self, X: np.ndarray, y_true: np.ndarray, vector: np.ndarray) -> float:
        """Compute Cross-Entropy Error (CEE)."""
        from .fitness import compute_cross_entropy_error
        return compute_cross_entropy_error(self, X, y_true, vector)

    def accuracy(self, X: np.ndarray, y_true: np.ndarray, vector: np.ndarray) -> float:
        """Compute classification accuracy %."""
        preds = self.predict(X, vector)
        return float(np.mean(preds == y_true.flatten()) * 100.0)
