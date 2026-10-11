"""
Standalone Implementation of Competing Leaders Grey Wolf Optimizer (CGWO) for MLP Training
Paper: Yang Z. (2024), Expert Systems with Applications, 239:122349.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- 1. MLP MODEL & CEE LOSS ---
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

class StandaloneMLP:
    def __init__(self, I, Q):
        self.I = I
        self.H = 2 * I + 1
        self.Q = Q
        self.is_binary = (Q == 2)
        self.O = 1 if self.is_binary else Q
        self.D = (self.I + self.O + 1) * self.H + self.O

    def forward(self, X, v):
        w1_sz = self.I * self.H
        w2_sz = self.H * self.O
        b1_sz = self.H
        
        w1 = v[:w1_sz].reshape(self.I, self.H)
        w2 = v[w1_sz : w1_sz + w2_sz].reshape(self.H, self.O)
        b1 = v[w1_sz + w2_sz : w1_sz + w2_sz + b1_sz]
        b2 = v[w1_sz + w2_sz + b1_sz :]
        
        s = sigmoid(X @ w1 - b1)
        z = s @ w2 - b2
        return sigmoid(z) if self.is_binary else softmax(z)

    def cee(self, X, y, v):
        y_hat = self.forward(X, v)
        eps = 1e-15
        Z = X.shape[0]
        if self.is_binary:
            y_hat = np.clip(y_hat.flatten(), eps, 1.0 - eps)
            y_t = y.flatten()
            return float(-np.mean(y_t * np.log(y_hat) + (1.0 - y_t) * np.log(1.0 - y_hat)))
        else:
            y_hat = np.clip(y_hat, eps, 1.0)
            oh = np.zeros((Z, self.Q))
            oh[np.arange(Z), y] = 1.0
            return float(-np.sum(oh * np.log(y_hat)) / (Z * self.Q))

    def accuracy(self, X, y, v):
        y_hat = self.forward(X, v)
        preds = (y_hat.flatten() >= 0.5).astype(int) if self.is_binary else np.argmax(y_hat, axis=1)
        return float(np.mean(preds == y.flatten()) * 100.0)

# --- 2. CGWO OPTIMIZER ---
def run_cgwo(mlp, X, y, pop_size=200, max_fes=20000, lb=-10.0, ub=10.0, seed=42):
    rng = np.random.RandomState(seed)
    N, D = pop_size, mlp.D
    M_min = 3
    MX_u = (M_min + 1) * 10 # 40
    MX_l = M_min # 3
    M_max = int(np.round((D / N) * ((MX_u - MX_l) / 4.0) + MX_l)) if D <= 4 * N else MX_u
    M_max = min(M_max, N)
    
    fes = 0
    def eval_fit(ind):
        nonlocal fes
        fes += 1
        return mlp.cee(X, y, ind)

    # IPDE Initialization
    IX1 = lb + rng.rand(N, D) * (ub - lb)
    f1 = np.array([eval_fit(IX1[i]) for i in range(N)])
    
    R2 = rng.rand(N, D) * 0.98 + 0.01
    for i in range(N):
        r = R2[i]
        for _ in range(10):
            r = np.sin(np.pi * r)
        R2[i] = r
    IX2 = lb + R2 * (ub - lb)
    f2 = np.array([eval_fit(IX2[i]) for i in range(N)])
    
    all_pop = np.vstack([IX1, IX2])
    all_f = np.concatenate([f1, f2])
    top_n = np.argsort(all_f)[:N]
    pop = all_pop[top_n].copy()
    fitness = all_f[top_n].copy()
    
    M = M_max
    lead_idx = np.argsort(fitness)[:M]
    leaders = [pop[idx].copy() for idx in lead_idx]
    leader_fits = [fitness[idx] for idx in lead_idx]
    
    T_max = max(1, (max_fes - 2 * N) // N)
    t = 0
    t0 = time.time()
    
    while fes < max_fes:
        t += 1
        a = 2.0 - 2.0 * min(1.0, t / T_max)
        if t < 0.75 * T_max:
            M = int(np.round(M_max - t * (4.0 * (M_max - M_min)) / (3.0 * T_max)))
        else:
            M = M_min
        M = max(M_min, min(M_max, M))
        
        if len(leaders) > M:
            leaders = leaders[:M]
            leader_fits = leader_fits[:M]
            
        lead_arr = np.array(leaders)
        
        for i in range(N):
            if fes >= max_fes: break
            r1 = rng.rand(M, D)
            r2 = rng.rand(M, D)
            C = 2.0 * r1
            A = 2.0 * a * r2 - a
            D_mat = np.abs(C * lead_arr - pop[i])
            U = lead_arr - A * D_mat
            
            x_new = np.clip(np.mean(U, axis=0), lb, ub)
            pop[i] = x_new
            f_new = eval_fit(x_new)
            fitness[i] = f_new
            
            # CWLU
            if f_new < leader_fits[-1]:
                leader_fits[-1] = f_new
                leaders[-1] = x_new.copy()
                order = np.argsort(leader_fits)
                leaders = [leaders[idx] for idx in order]
                leader_fits = [leader_fits[idx] for idx in order]
                lead_arr = np.array(leaders)
                
    elapsed = time.time() - t0
    return leaders[0], leader_fits[0], elapsed

if __name__ == "__main__":
    from mlp_cgwo.dataset import load_wine
    print("Testing Standalone CGWO on Wine dataset...")
    X, y = load_wine()
    mlp = StandaloneMLP(I=X.shape[1], Q=len(np.unique(y)))
    best_v, best_f, elapsed = run_cgwo(mlp, X, y, pop_size=100, max_fes=5000)
    acc = mlp.accuracy(X, y, best_v)
    print(f"✅ CGWO Test Complete! CEE: {best_f:.5f} | Acc: {acc:.2f}% | Time: {elapsed:.2f}s")
