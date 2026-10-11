"""
Standalone Implementation of Canonical Grey Wolf Optimizer (GWO) for MLP Training
Paper: Mirjalili et al. (2014) / Mirjalili (2015)
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mlp_cgwo.model import MLPClassifier
from mlp_cgwo.fitness import compute_cross_entropy_error
from mlp_cgwo.dataset import load_wine

def run_gwo(mlp, X, y, pop_size=200, max_fes=20000, lb=-10.0, ub=10.0, seed=42):
    rng = np.random.RandomState(seed)
    N, D = pop_size, mlp.D
    fes = 0
    
    def eval_fit(ind):
        nonlocal fes
        fes += 1
        return compute_cross_entropy_error(mlp, X, y, ind)

    pop = lb + rng.rand(N, D) * (ub - lb)
    fitness = np.array([eval_fit(pop[i]) for i in range(N)])
    
    sort_idx = np.argsort(fitness)
    alpha, alpha_fit = pop[sort_idx[0]].copy(), fitness[sort_idx[0]]
    beta, beta_fit = pop[sort_idx[1]].copy(), fitness[sort_idx[1]]
    delta, delta_fit = pop[sort_idx[2]].copy(), fitness[sort_idx[2]]
    
    T_max = max(1, (max_fes - N) // N)
    t = 0
    t0 = time.time()
    
    while fes < max_fes:
        t += 1
        a = 2.0 - 2.0 * min(1.0, t / T_max)
        
        r1_a, r2_a = rng.rand(N, D), rng.rand(N, D)
        X1 = alpha - (2.0 * a * r2_a - a) * np.abs(2.0 * r1_a * alpha - pop)
        
        r1_b, r2_b = rng.rand(N, D), rng.rand(N, D)
        X2 = beta - (2.0 * a * r2_b - a) * np.abs(2.0 * r1_b * beta - pop)
        
        r1_d, r2_d = rng.rand(N, D), rng.rand(N, D)
        X3 = delta - (2.0 * a * r2_d - a) * np.abs(2.0 * r1_d * delta - pop)
        
        pop = np.clip((X1 + X2 + X3) / 3.0, lb, ub)
        for i in range(N):
            if fes >= max_fes: break
            fitness[i] = eval_fit(pop[i])
            
        for i in range(N):
            f = fitness[i]
            if f < alpha_fit:
                delta, delta_fit = beta.copy(), beta_fit
                beta, beta_fit = alpha.copy(), alpha_fit
                alpha, alpha_fit = pop[i].copy(), f
            elif f < beta_fit:
                delta, delta_fit = beta.copy(), beta_fit
                beta, beta_fit = pop[i].copy(), f
            elif f < delta_fit:
                delta, delta_fit = pop[i].copy(), f
                
    elapsed = time.time() - t0
    return alpha, alpha_fit, elapsed

if __name__ == "__main__":
    print("Testing Standalone Canonical GWO on Wine dataset...")
    X, y = load_wine()
    mlp = MLPClassifier(n_features=X.shape[1], n_classes=len(np.unique(y)))
    best_v, best_f, elapsed = run_gwo(mlp, X, y, pop_size=100, max_fes=5000)
    preds = mlp.predict(X, best_v)
    acc = np.mean(preds == y) * 100.0
    print(f"✅ GWO Test Complete! CEE: {best_f:.5f} | Acc: {acc:.2f}% | Time: {elapsed:.2f}s")
