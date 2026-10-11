"""
Standalone Implementation of Tent-Cosine Competing Leaders Grey Wolf Optimizer (TCGWO)
Our Proposed Enhanced Variant (Item 11)
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mlp_cgwo.model import MLPClassifier
from mlp_cgwo.fitness import compute_cross_entropy_error
from mlp_cgwo.dataset import load_wine

def run_tcgwo(mlp, X, y, pop_size=200, max_fes=20000, lb=-10.0, ub=10.0, seed=42):
    rng = np.random.RandomState(seed)
    N, D = pop_size, mlp.D
    M_min = 3
    MX_u = (M_min + 1) * 10
    MX_l = M_min
    M_max = int(np.round((D / N) * ((MX_u - MX_l) / 4.0) + MX_l)) if D <= 4 * N else MX_u
    M_max = min(M_max, N)
    
    fes = 0
    def eval_fit(ind):
        nonlocal fes
        fes += 1
        return compute_cross_entropy_error(mlp, X, y, ind)

    # 1. Tent Chaotic Map IPDE
    IX1 = lb + rng.rand(N, D) * (ub - lb)
    f1 = np.array([eval_fit(IX1[i]) for i in range(N)])
    
    R2 = rng.rand(N, D) * 0.98 + 0.01
    for i in range(N):
        r = R2[i]
        for _ in range(10):
            r = np.where(r < 0.5, 2.0 * r, 2.0 * (1.0 - r))
            r = np.where(r == 0.0, 0.001, r)
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
        
        # 2. Smooth Cosine Leader Competition Schedule
        if t < 0.75 * T_max:
            fraction = t / (0.75 * T_max)
            cos_factor = 0.5 * (1.0 + np.cos(np.pi * fraction))
            M = int(np.round(M_min + (M_max - M_min) * cos_factor))
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
    print("Testing Standalone Tent-Cosine CGWO (TCGWO) on Wine dataset...")
    X, y = load_wine()
    mlp = MLPClassifier(n_features=X.shape[1], n_classes=len(np.unique(y)))
    best_v, best_f, elapsed = run_tcgwo(mlp, X, y, pop_size=100, max_fes=5000)
    preds = mlp.predict(X, best_v)
    acc = np.mean(preds == y) * 100.0
    print(f"✅ TCGWO Test Complete! CEE: {best_f:.5f} | Acc: {acc:.2f}% | Time: {elapsed:.2f}s")
