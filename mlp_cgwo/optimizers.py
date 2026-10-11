import numpy as np
import time
from typing import Callable, Tuple, Dict, Any

class BaseOptimizer:
    def __init__(self, obj_func: Callable[[np.ndarray], float], dim: int, 
                 pop_size: int = 200, max_fes: int = 40000, 
                 lb: float = -10.0, ub: float = 10.0, seed: int = None):
        self.obj_func = obj_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_fes = max_fes
        self.lb = lb
        self.ub = ub
        self.rng = np.random.RandomState(seed)
        self.fes = 0
        self.best_solution = None
        self.best_fitness = float('inf')
        self.history = []

    def evaluate(self, x: np.ndarray) -> float:
        fit = self.obj_func(x)
        self.fes += 1
        if fit < self.best_fitness:
            self.best_fitness = fit
            self.best_solution = x.copy()
        return fit


class GWO(BaseOptimizer):
    """
    Standard Grey Wolf Optimizer (Mirjalili et al., 2014; Mirjalili, 2015).
    """
    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        N = self.pop_size
        D = self.dim
        
        # 1. Initialize population uniformly randomly
        pop = self.lb + self.rng.rand(N, D) * (self.ub - self.lb)
        fitness = np.zeros(N)
        for i in range(N):
            fitness[i] = self.evaluate(pop[i])
            
        # Best 3 wolves
        sort_idx = np.argsort(fitness)
        alpha, alpha_fit = pop[sort_idx[0]].copy(), fitness[sort_idx[0]]
        beta, beta_fit = pop[sort_idx[1]].copy(), fitness[sort_idx[1]]
        delta, delta_fit = pop[sort_idx[2]].copy(), fitness[sort_idx[2]]
        
        self.history.append(alpha_fit)
        
        # Iteration budget
        T_max = max(1, (self.max_fes - N) // N)
        t = 0
        
        while self.fes < self.max_fes:
            t += 1
            a = 2.0 - 2.0 * min(1.0, t / T_max)
            
            # Vectorized position update for each wolf
            # Alpha direction
            r1_a = self.rng.rand(N, D)
            r2_a = self.rng.rand(N, D)
            C_a = 2.0 * r1_a
            A_a = 2.0 * a * r2_a - a
            D_a = np.abs(C_a * alpha - pop)
            X1 = alpha - A_a * D_a
            
            # Beta direction
            r1_b = self.rng.rand(N, D)
            r2_b = self.rng.rand(N, D)
            C_b = 2.0 * r1_b
            A_b = 2.0 * a * r2_b - a
            D_b = np.abs(C_b * beta - pop)
            X2 = beta - A_b * D_b
            
            # Delta direction
            r1_d = self.rng.rand(N, D)
            r2_d = self.rng.rand(N, D)
            C_d = 2.0 * r1_d
            A_d = 2.0 * a * r2_d - a
            D_d = np.abs(C_d * delta - pop)
            X3 = delta - A_d * D_d
            
            # New positions
            pop_new = (X1 + X2 + X3) / 3.0
            pop = np.clip(pop_new, self.lb, self.ub)
            
            # Evaluate all wolves
            for i in range(N):
                if self.fes >= self.max_fes:
                    break
                fitness[i] = self.evaluate(pop[i])
                
            # Update leaders at end of iteration
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
                    
            self.history.append(alpha_fit)
            
        elapsed = time.time() - start_time
        return {
            "algorithm": "GWO",
            "best_solution": alpha,
            "best_fitness": alpha_fit,
            "time_seconds": elapsed,
            "fes": self.fes,
            "history": self.history
        }


class CGWO(BaseOptimizer):
    """
    Competing Leaders Grey Wolf Optimizer (Yang, 2024).
    Features:
    - IPDE (Initiation with Population Diversity Enhancement) using sinusoidal chaotic map
    - Competing leaders shrinking linearly from M_max to M_min
    - CWLU (Compact Wolf Leaders Update)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.M_min = 3
        self.MX_u = (self.M_min + 1) * 10 # 40
        self.MX_l = self.M_min # 3
        # Calculate M_max from Eq. (13)
        if self.dim <= 4 * self.pop_size:
            self.M_max = int(np.round((self.dim / self.pop_size) * ((self.MX_u - self.MX_l) / 4.0) + self.MX_l))
        else:
            self.M_max = self.MX_u
        self.M_max = min(self.M_max, self.pop_size)

    def init_ipde(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Algorithm 3: IPDE initialization
        IX1: N wolves sampled uniformly (or Gaussian bounded)
        IX2: N wolves sampled via Sinusoidal chaotic map
        Select best N from IX1 U IX2
        """
        N = self.pop_size
        D = self.dim
        
        # IX1: Uniform random population
        R1 = self.rng.rand(N, D)
        IX1 = self.lb + R1 * (self.ub - self.lb)
        fit1 = np.zeros(N)
        for i in range(N):
            fit1[i] = self.evaluate(IX1[i])
            
        # IX2: Sinusoidal chaotic map (Eq. 18)
        # r_k = sin(pi * r_{k-1})
        R2 = np.zeros((N, D))
        R_init = self.rng.rand(N, D) * 0.98 + 0.01 # keep strictly in (0, 1)
        for i in range(N):
            r = R_init[i]
            for _ in range(10): # K = 10 iterations
                r = np.sin(np.pi * r)
            R2[i] = r
        IX2 = self.lb + R2 * (self.ub - self.lb)
        fit2 = np.zeros(N)
        for i in range(N):
            fit2[i] = self.evaluate(IX2[i])
            
        # Union and select top N
        all_pop = np.vstack([IX1, IX2])
        all_fit = np.concatenate([fit1, fit2])
        
        best_indices = np.argsort(all_fit)[:N]
        pop = all_pop[best_indices].copy()
        fitness = all_fit[best_indices].copy()
        return pop, fitness

    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        N = self.pop_size
        D = self.dim
        
        # Step 1: IPDE Initialization
        pop, fitness = self.init_ipde()
        
        # Step 2 & 3: Set M = M_max and pick top M leaders
        M = self.M_max
        leader_indices = np.argsort(fitness)[:M]
        leaders = [pop[idx].copy() for idx in leader_indices]
        leader_fits = [fitness[idx] for idx in leader_indices]
        
        self.history.append(leader_fits[0])
        
        # Estimate maximum iterations
        T_max = max(1, (self.max_fes - 2 * N) // N)
        t = 0
        
        while self.fes < self.max_fes:
            t += 1
            # Eq. (10): a linearly decreases from 2 to 0
            a = 2.0 - 2.0 * min(1.0, t / T_max)
            
            # Eq. (12): update M
            if t < 0.75 * T_max:
                M = int(np.round(self.M_max - t * (4.0 * (self.M_max - self.M_min)) / (3.0 * T_max)))
            else:
                M = self.M_min
            M = max(self.M_min, min(self.M_max, M))
            
            # Shrink leader list if M decreased
            if len(leaders) > M:
                leaders = leaders[:M]
                leader_fits = leader_fits[:M]
                
            lead_arr = np.array(leaders) # shape: (M, D)
            
            # Update each wolf sequentially with CWLU
            for i in range(N):
                if self.fes >= self.max_fes:
                    break
                    
                # Eq. (6), (7), (8), (9), (16)
                # Compute U_{i, j} for all M leaders
                r1 = self.rng.rand(M, D)
                r2 = self.rng.rand(M, D)
                C = 2.0 * r1
                A = 2.0 * a * r2 - a
                # Distance from current wolf to each leader
                D_mat = np.abs(C * lead_arr - pop[i])
                U = lead_arr - A * D_mat # (M, D)
                
                # New position by averaging over M leaders (Eq. 16)
                x_new = np.mean(U, axis=0)
                x_new = np.clip(x_new, self.lb, self.ub)
                pop[i] = x_new
                
                # Evaluate new position
                f_new = self.evaluate(x_new)
                fitness[i] = f_new
                
                # CWLU: Compact wolf leaders update (lines 10-12 of Algorithm 2)
                # If better than the worst current leader, replace and re-sort
                if f_new < leader_fits[-1]:
                    leader_fits[-1] = f_new
                    leaders[-1] = x_new.copy()
                    
                    # Re-sort leaders
                    sorted_order = np.argsort(leader_fits)
                    leaders = [leaders[idx] for idx in sorted_order]
                    leader_fits = [leader_fits[idx] for idx in sorted_order]
                    lead_arr = np.array(leaders)
                    
            self.history.append(leader_fits[0])
            
        elapsed = time.time() - start_time
        return {
            "algorithm": "CGWO",
            "best_solution": leaders[0],
            "best_fitness": leader_fits[0],
            "time_seconds": elapsed,
            "fes": self.fes,
            "history": self.history
        }


class TCGWO(CGWO):
    """
    Enhanced Variant (Proposed Improvement for Item 11):
    Tent-Cosine Competing Leaders Grey Wolf Optimizer (TCGWO).
    Modifications over CGWO:
    1. Tent Chaotic Map instead of Sinusoidal in IPDE:
       The Sinusoidal map has an arcsine invariant distribution clustering at edges.
       The Tent map achieves uniform ergodicity over (0, 1), guaranteeing better exploration.
    2. Cosine Nonlinear Schedule for M(t):
       Smooth cosine decrease retains larger leader diversity longer in early exploration,
       avoiding premature attrition of search centers.
    """
    def init_ipde(self) -> Tuple[np.ndarray, np.ndarray]:
        N = self.pop_size
        D = self.dim
        
        # IX1: Uniform random
        R1 = self.rng.rand(N, D)
        IX1 = self.lb + R1 * (self.ub - self.lb)
        fit1 = np.zeros(N)
        for i in range(N):
            fit1[i] = self.evaluate(IX1[i])
            
        # IX2: Tent Chaotic Map
        # r_{k} = 2*r_{k-1} if r < 0.5 else 2*(1 - r_{k-1})
        R2 = np.zeros((N, D))
        R_init = self.rng.rand(N, D) * 0.98 + 0.01
        for i in range(N):
            r = R_init[i]
            for _ in range(10):
                # Tent map with slight perturbation to avoid zero-cycling
                r = np.where(r < 0.5, 2.0 * r, 2.0 * (1.0 - r))
                r = np.where(r == 0.0, 0.001, r)
            R2[i] = r
        IX2 = self.lb + R2 * (self.ub - self.lb)
        fit2 = np.zeros(N)
        for i in range(N):
            fit2[i] = self.evaluate(IX2[i])
            
        # Union and select top N
        all_pop = np.vstack([IX1, IX2])
        all_fit = np.concatenate([fit1, fit2])
        best_indices = np.argsort(all_fit)[:N]
        pop = all_pop[best_indices].copy()
        fitness = all_fit[best_indices].copy()
        return pop, fitness

    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        N = self.pop_size
        D = self.dim
        
        pop, fitness = self.init_ipde()
        
        M = self.M_max
        leader_indices = np.argsort(fitness)[:M]
        leaders = [pop[idx].copy() for idx in leader_indices]
        leader_fits = [fitness[idx] for idx in leader_indices]
        
        self.history.append(leader_fits[0])
        
        T_max = max(1, (self.max_fes - 2 * N) // N)
        t = 0
        
        while self.fes < self.max_fes:
            t += 1
            a = 2.0 - 2.0 * min(1.0, t / T_max)
            
            # Non-linear Cosine schedule for M(t)
            if t < 0.75 * T_max:
                fraction = t / (0.75 * T_max)
                cos_factor = 0.5 * (1.0 + np.cos(np.pi * fraction))
                M = int(np.round(self.M_min + (self.M_max - self.M_min) * cos_factor))
            else:
                M = self.M_min
            M = max(self.M_min, min(self.M_max, M))
            
            if len(leaders) > M:
                leaders = leaders[:M]
                leader_fits = leader_fits[:M]
                
            lead_arr = np.array(leaders)
            
            for i in range(N):
                if self.fes >= self.max_fes:
                    break
                    
                r1 = self.rng.rand(M, D)
                r2 = self.rng.rand(M, D)
                C = 2.0 * r1
                A = 2.0 * a * r2 - a
                D_mat = np.abs(C * lead_arr - pop[i])
                U = lead_arr - A * D_mat
                
                x_new = np.mean(U, axis=0)
                x_new = np.clip(x_new, self.lb, self.ub)
                pop[i] = x_new
                
                f_new = self.evaluate(x_new)
                fitness[i] = f_new
                
                if f_new < leader_fits[-1]:
                    leader_fits[-1] = f_new
                    leaders[-1] = x_new.copy()
                    sorted_order = np.argsort(leader_fits)
                    leaders = [leaders[idx] for idx in sorted_order]
                    leader_fits = [leader_fits[idx] for idx in sorted_order]
                    lead_arr = np.array(leaders)
                    
            self.history.append(leader_fits[0])
            
        elapsed = time.time() - start_time
        return {
            "algorithm": "TCGWO",
            "best_solution": leaders[0],
            "best_fitness": leader_fits[0],
            "time_seconds": elapsed,
            "fes": self.fes,
            "history": self.history
        }
