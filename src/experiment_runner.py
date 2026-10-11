import os
import sys
import time
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import DATASET_LOADERS, get_k_fold_indices
from src.mlp import MLPClassifier
from src.optimizers import GWO, CGWO, TCGWO

def run_single_fold(dataset_name: str, algo_name: str, run_id: int, fold_id: int, 
                    pop_size: int = 200, max_fes: int = 40000, seed: int = 42):
    """
    Executes a single fold of training and testing for a given algorithm and dataset.
    """
    # 1. Load dataset
    loader = DATASET_LOADERS[dataset_name]
    X, y = loader()
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    # 2. Get 5-fold split
    splits = get_k_fold_indices(y, k=5, seed=run_id * 100 + 42)
    train_idx, test_idx = splits[fold_id]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 3. Instantiate MLP
    mlp = MLPClassifier(n_features=n_features, n_classes=n_classes)
    
    # 4. Objective function (CEE on training set)
    def objective(vector: np.ndarray) -> float:
        return mlp.compute_cee(X_train, y_train, vector)
        
    # 5. Instantiate Optimizer
    opt_seed = (run_id + 1) * 1000 + fold_id * 37 + seed
    if algo_name == "GWO":
        opt = GWO(objective, dim=mlp.D, pop_size=pop_size, max_fes=max_fes, seed=opt_seed)
    elif algo_name == "CGWO":
        opt = CGWO(objective, dim=mlp.D, pop_size=pop_size, max_fes=max_fes, seed=opt_seed)
    elif algo_name == "TCGWO":
        opt = TCGWO(objective, dim=mlp.D, pop_size=pop_size, max_fes=max_fes, seed=opt_seed)
    else:
        raise ValueError(f"Unknown algorithm {algo_name}")
        
    # 6. Run optimization
    res = opt.run()
    best_weights = res["best_solution"]
    
    # 7. Evaluate on train and test sets
    train_cee = mlp.compute_cee(X_train, y_train, best_weights)
    test_cee = mlp.compute_cee(X_test, y_test, best_weights)
    train_acc = mlp.accuracy(X_train, y_train, best_weights)
    test_acc = mlp.accuracy(X_test, y_test, best_weights)
    
    return {
        "dataset": dataset_name,
        "algorithm": algo_name,
        "run_id": run_id,
        "fold_id": fold_id,
        "train_cee": train_cee,
        "test_cee": test_cee,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "runtime_s": res["time_seconds"]
    }

def main():
    datasets = ["Wine", "Seeds", "Breast cancer", "Zoo"]
    algorithms = ["GWO", "CGWO", "TCGWO"]
    n_runs = 3 # 3 independent runs of 5-fold CV (15 fold evaluations per dataset-algorithm pair)
    k_folds = 5
    pop_size = 200
    max_fes = 40000
    
    tasks = []
    for d_name in datasets:
        for a_name in algorithms:
            for r in range(n_runs):
                for f in range(k_folds):
                    tasks.append((d_name, a_name, r, f, pop_size, max_fes, 42))
                    
    print(f"Total experiment tasks to execute: {len(tasks)}")
    print(f"Using ProcessPoolExecutor with 8 workers...")
    
    results = []
    t_start = time.time()
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_single_fold, *task): task for task in tasks}
        completed_count = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(tasks):
                print(f"Completed {completed_count}/{len(tasks)} tasks ({round(time.time() - t_start, 1)}s elapsed)")
                
    total_time = time.time() - t_start
    print(f"All experiments finished in {round(total_time, 2)} seconds!")
    
    # Save raw results
    os.makedirs("results", exist_ok=True)
    with open("results/raw_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Raw results saved to results/raw_results.json")

if __name__ == "__main__":
    main()
