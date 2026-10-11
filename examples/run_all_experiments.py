"""
Parametric Multi-Model, Multi-Dataset Experiment Orchestrator
CLI Arguments:
  --models: Model list (e.g., GWO, CGWO, TCGWO)
  --datasets: Dataset list (e.g., Wine, Seeds, "Breast cancer", Zoo)
  --agents: Swarm population size N (default: 200)
  --fes: Maximum function evaluations budget (default: 40000)
  --folds: Number of cross-validation folds (default: 5)
  --runs: Number of independent runs (default: 3)
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mlp_cgwo.dataset import DATASET_LOADERS, get_k_fold_indices
from mlp_cgwo.model import MLPClassifier
from mlp_cgwo.optimizers import GWO, CGWO, TCGWO

def run_single_fold(dataset_name, algo_name, run_id, fold_id, pop_size, max_fes, seed):
    loader = DATASET_LOADERS[dataset_name]
    X, y = loader()
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    splits = get_k_fold_indices(y, k=5, seed=run_id * 100 + 42)
    train_idx, test_idx = splits[fold_id]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    mlp = MLPClassifier(n_features=n_features, n_classes=n_classes)
    def obj(v): return mlp.compute_cee(X_train, y_train, v)
    
    opt_seed = (run_id + 1) * 1000 + fold_id * 37 + seed
    if algo_name == "GWO":
        opt = GWO(obj, dim=mlp.D, pop_size=pop_size, max_fes=max_fes, seed=opt_seed)
    elif algo_name == "CGWO":
        opt = CGWO(obj, dim=mlp.D, pop_size=pop_size, max_fes=max_fes, seed=opt_seed)
    elif algo_name == "TCGWO":
        opt = TCGWO(obj, dim=mlp.D, pop_size=pop_size, max_fes=max_fes, seed=opt_seed)
    else:
        raise ValueError(f"Unknown optimizer: {algo_name}")
        
    res = opt.run()
    best_v = res["best_solution"]
    
    return {
        "dataset": dataset_name,
        "algorithm": algo_name,
        "run_id": run_id,
        "fold_id": fold_id,
        "train_cee": mlp.compute_cee(X_train, y_train, best_v),
        "test_cee": mlp.compute_cee(X_test, y_test, best_v),
        "train_acc": mlp.accuracy(X_train, y_train, best_v),
        "test_acc": mlp.accuracy(X_test, y_test, best_v),
        "runtime_s": res["time_seconds"]
    }

def main():
    parser = argparse.ArgumentParser(description="Run MLP Metaheuristic Experiments")
    parser.add_argument("--models", nargs="+", default=["CGWO", "GWO", "TCGWO"], help="Optimizers to evaluate")
    parser.add_argument("--datasets", nargs="+", default=["Wine", "Seeds", "Breast cancer", "Zoo"], help="UCI datasets")
    parser.add_argument("--agents", type=int, default=200, help="Swarm population size N")
    parser.add_argument("--fes", type=int, default=40000, help="Max function evaluations")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent runs")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    
    args = parser.parse_args()
    
    tasks = []
    for d in args.datasets:
        for m in args.models:
            for r in range(args.runs):
                for f in range(args.folds):
                    tasks.append((d, m, r, f, args.agents, args.fes, 42))
                    
    print(f"🚀 Launching {len(tasks)} benchmark evaluations across models: {args.models} on datasets: {args.datasets}...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_fold, *task): task for task in tasks}
        count = 0
        for future in as_completed(futures):
            results.append(future.result())
            count += 1
            if count % 10 == 0 or count == len(tasks):
                print(f"Progress: {count}/{len(tasks)} tasks completed ({round(time.time() - t0, 1)}s)")
                
    out_file = "results/raw_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Finished in {round(time.time() - t0, 2)}s! Saved to {out_file}")

if __name__ == "__main__":
    main()
