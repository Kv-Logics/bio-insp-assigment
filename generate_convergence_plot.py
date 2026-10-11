import os
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import load_seeds, load_breast_cancer, get_k_fold_indices
from src.mlp import MLPClassifier
from src.optimizers import GWO, CGWO, TCGWO

os.makedirs("figures", exist_ok=True)

def generate_plot():
    datasets = [
        ("Seeds", load_seeds),
        ("Breast cancer", load_breast_cancer)
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (d_name, loader) in zip(axes, datasets):
        X, y = loader()
        splits = get_k_fold_indices(y, k=5, seed=42)
        train_idx, _ = splits[0]
        X_train, y_train = X[train_idx], y[train_idx]
        
        mlp = MLPClassifier(X.shape[1], len(np.unique(y)))
        
        def obj(w):
            return mlp.compute_cee(X_train, y_train, w)
            
        gwo = GWO(obj, mlp.D, pop_size=200, max_fes=20000, seed=42)
        res_gwo = gwo.run()
        
        cgwo = CGWO(obj, mlp.D, pop_size=200, max_fes=20000, seed=42)
        res_cgwo = cgwo.run()
        
        tcgwo = TCGWO(obj, mlp.D, pop_size=200, max_fes=20000, seed=42)
        res_tcgwo = tcgwo.run()
        
        # Plot curves
        iters = np.arange(len(res_gwo["history"]))
        ax.plot(iters, res_gwo["history"], label="GWO (Baseline)", color="#d9534f", linestyle="--", linewidth=1.8)
        ax.plot(np.arange(len(res_cgwo["history"])), res_cgwo["history"], label="CGWO (Paper)", color="#0275d8", linewidth=2.0)
        ax.plot(np.arange(len(res_tcgwo["history"])), res_tcgwo["history"], label="TCGWO (Our Improved)", color="#5cb85c", linewidth=2.2)
        
        ax.set_title(f"Convergence Comparison on {d_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration Number", fontsize=10)
        ax.set_ylabel("Cross-Entropy Error (CEE)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend(frameon=True, facecolor="#f8f9fa")
        
    plt.tight_layout()
    plot_path = "figures/convergence_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Convergence plot saved successfully to {plot_path}")

if __name__ == "__main__":
    generate_plot()
