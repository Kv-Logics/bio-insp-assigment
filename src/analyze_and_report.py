import json
import numpy as np
from scipy import stats

def compute_median_iqr(values):
    med = np.median(values)
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    return med, iqr

def analyze():
    with open("results/raw_results.json", "r") as f:
        data = json.load(f)
        
    datasets = ["Wine", "Seeds", "Breast cancer", "Zoo"]
    algorithms = ["GWO", "CGWO", "TCGWO"]
    
    summary = {}
    for d in datasets:
        summary[d] = {}
        for a in algorithms:
            # Filter entries
            entries = [x for x in data if x["dataset"] == d and x["algorithm"] == a]
            
            # Group by run to get run-average or evaluate across all folds
            # Paper: "calculated based on the average results of the 5-fold cross-validation across 10 independent runs"
            runs = {}
            for e in entries:
                r_id = e["run_id"]
                if r_id not in runs:
                    runs[r_id] = []
                runs[r_id].append(e)
                
            run_train_cees = [np.mean([x["train_cee"] for x in folds]) for folds in runs.values()]
            run_test_cees = [np.mean([x["test_cee"] for x in folds]) for folds in runs.values()]
            run_train_accs = [np.mean([x["train_acc"] for x in folds]) for folds in runs.values()]
            run_test_accs = [np.mean([x["test_acc"] for x in folds]) for folds in runs.values()]
            run_runtimes = [np.mean([x["runtime_s"] for x in folds]) for folds in runs.values()]
            
            med_cee, iqr_cee = compute_median_iqr(run_train_cees)
            med_train_acc, iqr_train_acc = compute_median_iqr(run_train_accs)
            med_test_acc, iqr_test_acc = compute_median_iqr(run_test_accs)
            med_time, _ = compute_median_iqr(run_runtimes)
            
            summary[d][a] = {
                "cee_median": med_cee,
                "cee_iqr": iqr_cee,
                "train_acc_median": med_train_acc,
                "train_acc_iqr": iqr_train_acc,
                "test_acc_median": med_test_acc,
                "test_acc_iqr": iqr_test_acc,
                "runtime_median": med_time,
                "raw_runs_cee": run_train_cees,
                "raw_runs_test_acc": run_test_accs
            }
            
    print("="*80)
    print("EXPERIMENTAL SUMMARY TABLE (Our Implementation)")
    print("="*80)
    print(f"{'Dataset':<15} | {'Metric':<14} | {'GWO (Baseline)':<18} | {'CGWO (Paper)':<18} | {'TCGWO (Our Imprv)':<18}")
    print("-" * 88)
    for d in datasets:
        c_gwo = f"{summary[d]['GWO']['cee_median']:.4f} ({summary[d]['GWO']['cee_iqr']:.4f})"
        c_cgwo = f"{summary[d]['CGWO']['cee_median']:.4f} ({summary[d]['CGWO']['cee_iqr']:.4f})"
        c_tcgwo = f"{summary[d]['TCGWO']['cee_median']:.4f} ({summary[d]['TCGWO']['cee_iqr']:.4f})"
        print(f"{d:<15} | CEE median(IQR)| {c_gwo:<18} | {c_cgwo:<18} | {c_tcgwo:<18}")
        
        tr_gwo = f"{summary[d]['GWO']['train_acc_median']:.2f}%"
        tr_cgwo = f"{summary[d]['CGWO']['train_acc_median']:.2f}%"
        tr_tcgwo = f"{summary[d]['TCGWO']['train_acc_median']:.2f}%"
        print(f"{'':<15} | Train Acc %    | {tr_gwo:<18} | {tr_cgwo:<18} | {tr_tcgwo:<18}")
        
        te_gwo = f"{summary[d]['GWO']['test_acc_median']:.2f}%"
        te_cgwo = f"{summary[d]['CGWO']['test_acc_median']:.2f}%"
        te_tcgwo = f"{summary[d]['TCGWO']['test_acc_median']:.2f}%"
        print(f"{'':<15} | Test Acc %     | {te_gwo:<18} | {te_cgwo:<18} | {te_tcgwo:<18}")
        
        t_gwo = f"{summary[d]['GWO']['runtime_median']:.2f} s"
        t_cgwo = f"{summary[d]['CGWO']['runtime_median']:.2f} s"
        t_tcgwo = f"{summary[d]['TCGWO']['runtime_median']:.2f} s"
        print(f"{'':<15} | Runtime / fold | {t_gwo:<18} | {t_cgwo:<18} | {t_tcgwo:<18}")
        print("-" * 88)
        
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    analyze()
