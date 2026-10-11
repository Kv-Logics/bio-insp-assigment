"""
Automated Results Table Formatter exporting high-resolution PNG summary graphic
Saves: outputs/plots/comparison_table_summary.png
"""

import json
import os
import matplotlib.pyplot as plt

os.makedirs("outputs/plots", exist_ok=True)

def export_table_image():
    with open("results/summary.json", "r") as f:
        data = json.load(f)
        
    datasets = ["Wine", "Seeds", "Breast cancer", "Zoo"]
    
    headers = [
        "Dataset", 
        "Metric", 
        "GWO (Baseline)", 
        "CGWO (Paper)", 
        "TCGWO (Our Improved)"
    ]
    
    rows = []
    for d in datasets:
        # CEE
        rows.append([
            d, 
            "CEE median (IQR)",
            f"{data[d]['GWO']['cee_median']:.4f} ({data[d]['GWO']['cee_iqr']:.4f})",
            f"{data[d]['CGWO']['cee_median']:.4f} ({data[d]['CGWO']['cee_iqr']:.4f})",
            f"{data[d]['TCGWO']['cee_median']:.4f} ({data[d]['TCGWO']['cee_iqr']:.4f})"
        ])
        # Train Acc
        rows.append([
            "", 
            "Train Accuracy %",
            f"{data[d]['GWO']['train_acc_median']:.2f}%",
            f"{data[d]['CGWO']['train_acc_median']:.2f}%",
            f"{data[d]['TCGWO']['train_acc_median']:.2f}%"
        ])
        # Test Acc
        rows.append([
            "", 
            "Test Accuracy %",
            f"{data[d]['GWO']['test_acc_median']:.2f}%",
            f"{data[d]['CGWO']['test_acc_median']:.2f}%",
            f"{data[d]['TCGWO']['test_acc_median']:.2f}%"
        ])
        # Runtime
        rows.append([
            "", 
            "Runtime / fold",
            f"{data[d]['GWO']['runtime_median']:.2f} s",
            f"{data[d]['CGWO']['runtime_median']:.2f} s",
            f"{data[d]['TCGWO']['runtime_median']:.2f} s"
        ])

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center"
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.15, 1.45)
    
    # Styling
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#1e40af")
            cell.set_text_props(color="white", weight="bold")
        else:
            block = (row_idx - 1) // 4
            if block % 2 == 1:
                cell.set_facecolor("#f8fafc")
            else:
                cell.set_facecolor("#ffffff")
                
            # Highlight best TCGWO or CGWO results
            if col_idx == 4 and row_idx in [5, 6, 9, 10]:
                cell.set_text_props(weight="bold", color="#15803d")
            elif col_idx == 3 and row_idx in [14]:
                cell.set_text_props(weight="bold", color="#15803d")
                
    ax.set_title("Experimental Results Summary: CGWO vs GWO and Enhanced TCGWO\n(Stratified 5-Fold Cross-Validation, Pop=200, FEs=4.0×10⁴)", 
                 fontsize=12, fontweight="bold", pad=25)
                 
    out = "outputs/plots/comparison_table_summary.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved results table graphic to {out}")

if __name__ == "__main__":
    export_table_image()
