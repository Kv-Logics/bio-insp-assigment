"""
Publication-Quality Visualization Suite for MLP-CGWO Paper Reproduction
Generates:
1. visualization/mlp_real_weights.png: Neural architecture diagram with trained connection weights
2. visualization/weight_heatmaps.png: Input-to-Hidden and Hidden-to-Output weight heatmaps
3. visualization/flat_vector_mapping.png: 1D search agent parameter unpacking diagram
4. visualization/optimization_flowchart.png: Complete CGWO algorithmic pipeline flowchart
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mlp_cgwo.model import MLPClassifier
from mlp_cgwo.dataset import load_seeds
from mlp_cgwo.optimizers import CGWO

os.makedirs("visualization", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

def train_sample_model():
    """Train a quick CGWO model on Seeds to obtain actual optimized weights."""
    X, y = load_seeds()
    mlp = MLPClassifier(n_features=X.shape[1], n_classes=len(np.unique(y)))
    def obj(w): return mlp.compute_cee(X, y, w)
    opt = CGWO(obj, dim=mlp.D, pop_size=100, max_fes=8000, seed=42)
    res = opt.run()
    w1, w2, b1, b2 = mlp.unpack_vector(res["best_solution"])
    return mlp, w1, w2, b1, b2

def plot_weight_heatmaps(w1, w2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # W1 Heatmap
    im1 = ax1.imshow(w1, cmap="coolwarm", aspect="auto")
    ax1.set_title("Input-to-Hidden Weights ($W_1$)\nShape: 7 attributes $\\times$ 15 hidden nodes", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Hidden Neurons (j = 1..15)", fontsize=10)
    ax1.set_ylabel("Input Features (i = 1..7)", fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # W2 Heatmap
    im2 = ax2.imshow(w2, cmap="viridis", aspect="auto")
    ax2.set_title("Hidden-to-Output Weights ($W_2$)\nShape: 15 hidden nodes $\\times$ 3 output classes", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Output Neurons (k = 1..3)", fontsize=10)
    ax2.set_ylabel("Hidden Neurons (j = 1..15)", fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    out = "visualization/weight_heatmaps.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Saved weight heatmaps to {out}")

def plot_flat_vector_mapping(mlp):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    total_d = mlp.D
    w1_sz = mlp.w1_size
    w2_sz = mlp.w2_size
    b1_sz = mlp.b1_size
    b2_sz = mlp.b2_size
    
    # Title
    ax.text(0.5, 0.94, f"Continuous Search Agent 1D Parameter Vector Unpacking (Dimension D = {total_d})", 
            ha="center", va="center", fontsize=14, fontweight="bold", color="#111827")
    ax.text(0.5, 0.88, "Vector representation for 3-layer MLP classifier evolved via CGWO: [W₁ (I×H) | W₂ (H×O) | θ₁ (H) | θ₂ (O)]",
            ha="center", va="center", fontsize=10.5, color="#4b5563")

    # 1D Vector Bar in Middle
    bar_y = 0.68
    bar_h = 0.12
    x_start = 0.06
    bar_total_w = 0.88
    
    segments = [
        ("W₁", w1_sz, "#3b82f6", f"[0 .. {w1_sz-1}]"),
        ("W₂", w2_sz, "#10b981", f"[{w1_sz} .. {w1_sz+w2_sz-1}]"),
        ("θ₁", b1_sz, "#f59e0b", f"[{w1_sz+w2_sz} .. {w1_sz+w2_sz+b1_sz-1}]"),
        ("θ₂", b2_sz, "#ec4899", f"[{w1_sz+w2_sz+b1_sz} .. {total_d-1}]")
    ]
    
    seg_coords = []
    curr_x = x_start
    for name, sz, col, idx_str in segments:
        w = (sz / total_d) * bar_total_w
        rect = patches.FancyBboxPatch((curr_x, bar_y), w, bar_h, boxstyle="square,pad=0",
                                      linewidth=1.5, edgecolor="#1f2937", facecolor=col, alpha=0.9)
        ax.add_patch(rect)
        
        # Center symbol if width permits
        if w > 0.05:
            ax.text(curr_x + w/2, bar_y + bar_h/2, name, ha="center", va="center", 
                    fontsize=12, fontweight="bold", color="#ffffff")
        else:
            ax.text(curr_x + w/2, bar_y + bar_h/2, name, ha="center", va="center", 
                    fontsize=9, fontweight="bold", color="#ffffff")
                    
        seg_coords.append((curr_x, w, name, sz, col, idx_str))
        curr_x += w

    # Draw bottom details panel with 4 distinct cards
    card_y = 0.16
    card_h = 0.38
    card_gap = 0.02
    card_w = (bar_total_w - 3 * card_gap) / 4
    
    card_details = [
        {
            "title": "Input-to-Hidden Weights (W₁)",
            "shape": f"{mlp.I} inputs × {mlp.H} hidden",
            "params": f"{w1_sz} variables",
            "indices": f"Indices: [0 .. {w1_sz - 1}]",
            "color": "#3b82f6",
            "desc": "Synaptic connection matrix\nfrom input layer to hidden units"
        },
        {
            "title": "Hidden-to-Output Weights (W₂)",
            "shape": f"{mlp.H} hidden × {mlp.O} outputs",
            "params": f"{w2_sz} variables",
            "indices": f"Indices: [{w1_sz} .. {w1_sz + w2_sz - 1}]",
            "color": "#10b981",
            "desc": "Synaptic connection matrix\nfrom hidden layer to output units"
        },
        {
            "title": "Hidden Layer Biases (θ₁)",
            "shape": f"{mlp.H} bias terms",
            "params": f"{b1_sz} variables",
            "indices": f"Indices: [{w1_sz + w2_sz} .. {w1_sz + w2_sz + b1_sz - 1}]",
            "color": "#f59e0b",
            "desc": "Threshold bias values for\neach hidden sigmoid neuron"
        },
        {
            "title": "Output Layer Biases (θ₂)",
            "shape": f"{mlp.O} bias terms",
            "params": f"{b2_sz} variables",
            "indices": f"Indices: [{w1_sz + w2_sz + b1_sz} .. {total_d - 1}]",
            "color": "#ec4899",
            "desc": "Threshold bias values for\neach output classifier neuron"
        }
    ]
    
    for i, (info, (sx, sw, _, _, col, _)) in enumerate(zip(card_details, seg_coords)):
        cx = x_start + i * (card_w + card_gap)
        
        # Connect segment center to card top with dashed guide line
        ax.plot([sx + sw/2, cx + card_w/2], [bar_y, card_y + card_h], color="#9ca3af", 
                linestyle="--", linewidth=1.2, zorder=1)
                
        # Card Background
        card_rect = patches.FancyBboxPatch((cx, card_y), card_w, card_h, boxstyle="round,pad=0.015",
                                           linewidth=1.5, edgecolor=info["color"], facecolor="#f8fafc", zorder=2)
        ax.add_patch(card_rect)
        
        # Header banner inside card
        header_h = 0.08
        header_rect = patches.FancyBboxPatch((cx, card_y + card_h - header_h), card_w, header_h,
                                             boxstyle="round,pad=0.015", linewidth=0, facecolor=info["color"], zorder=3)
        ax.add_patch(header_rect)
        
        # Header text
        ax.text(cx + card_w/2, card_y + card_h - header_h/2, info["title"],
                ha="center", va="center", fontsize=8.5, fontweight="bold", color="#ffffff", zorder=4)
                
        # Card body texts
        text_y = card_y + card_h - header_h - 0.05
        ax.text(cx + card_w/2, text_y, f"Dimension: {info['shape']}",
                ha="center", va="center", fontsize=9, fontweight="bold", color="#1f2937", zorder=4)
                
        ax.text(cx + card_w/2, text_y - 0.06, info["params"],
                ha="center", va="center", fontsize=9.5, fontweight="bold", color=info["color"], zorder=4)
                
        ax.text(cx + card_w/2, text_y - 0.12, info["indices"],
                ha="center", va="center", fontsize=8, family="monospace", color="#374151", zorder=4)
                
        ax.text(cx + card_w/2, text_y - 0.19, info["desc"],
                ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#6b7280", zorder=4)

    # Footer
    ax.text(0.5, 0.04, "Search Bounds: [-10.0, +10.0]^D  |  Evaluated via Cross-Entropy Error (CEE)  |  Optimized via CGWO", 
            ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#6b7280")
            
    out = "visualization/flat_vector_mapping.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved vector mapping diagram to {out}")

def plot_network_diagram(w1, w2, b1, b2):
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.02, 0.98)
    
    I_nodes = min(w1.shape[0], 7)
    H_nodes = min(w1.shape[1], 10)
    O_nodes = w2.shape[1]
    
    x_in, x_hid, x_out = 0.16, 0.50, 0.84
    y_in = np.linspace(0.72, 0.14, I_nodes)
    y_hid = np.linspace(0.76, 0.10, H_nodes)
    y_out = np.linspace(0.60, 0.26, O_nodes)
    
    # Draw connections
    for i in range(I_nodes):
        for j in range(H_nodes):
            wt = w1[i, j]
            alpha = min(0.6, max(0.05, abs(wt)/10.0))
            col = "#ef4444" if wt < 0 else "#3b82f6"
            ax.plot([x_in, x_hid], [y_in[i], y_hid[j]], color=col, alpha=alpha, linewidth=0.8, zorder=1)
            
    for j in range(H_nodes):
        for k in range(O_nodes):
            wt = w2[j, k]
            alpha = min(0.6, max(0.05, abs(wt)/10.0))
            col = "#ef4444" if wt < 0 else "#10b981"
            ax.plot([x_hid, x_out], [y_hid[j], y_out[k]], color=col, alpha=alpha, linewidth=1.0, zorder=1)
            
    # Draw Nodes
    for i, y in enumerate(y_in):
        c = plt.Circle((x_in, y), 0.024, color="#3b82f6", ec="#1e293b", lw=1.5, zorder=5)
        ax.add_patch(c)
        ax.text(x_in - 0.038, y, f"$x_{{{i+1}}}$", ha="right", va="center", fontsize=9.5, fontweight="bold")
        
    for j, y in enumerate(y_hid):
        c = plt.Circle((x_hid, y), 0.022, color="#f59e0b", ec="#1e293b", lw=1.5, zorder=5)
        ax.add_patch(c)
        ax.text(x_hid, y, "$\\sigma$", ha="center", va="center", fontsize=8.5, color="#ffffff", fontweight="bold", zorder=6)
        
    for k, y in enumerate(y_out):
        c = plt.Circle((x_out, y), 0.026, color="#10b981", ec="#1e293b", lw=1.5, zorder=5)
        ax.add_patch(c)
        ax.text(x_out + 0.038, y, rf"$\hat{{y}}_{{{k+1}}}$", ha="left", va="center", fontsize=9.5, fontweight="bold")
        
    # Headers with clear separation
    ax.text(x_in, 0.84, f"Input Layer\n(I = {I_nodes})", ha="center", va="center", fontsize=11, fontweight="bold", color="#1e40af")
    ax.text(x_hid, 0.84, f"Hidden Layer (Sigmoid)\n(H = 2I+1 = {w1.shape[1]})", ha="center", va="center", fontsize=11, fontweight="bold", color="#b45309")
    ax.text(x_out, 0.84, f"Output Layer (Softmax)\n(O = {O_nodes} Classes)", ha="center", va="center", fontsize=11, fontweight="bold", color="#047857")
    
    ax.text(0.5, 0.94, "Publication Architecture: Three-Layer MLP with Actual CGWO-Optimized Weights", 
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111827")
            
    out = "visualization/mlp_real_weights.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved MLP architecture diagram to {out}")

def plot_optimization_flowchart():
    fig, ax = plt.subplots(figsize=(10, 9), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    stages = [
        ("Step 1: Dataset Min-Max Scaling (Eq. 19)\nX' in [0, 1] | Stratified 5-Fold Cross-Validation Partitioning", 0.89, "#dbeafe"),
        ("Step 2: IPDE Population Initialization (Algorithm 3)\nUnion of Uniform Random & Sinusoidal Chaotic (r_k = sin(π r_{k-1}))", 0.74, "#dcfce7"),
        ("Step 3: Competing Leaders Group Allocation\nM_max from Dimension D (Eq. 13) | Top M wolves initialized as Leaders", 0.59, "#fef3c7"),
        ("Step 4: Dynamic Shrinking Leader Competition (Eq. 12)\nM linearly shrinks from M_max -> 3 over first 75% of iterations", 0.44, "#fce7f3"),
        ("Step 5: Mean-Filtered Position Update (Eq. 16)\nX_i(t+1) = (1/M) * sum(U_{i,j}) averaged over all M competing leaders", 0.29, "#f3e8ff"),
        ("Step 6: Compact Wolf Leaders Update (CWLU) & CEE Evaluation\nInstant replacement of worst leader upon each wolf evaluation", 0.14, "#ccfbf1")
    ]
    
    box_w = 0.84
    box_h = 0.085
    box_x = 0.08
    
    for i, (text, y, col) in enumerate(stages):
        rect = patches.FancyBboxPatch((box_x, y - box_h/2), box_w, box_h, boxstyle="round,pad=0.015",
                                      fc=col, ec="#1e293b", lw=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(0.5, y, text, ha="center", va="center", fontsize=10, fontweight="bold", color="#0f172a", zorder=3)
        
        # Downward arrow to next box
        if i < len(stages) - 1:
            y_curr_bottom = y - box_h/2
            y_next_top = stages[i+1][1] + box_h/2
            ax.annotate("", xy=(0.5, y_next_top), xytext=(0.5, y_curr_bottom),
                        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#1e40af", mutation_scale=15), zorder=4)
                        
    ax.text(0.5, 0.96, "End-to-End Competing Leaders Grey Wolf Optimizer (CGWO) Architecture Flow", 
            ha="center", va="center", fontsize=13, fontweight="bold", color="#0f172a")
                 
    out = "visualization/optimization_flowchart.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved optimization flowchart to {out}")

if __name__ == "__main__":
    print("Generating Publication-Quality Visualizations...")
    mlp, w1, w2, b1, b2 = train_sample_model()
    plot_weight_heatmaps(w1, w2)
    plot_flat_vector_mapping(mlp)
    plot_network_diagram(w1, w2, b1, b2)
    plot_optimization_flowchart()
    print("All visualizations generated successfully in visualization/")
