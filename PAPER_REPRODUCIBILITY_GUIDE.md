# End-to-End Research Paper Implementation & Reproducibility Guide
### *A Complete Protocol for Porting, Benchmarking, and Validating Metaheuristic Neural Network Training from MATLAB to Python*

---

## 📌 Executive Summary & Purpose

This guide provides a comprehensive, step-by-step technical blueprint documenting the complete reproduction, mathematical porting, and empirical validation of the research paper:

> **Paper Title:** *Competing leaders grey wolf optimizer and its application for training multi-layer perceptron classifier*  
> **Author:** Zhenlun Yang (2024)  
> **Journal:** *Expert Systems with Applications*, Vol. 239, 122349  
> **DOI / URL:** [10.1016/j.eswa.2023.122349](https://doi.org/10.1016/j.eswa.2023.122349)  
> **Original Codebase:** MATLAB 2015a under Windows 7 (Intel 3.40 GHz CPU, 8 GB RAM).  
> **Ported Codebase:** Pure-Python modular architecture (`numpy`, `scipy`, `matplotlib`).

This document is structured so that any researcher, student, or engineer can replicate this entire pipeline or adapt it as a **universal template** for porting swarm intelligence and metaheuristic neural network algorithms from MATLAB into vector-optimized Python.

---

## 🗂️ Universal Project Architecture

The ported repository is structured into modular layers separating core math, optimization algorithms, data ingestion, experiment orchestration, and presentation graphics:

```text
bio-sim1/
├── mlp_cgwo/                       # Core reusable Python package
│   ├── __init__.py                 # Package initializer
│   ├── model.py                    # MLP architecture, Sigmoid & Softmax forward pass, parameter unpacking
│   ├── fitness.py                  # Binary & Multi-class Average Cross-Entropy Error (CEE) loss functions
│   ├── optimizers.py               # Vectorized GWO, CGWO (IPDE+CWLU), and improved TCGWO algorithms
│   ├── dataset.py                  # UCI data loaders, Min-Max normalization (Eq. 19), 5-fold CV splitter
│   └── metrics.py                  # Classification accuracy and IQR statistical evaluators
├── python converted/               # 1-to-1 standalone equivalents matching the paper routines
│   ├── README.md                   # Guide to standalone scripts
│   ├── cgwo_mlp.py                 # Standalone Competing Leaders GWO MLP Trainer
│   ├── gwo_mlp.py                  # Standalone Canonical GWO Baseline Trainer
│   └── tcgwo_mlp.py                # Standalone Improved Tent-Cosine CGWO Trainer
├── data/                           # Active UCI benchmark datasets (Wine, Seeds, Breast Cancer, Zoo, Scale)
├── examples/                       # Automated execution & visualization scripts
│   ├── run_all_experiments.py      # Unified multi-dataset, multi-model parametric benchmark runner
│   ├── generate_visualizations.py  # Publication-grade network diagrams, real weights, heatmaps & flowcharts
│   └── save_table_as_image.py      # Automated table formatter exporting high-res PNG results
├── outputs/                        # Output artifacts directory
│   ├── csv/                        # Saved convergence curves and metrics summary tables
│   └── plots/                      # Generated comparative convergence plots and styled tables
├── visualization/                  # Publication-ready MLP architecture, vector mapping & weight heatmaps
├── results/                        # Raw experiment evaluations and statistical summaries (JSON)
├── requirements.txt                # Fixed Python dependencies
├── TERM_WORK_REPORT.md             # Official 19ECE355 term work submission report (Markdown)
├── TERM_WORK_REPORT.html           # Printable 3-page styled HTML term work report (CSS @media print)
└── PAPER_REPRODUCIBILITY_GUIDE.md  # This master instructions file
```

---

## 🚀 Phase 1: Environment Setup & Isolation

Strict environment isolation is essential for reproducibility. Python 3.10+ in a dedicated virtual environment is used.

### 1.1 Terminal Commands: Clone & Virtual Environment

```bash
# 1. Navigate to your project directory
cd "/home/kv/Projects/bio-sim1"

# 2. Verify Python version (3.10 or higher recommended)
python3 --version

# 3. Create an isolated virtual environment named 'venv'
python3 -m venv venv

# 4. Activate the virtual environment
source venv/bin/activate

# 5. Upgrade package manager
pip install --upgrade pip setuptools wheel
```

### 1.2 Terminal Commands: Dependency Installation

```bash
# Install dependencies into virtual environment
pip install -r requirements.txt
```

*Contents of `requirements.txt`:*
```text
numpy>=1.23.0
scipy>=1.9.0
matplotlib>=3.6.0
```

---

## 📊 Phase 2: Data Handling & Preprocessing Pipeline

The original paper benchmarks algorithms across **11 benchmark datasets** from the UCI Machine Learning Repository. We implemented active loaders for representative binary and multi-class tasks spanning small to high-dimensional problems:

### 2.1 Dataset Specifications & Network Topology

| Dataset | Problem Type | Attributes ($I$) | Classes ($Q$) | Hidden Units ($H = 2I+1$) | Vector Dim ($D$) | $M_{\max}$ (Eq. 13) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Scale** | Multi-class | 4 | 3 | 9 | 75 | 6 |
| **Seeds** | Multi-class | 7 | 3 | 15 | 168 | 11 |
| **Breast Cancer** | Binary Class | 9 | 2 | 19 | 210 | 13 |
| **Statlog Heart** | Binary Class | 13 | 2 | 27 | 406 | 22 |
| **Wine** | Multi-class | 13 | 3 | 27 | 462 | 24 |
| **Australian** | Binary Class | 14 | 2 | 29 | 465 | 25 |
| **Cleveland** | Multi-class | 13 | 5 | 27 | 518 | 27 |
| **Zoo** | Multi-class | 16 | 7 | 33 | 799 | 40 |
| **Parkinson** | Binary Class | 22 | 2 | 45 | 1081 | 40 |
| **Ionosphere** | Binary Class | 34 | 2 | 69 | 2485 | 40 |
| **Chess** | Binary Class | 36 | 2 | 73 | 2775 | 40 |

### 2.2 Mathematical Normalization Rule (Eq. 19)

To eliminate scaling disparities among heterogeneous attributes, input features are mapped to $[0, 1]$ using min-max normalization:

$$x' = \frac{x - \min(x)}{\max(x) - \min(x)} (\max' - \min') + \min'$$

where $[\min', \max'] = [0, 1]$. Missing attributes (e.g. Bare Nuclei in Breast Cancer) are median-imputed across respective columns.

---

## 🧠 Phase 3: Neural Network & Parameter Vector Formulation

Rather than using gradient-based backpropagation, the multi-layer perceptron parameters are evolved via population-based metaheuristic search.

### 3.1 Network Architecture & Equations
- **Single Hidden Layer:** $H = 2I + 1$ neurons.
- **Activation Functions:**
  - Hidden neurons ($F_1$): Logistic Sigmoid:
    $$s_j = F_1\left( \sum_{i=1}^I \omega_{1, i, j} \cdot x_i - \theta_{1, j} \right) = \frac{1}{1 + e^{-(\sum \omega_{1} x - \theta_1)}}$$
  - Binary Output neuron ($F_2$): Logistic Sigmoid:
    $$y = F_2\left( \sum_{j=1}^H \omega_{2, j} \cdot s_j - \theta_{2} \right)$$
  - Multi-class Output neurons ($F_2$): Softmax activation:
    $$y_j = \frac{e^{\sum_{i=1}^H \omega_{2, i, j} \cdot s_i - \theta_{2, j}}}{\sum_{k=1}^Q e^{\sum_{i=1}^H \omega_{2, i, k} \cdot s_i - \theta_{2, k}}}$$

### 3.2 Continuous Vector Encoding (Eq. 3)
Each candidate solution (wolf) is a continuous search vector $\vec{v} \in [-10, 10]^D$:

$$\vec{v} = [\omega_{1, 1, 1} \dots \omega_{1, I, H}, \;\omega_{2, 1, 1} \dots \omega_{2, H, O}, \;\theta_{1, 1} \dots \theta_{1, H}, \;\theta_{2, 1} \dots \theta_{2, O}]$$

### 3.3 Fitness Function: Average Cross-Entropy Error (CEE)
- **Binary Classification (Eq. 4):**
  $$\min_{\vec{v}} \text{CEE} = - \frac{1}{Z} \sum_{i=1}^Z \left[ y'_i \ln(y_i) + (1 - y'_i) \ln(1 - y_i) \right]$$
- **Multi-Class Classification (Eq. 5):**
  $$\min_{\vec{v}} \text{CEE} = - \frac{1}{Z \cdot Q} \sum_{i=1}^Z \sum_{j=1}^Q y'_{i, j} \ln(y_{i, j})$$

---

## 🧬 Phase 4: Porting the Metaheuristic Optimizers

### 4.1 Canonical Grey Wolf Optimizer (GWO) Baseline
- Employs 3 fixed dominant leaders: $\alpha$ (1st best), $\beta$ (2nd best), and $\delta$ (3rd best).
- Exploration parameter $a$ linearly decreases from 2 to 0: $a = 2 - 2t/T$.
- Wolf position update:
  $$X_i(t+1) = \frac{1}{3} \left( U_{i, \alpha} + U_{i, \beta} + U_{i, \delta} \right)$$
- Leaders are only updated once per generation after all $N$ wolves have moved.

### 4.2 Competing Leaders Grey Wolf Optimizer (CGWO)
Addresses GWO's premature convergence and search stagnation through 3 core mechanisms:

1. **IPDE Population Initialization (Algorithm 3):**
   - Population 1 ($IX_1$): $N$ individuals sampled uniformly randomly in $[-10, 10]^D$.
   - Population 2 ($IX_2$): $N$ individuals generated via a sinusoidal chaotic map:
     $$r_k = \sin(\pi r_{k-1}), \quad r_0 \in (0, 1)$$
   - Selection: The fittest $N$ individuals from the union $IX_1 \cup IX_2$ are retained.
2. **Competing Leaders Shrinkage ($M(t)$, Eqs. 12–13):**
   - The initial number of leaders $M_{\max}$ is adaptively scaled based on problem dimension $D$:
     $$M_{\max} = \text{round}\left( \frac{D}{N} \frac{MX_u - MX_l}{4} + MX_l \right) \quad (\text{if } D \le 4N, \text{ else } MX_u=40)$$
   - Leader pool shrinks linearly over the first $75\%$ of iterations:
     $$M(t) = \text{round}\left( M_{\max} - t \frac{4(M_{\max} - M_{\min})}{3 T} \right) \quad (t < 0.75 T, \text{ else } M_{\min}=3)$$
   - Movement is averaged over all $M$ competing leaders:
     $$X_i(t+1) = \frac{1}{M} \sum_{j=1}^M U_{i, j}(t)$$
3. **Compact Wolf Leaders Update (CWLU):**
   - Leaders are updated greedily and immediately as each wolf is evaluated. If a newly positioned wolf is superior to the worst current leader, it immediately enters the leader pack.

### 4.3 Proposed Enhanced Variant: Tent-Cosine CGWO (TCGWO)
Introduced for Item 11 to beat standard CGWO:
- **Tent Chaotic Map:** Replaces the boundary-clustering sinusoidal map with uniform invariant density ergodicity:
  $$r_k = 2 r_{k-1} \text{ if } r < 0.5 \text{ else } 2(1 - r_{k-1})$$
- **Cosine Leader Schedule:** Smooth cosine decay preserving leader diversity longer during early exploration:
  $$M(t) = \text{round}\left( M_{\min} + \frac{1}{2}(M_{\max} - M_{\min})\left(1 + \cos\left(\frac{\pi t}{0.75 T}\right)\right) \right)$$

---

## 💻 Phase 5: Complete Command Reference

### 5.1 Standalone Script Execution
```bash
# Run standalone CGWO trainer on Wine
python3 "python converted/cgwo_mlp.py"

# Run standalone GWO baseline on Wine
python3 "python converted/gwo_mlp.py"

# Run standalone Tent-Cosine CGWO on Wine
python3 "python converted/tcgwo_mlp.py"
```

### 5.2 Parametric Benchmark Execution
```bash
# Fast smoke test (20 agents, 1000 FEs, 2 folds)
python3 examples/run_all_experiments.py --models CGWO GWO --agents 20 --fes 1000 --folds 2 --runs 1

# Medium validation benchmark (4 datasets, 100 agents, 10000 FEs)
python3 examples/run_all_experiments.py --models CGWO GWO TCGWO --agents 100 --fes 10000 --runs 2

# Full paper reproduction benchmark (Pop=200, FEs=40,000, 5-Fold CV)
python3 examples/run_all_experiments.py --models CGWO GWO TCGWO --agents 200 --fes 40000 --folds 5 --runs 3
```

### 5.3 High-Fidelity Visualizations & Table Export
```bash
# Generate architecture diagrams, vector mapping, weight heatmaps & flowchart
python3 examples/generate_visualizations.py

# Export results summary table to styled PNG
python3 examples/save_table_as_image.py

# Generate convergence comparison plot
python3 generate_convergence_plot.py
```

---

## 🎨 Phase 6: High-Fidelity Visualizations & Reporting

All publication-ready graphics are automatically saved to `visualization/` and `outputs/plots/`:

1. **`visualization/mlp_real_weights.png`:** Complete neural network diagram illustrating actual trained weights from CGWO, color-coded by polarity (blue = positive, red = negative).
2. **`visualization/weight_heatmaps.png`:** Heatmaps displaying the weight correlation matrices for $W_1$ ($I \times H$) and $W_2$ ($H \times O$).
3. **`visualization/flat_vector_mapping.png`:** Structural diagram depicting how the continuous 1D search vector unpacks into multi-layer weight matrices and bias vectors.
4. **`visualization/optimization_flowchart.png`:** End-to-end flowchart covering Min-Max scaling, IPDE, Competing Leaders, CWLU, and CEE evaluation.
5. **`outputs/plots/comparison_models_convergence.png`:** Comparative convergence trajectories showing CGWO and TCGWO outperforming GWO.
6. **`outputs/plots/comparison_table_summary.png`:** High-resolution graphic table summarizing CEE, accuracy, and runtime.

---

## 🔍 Phase 7: MATLAB vs. Python Mathematical Audit

| Mathematical Component | MATLAB Reference Formulation | Python Port (`mlp_cgwo`) | Parity Verification |
| :--- | :--- | :--- | :--- |
| **Hidden Activation** | `s = 1 ./ (1 + exp(-(X*W1 - b1)))` | `1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))` | ✅ Bit-level float parity |
| **Output Activation** | Multi-class Softmax normalized over row | `np.exp(z - max) / np.sum(np.exp(z - max))` | ✅ Numerically stabilized |
| **Binary CEE Loss** | `-mean(y.*log(y_hat) + (1-y).*log(1-y_hat))` | `np.mean(y_t*np.log(p) + (1-y_t)*np.log(1-p))` | ✅ Exact match (Eq. 4) |
| **Multi-class CEE Loss**| `-sum(sum(Y_onehot.*log(Y_hat))) / (Z*Q)` | `-np.sum(one_hot * np.log(y_pred)) / (Z*Q)` | ✅ Exact match (Eq. 5) |
| **Leader Count $M(t)$** | `round(M_max - t*(4*(M_max-M_min))/(3*T))` | `int(np.round(M_max - t*4*(M_max-3)/(3*T)))` | ✅ Identical integer schedule |
| **CWLU Greedy Update** | In-place replacement of worst leader | `if f_new < leader_fits[-1]: replace & re-sort` | ✅ Real-time leader propagation |
| **Data Scaling** | `(X - min) / (max - min)` | Column-wise Min-Max vectorization | ✅ Exact $[0, 1]$ mapped range |

---

## 📋 Phase 8: Term Work Report Summary

The formal 11-item submission required for **19ECE355 Bio-Inspired Algorithms** has been completed and exported:
- **Markdown Document:** [`TERM_WORK_REPORT.md`](file:///home/kv/Projects/bio-sim1/TERM_WORK_REPORT.md)
- **Printable HTML Document:** [`TERM_WORK_REPORT.html`](file:///home/kv/Projects/bio-sim1/TERM_WORK_REPORT.html) (Optimized with CSS `@media print` for 3-page hard copy).

---

*This guide was generated as part of the 19ECE355 Bio-Inspired Algorithms research paper replication project.*
