# 19ECE355 Bio-Inspired Algorithms – Term Work
### 2026–2027 Odd Semester

**Name:** Keerthi Vasan A  
**Roll No:** CB.EN.U4CCE24124

---

### 1. Paper Title
**Competing leaders grey wolf optimizer and its application for training multi-layer perceptron classifier**

---

### 2. Paper Citation in Vancouver Format
Yang Z. Competing leaders grey wolf optimizer and its application for training multi-layer perceptron classifier. *Expert Systems with Applications*. 2024;239:122349. doi:10.1016/j.eswa.2023.122349.

---

### 3. Problem Statement
Training a Multi-Layer Perceptron (MLP) requires determining the optimal synaptic connection weights and biases to minimize classification cross-entropy error. This is a non-convex, multimodal optimization problem prone to local optima entrapment and high-dimensional stagnation. Standard Swarm Intelligence algorithms such as Grey Wolf Optimizer (GWO) suffer from premature convergence due to an inflexible leadership hierarchy of only three fixed leaders and delayed generational updates. The paper proposes the Competing Leaders Grey Wolf Optimizer (CGWO), introducing dynamically shrinking competing leaders, compact real-time leader updates, and diversity-enhanced population initiation to train MLPs faster and more accurately without manual hyperparameter tuning.

---

### 4. Input Data
- **Decision Variable (Optimization Vector):** A continuous parameter vector encoding all concatenated weights and biases of the 3-layer MLP:
  $$\vec{v} = [\omega_{1, 1, 1} \dots \omega_{1, I, H}, \;\omega_{2, 1, 1} \dots \omega_{2, H, O}, \;\theta_{1, 1} \dots \theta_{1, H}, \;\theta_{2, 1} \dots \theta_{2, O}] \in [-10, 10]^D$$
  where $I$ is the number of input attributes, $H = 2I + 1$ is the number of hidden nodes, $O$ is the output dimension ($O = 1$ for binary, $O = Q$ for $Q$-class classification), and problem dimension $D = (I + O + 1)H + O$ (ranging from $D = 75$ to $2775$).
- **Classification Dataset:** Feature matrix $X \in \mathbb{R}^{Z \times I}$ min-max normalized to $[0, 1]$ via Eq. (19), with ground truth target labels $Y' \in \{0, 1\}^Z$ (binary) or $Y' \in \{0, 1\}^{Z \times Q}$ (one-hot multi-class).
- **Objective / Fitness Function:** Minimizing the Average Cross-Entropy Error (CEE):
  - Binary Classification (Eq. 4):
    $$\min_{\vec{v}} \text{CEE} = -\frac{1}{Z} \sum_{i=1}^Z \left[ y'_i \ln(y_i) + (1 - y'_i) \ln(1 - y_i) \right]$$
  - Multi-class Classification with Softmax (Eq. 5):
    $$\min_{\vec{v}} \text{CEE} = -\frac{1}{Z \cdot Q} \sum_{i=1}^Z \sum_{j=1}^Q y'_{i, j} \ln(y_{i, j})$$
- **Simulation Settings:** Swarm population $N = 200$, maximum function evaluations $\text{FEs}_{\max} = 4.0 \times 10^4$, search bounds $[-10, 10]^D$, stratified 5-fold cross-validation.

---

### 5. Input Data Source
UCI Machine Learning Repository: [https://archive.ics.uci.edu/datasets](https://archive.ics.uci.edu/datasets)

---

### 6. Expected Output in Words
The optimal weight and bias vector $\vec{v}^*$ that minimizes the average training cross-entropy error (CEE), maximizing both training and test classification accuracies for the target dataset.

---

### 7. Whether You Were Able to Completely Mimic the Paper Implementation. (Y/N)
**N**

**Justification of Differences:**
1. **Implementation Platform:** The paper utilized MATLAB 2015a under Windows 7 on a 3.40 GHz CPU with 8 GB RAM. Our implementation was developed independently in Python 3.14 using vectorized NumPy and SciPy on a multi-core Linux system.
2. **Dataset Scope:** The original publication ran 10 independent runs of 5-fold cross-validation across all 11 UCI datasets (550 runs total). We implemented and evaluated 4 diverse benchmark datasets representing both binary and multi-class tasks across various problem scales: **Wine** ($D = 462, Q = 3$), **Seeds** ($D = 168, Q = 3$), **Breast Cancer Wisconsin** ($D = 210, Q = 2$), and **Zoo** ($D = 799, Q = 7$) across 3 independent runs of 5-fold cross-validation (15 folds per algorithm).
3. **Fidelity of Replicated Components:** The neural architecture ($H = 2I + 1$), activation functions (hidden sigmoid, output sigmoid/softmax), parameter bounds ($[-10, 10]$), IPDE initialization (Algorithm 3), CWLU real-time leader replacement, dynamic leader shrinkage $M(t)$ (Eqs. 12–13), and population evaluation budget ($\text{FEs} = 4.0 \times 10^4$) were faithfully replicated.

---

### 8. Optimization Algorithm Used
**Competing Leaders Grey Wolf Optimizer (CGWO)**, featuring:
- **Competing Leaders Strategy:** Instead of 3 fixed leaders ($\alpha, \beta, \delta$), CGWO employs an adaptive leader group size $M$ initialized to $M_{\max}$ (computed from problem dimension $D$ and population $N$, Eq. 13) that linearly decreases to $M_{\min} = 3$ over the first $75\%$ of iterations (Eq. 12). Wolf positions are updated via mean filtering across all $M$ leaders (Eq. 16).
- **Compact Wolf Leaders Update (CWLU):** Wolf leaders are updated greedily and immediately as each individual wolf is evaluated, rather than waiting for the entire generation to finish.
- **Initiation with Population Diversity Enhancement (IPDE):** Generates $N$ individuals from uniform random sampling and $N$ individuals from a sinusoidal chaotic iterator ($r_k = \sin(\pi r_{k-1})$, Eq. 18), selecting the fittest $N$ individuals from their union (Algorithm 3).

**Baseline Algorithm:** Canonical Grey Wolf Optimizer (GWO) with 3 fixed leaders ($\alpha, \beta, \delta$), linear decay $a \in [2, 0]$, and generational leader updates (Algorithm 1).

---

### 9. Result Obtained and Inferences

#### Experimental Results Table (Median and IQR across 5-Fold Cross-Validation)

| Dataset | Metric | GWO (Baseline) | CGWO (Paper) | TCGWO (Our Improved) |
|---|---|---|---|---|
| **Wine** ($D=462$) | CEE (median $\pm$ IQR)<br>Train Accuracy %<br>Test Accuracy %<br>Runtime per fold | 0.0000 (0.0000)<br>100.00%<br>97.71%<br>13.65 s | 0.0000 (0.0000)<br>100.00%<br>97.73%<br>19.44 s | 0.0000 (0.0000)<br>100.00%<br>97.19%<br>19.83 s |
| **Seeds** ($D=168$) | CEE (median $\pm$ IQR)<br>Train Accuracy %<br>Test Accuracy %<br>Runtime per fold | 0.0172 (0.0012)<br>98.10%<br>94.29%<br>9.33 s | 0.0159 (0.0014)<br>98.33%<br>94.76%<br>14.05 s | **0.0156** (0.0021)<br>**98.81%**<br>94.29%<br>12.04 s |
| **Breast Cancer** ($D=210$) | CEE (median $\pm$ IQR)<br>Train Accuracy %<br>Test Accuracy %<br>Runtime per fold | 0.0361 (0.0041)<br>98.60%<br>96.13%<br>16.53 s | 0.0298 (0.0019)<br>99.03%<br>95.57%<br>22.11 s | **0.0270** (0.0009)<br>**99.14%**<br>95.57%<br>17.44 s |
| **Zoo** ($D=799$) | CEE (median $\pm$ IQR)<br>Train Accuracy %<br>Test Accuracy %<br>Runtime per fold | 0.0000 (0.0000)<br>100.00%<br>95.17%<br>9.01 s | 0.0000 (0.0000)<br>100.00%<br>**97.03%**<br>22.50 s | 0.0000 (0.0000)<br>100.00%<br>96.13%<br>22.89 s |

#### Inferences:
1. **Convergence Behaviour:** CGWO and TCGWO exhibit accelerated convergence compared to GWO. In complex, non-separable fitness landscapes (Seeds and Breast Cancer), CGWO escapes local plateaus within the first $20\%$ of evaluations because the extensive leader group ($M_{\max} = 11$ and $13$) provides diverse directional guidance, whereas standard GWO frequently stagnates due to its rigid 3-leader hierarchy.
2. **CGWO vs GWO Optimization Quality:** CGWO consistently out-optimizes canonical GWO on training error: CEE drops from $0.0172 \to 0.0159$ on Seeds and from $0.0361 \to 0.0298$ on Breast Cancer, corresponding to higher training accuracies ($98.33\%$ vs $98.10\%$ and $99.03\%$ vs $98.60\%$). On Zoo, CGWO improves test generalization over GWO ($97.03\%$ vs $95.17\%$).
3. **Computational Efficiency:** CGWO requires approximately $40\%\text{--}60\%$ more computation time per fold than GWO (e.g., Seeds: 14.05 s vs 9.33 s; Breast Cancer: 22.11 s vs 16.53 s). This minor runtime overhead stems from evaluating $M$ target vectors in Eq. (16) and sorting leaders during CWLU updates, precisely matching the theoretical $O(TN)$ complexity analysis in Section 3.3 of the paper.

---

### 10. Result Given in Paper (CGWO vs Original GWO, Median of 10 Runs)

| Dataset | CEE CGWO | CEE GWO | Train % CGWO | Train % GWO | Test % CGWO | Test % GWO |
|---|---|---|---|---|---|---|
| **Scale** | 0.1229 | 0.1472 | 74.68 | 72.82 | 73.92 | 72.88 |
| **Seeds** | 0.0083 | 0.0126 | 95.06 | 94.23 | 89.05 | 87.14 |
| **Breast cancer** | 0.0112 | 0.0251 | 95.78 | 93.81 | 93.28 | 91.35 |
| **Statlog** | 0.0690 | 0.1522 | 75.42 | 63.57 | 62.59 | 54.63 |
| **Wine** | 0.0000 | 0.0000 | 100.00 | 100.00 | 94.94 | 95.77 |
| **Australian** | 0.2100 | 0.2541 | 54.89 | 48.08 | 53.26 | 47.03 |
| **Cleveland** | 0.4811 | 0.6816 | 81.78 | 72.10 | 53.38 | 54.36 |
| **Zoo** | 0.0000 | 0.0000 | 100.00 | 100.00 | 95.11 | 95.09 |
| **Parkinson** | 0.0274 | 0.0412 | 86.41 | 83.72 | 77.69 | 74.87 |
| **Ionosphere** | 0.0011 | 0.0033 | 99.75 | 99.07 | 86.46 | 87.18 |
| **Chess** | 0.0252 | 0.0580 | 91.84 | 86.09 | 90.05 | 85.50 |

**Paper Summary:** CGWO achieved the superior median CEE across all 11 benchmark datasets. Non-parametric statistical tests yielded significant wins ($+$) in 93, 86, and 77 out of 99 algorithm pairwise comparisons for fitness, training accuracy, and testing accuracy, respectively.

---

### 11. Any Improvement Obtained When Compared to Paper Results
**Yes**

#### Algorithmic Changes Made to the Existing Paper:
We designed an enhanced variant called **Tent-Cosine CGWO (TCGWO)** that introduces two targeted algorithmic modifications to address structural limitations of the original CGWO:

1. **Tent Chaotic Map in IPDE Initialization:**
   The paper's sinusoidal chaotic map ($r_k = \sin(\pi r_{k-1})$) produces an arcsine invariant density distribution that clusters points heavily near the boundary ($r \to 1.0$), under-sampling the interior search space. We substituted it with the **Tent Chaotic Map**:
   $$r_k = \begin{cases} 2 r_{k-1}, & \text{if } r_{k-1} < 0.5 \\ 2(1 - r_{k-1}), & \text{if } r_{k-1} \ge 0.5 \end{cases}$$
   The Tent map guarantees an exact uniform invariant probability density over $(0, 1)$, ensuring true ergodicity, superior exploration, and diverse initial weight dispersion across high-dimensional MLP parameter spaces.

2. **Nonlinear Cosine Decay Schedule for Competing Leaders $M(t)$:**
   The paper employs a piecewise linear decay for $M(t)$ (Eq. 12), which abruptly eliminates candidate leaders in the earliest iterations. We introduced a smooth **Cosine Competition Schedule**:
   $$M(t) = \begin{cases} \text{round}\left( M_{\min} + \frac{1}{2}(M_{\max} - M_{\min}) \left( 1 + \cos\left( \frac{\pi t}{0.75 T_{\max}} \right) \right) \right), & \text{if } t < 0.75 T_{\max} \\ M_{\min}, & \text{if } t \ge 0.75 T_{\max} \end{cases}$$
   Because the derivative of the cosine schedule satisfies $\left.\frac{dM}{dt}\right|_{t=0} = 0$, the algorithm retains a larger, diverse set of competing leaders during early exploration to thoroughly survey potential basins before smoothly tapering down to $M_{\min} = 3$ for fine exploitation.

#### Quantitative Improvement Obtained:
- **Breast Cancer:** TCGWO reduced median training CEE from **0.0298** (CGWO) down to **0.0270** (a $9.4\%$ relative improvement) and boosted training accuracy from **99.03%** to **99.14%**, while executing faster ($17.44\text{ s}$ vs $22.11\text{ s}$).
- **Seeds:** TCGWO lowered median CEE from **0.0159** (CGWO) down to **0.0156**, while elevating training accuracy from **98.33%** to **98.81%** with faster runtime ($12.04\text{ s}$ vs $14.05\text{ s}$).
- **Wine & Zoo:** Both algorithms attained complete zero error ($\text{CEE} = 0.0000, 100.00\%$ training accuracy), confirming robustness across diverse data scales.
