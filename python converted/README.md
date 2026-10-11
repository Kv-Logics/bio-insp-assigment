# Standalone Python Converted Scripts

This directory contains standalone, independent Python scripts replicating the core algorithms from:
> **Yang Z. (2024)**, *Competing leaders grey wolf optimizer and its application for training multi-layer perceptron classifier*, Expert Systems with Applications.

Each script can be run independently without any external framework dependencies (requires only standard `numpy`):

- **`cgwo_mlp.py`**: Standalone implementation of the proposed **Competing Leaders Grey Wolf Optimizer (CGWO)** featuring IPDE initialization, dynamic leader shrinkage $M(t)$, and Compact Wolf Leaders Update (CWLU).
- **`gwo_mlp.py`**: Standalone baseline implementation of the canonical **Grey Wolf Optimizer (GWO)** (Mirjalili, 2015).
- **`tcgwo_mlp.py`**: Standalone implementation of our improved variant, **Tent-Cosine CGWO (TCGWO)**, incorporating the Tent Chaotic Map and smooth Cosine competition schedule.

### Quick Usage:
```bash
python3 "python converted/cgwo_mlp.py"
python3 "python converted/gwo_mlp.py"
python3 "python converted/tcgwo_mlp.py"
```
