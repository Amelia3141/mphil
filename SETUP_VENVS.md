# Setup Instructions: Separate Virtual Environments

Complete guide for running sensitivity analysis and model selection in isolated environments to avoid dependency conflicts.

---

## Why Separate Venvs?

Different analyses may have conflicting dependencies or versions. Using separate virtual environments ensures:
- No dependency conflicts
- Reproducible results
- Clean, isolated environments
- Easy debugging

---

## Quick Start (Copy-Paste Ready)

### For Sensitivity Analysis:

```bash
# Clone and setup
git clone https://github.com/Amelia3141/mphil.git
cd mphil
git checkout claude/convert-to-jupyter-01GY8iZvAixjYs4t3VyLsWHf

# Create venv
python3 -m venv sensitivity_env
source sensitivity_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch numpy scipy matplotlib pandas seaborn scikit-learn tqdm pathos tabulate

# Run
python sensitivity_analysis.py

# When done
deactivate
```

### For Model Selection:

```bash
# Navigate to repo (or clone if not already)
cd mphil

# Create separate venv
python3 -m venv model_selection_env
source model_selection_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch numpy scipy matplotlib pandas seaborn scikit-learn tqdm pathos tabulate

# Run
python model_selection.py --k-max 5 --true-k 3

# When done
deactivate
```

---

## Detailed Setup

### 1. System Requirements

**Minimum:**
- Python 3.8+
- 16GB RAM
- NVIDIA GPU with CUDA (or use `--no-gpu` for CPU)

**Recommended:**
- Python 3.10+
- 32GB RAM
- NVIDIA GPU (T4, V100, or A100)
- CUDA 11.0+

### 2. Check GPU Availability

```bash
# Check if CUDA is available
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check GPU details
nvidia-smi
```

### 3. Environment A: Sensitivity Analysis

```bash
# Step 1: Create virtual environment
python3 -m venv sensitivity_env

# Step 2: Activate
source sensitivity_env/bin/activate  # Linux/Mac
# OR
sensitivity_env\Scripts\activate  # Windows

# Step 3: Verify activation
which python  # Should show path to sensitivity_env

# Step 4: Install dependencies
pip install --upgrade pip
pip install torch numpy scipy matplotlib pandas seaborn scikit-learn tqdm

# For sensitivity analysis specifically:
pip install pathos tabulate

# Step 5: Test installation
python -c "import torch; import numpy; import seaborn; print('OK')"

# Step 6: Run sensitivity analysis
python sensitivity_analysis.py --experiments 1 2 3 4 5

# Alternative: Run specific experiments
python sensitivity_analysis.py --experiments 1 2
python sensitivity_analysis.py --experiments 3 4 5

# Alternative: CPU only (slower)
python sensitivity_analysis.py --no-gpu

# Step 7: Check outputs
ls sensitivity_output/

# Step 8: Deactivate when done
deactivate
```

**Expected Runtime:**
- With GPU (T4): ~25-30 minutes for all 5 experiments
- With GPU (A100): ~15-20 minutes
- With CPU: ~8-10 hours

**Output Files:**
```
sensitivity_output/
├── figures/
│   ├── exp1_vary_subtypes.png
│   ├── exp2_missing_data.png
│   ├── exp3_sample_size.png
│   ├── exp4_noise_levels.png
│   └── exp5_combined_stress.png
├── tables/
│   ├── experiment_1_vary_subtypes.csv
│   ├── experiment_2_missing_data.csv
│   ├── experiment_3_sample_size.csv
│   ├── experiment_4_noise_levels.csv
│   └── experiment_5_combined_stress.csv
└── SENSITIVITY_ANALYSIS_REPORT.md
```

---

### 4. Environment B: Model Selection

```bash
# Step 1: Create separate virtual environment
python3 -m venv model_selection_env

# Step 2: Activate
source model_selection_env/bin/activate  # Linux/Mac
# OR
model_selection_env\Scripts\activate  # Windows

# Step 3: Verify activation
which python  # Should show path to model_selection_env

# Step 4: Install dependencies
pip install --upgrade pip
pip install torch numpy scipy matplotlib pandas seaborn scikit-learn tqdm

# For model selection specifically:
pip install pathos tabulate

# Step 5: Test installation
python -c "import torch; import numpy; print('OK')"

# Step 6: Run model selection
python model_selection.py --k-min 1 --k-max 5 --true-k 3

# Alternative: More CV folds (more robust but slower)
python model_selection.py --k-max 5 --n-cv-folds 10

# Alternative: Larger dataset
python model_selection.py --k-max 5 --n-subjects 5000

# Alternative: CPU only
python model_selection.py --k-max 5 --no-gpu

# Step 7: Check outputs
ls model_selection_output/

# Step 8: Deactivate when done
deactivate
```

**Expected Runtime:**
- With GPU (T4): ~45-60 minutes for k=1 to k=5 (5-fold CV)
- With GPU (A100): ~25-35 minutes
- With CPU: ~12-15 hours

**Output Files:**
```
model_selection_output/
├── figures/
│   ├── model_selection_comprehensive.png
│   └── cvic_detailed.png
├── tables/
│   └── model_selection_results.csv
├── models/  (trained models for each k)
└── MODEL_SELECTION_REPORT.md
```

---

## Running Both Scripts

### Option 1: Sequential

```bash
# Sensitivity first
source sensitivity_env/bin/activate
python sensitivity_analysis.py
deactivate

# Then model selection
source model_selection_env/bin/activate
python model_selection.py --k-max 5
deactivate
```

### Option 2: Parallel (if you have multiple GPUs)

```bash
# Terminal 1: Sensitivity on GPU 0
source sensitivity_env/bin/activate
python sensitivity_analysis.py --device 0
deactivate

# Terminal 2: Model selection on GPU 1
source model_selection_env/bin/activate
python model_selection.py --k-max 5 --device 1
deactivate
```

---

## Troubleshooting

### Issue: "python3: command not found"
```bash
# Use python instead
python -m venv sensitivity_env
```

### Issue: "No module named 'torch'"
```bash
# Ensure venv is activated
which python  # Should show venv path

# Reinstall torch
pip install --force-reinstall torch
```

### Issue: "CUDA not available"
```bash
# Check CUDA installation
nvidia-smi

# Try CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "Out of memory"
```bash
# Reduce parameters
python sensitivity_analysis.py --experiments 1  # One at a time
python model_selection.py --k-max 3  # Fewer models
```

### Issue: "Permission denied"
```bash
# Make scripts executable
chmod +x sensitivity_analysis.py model_selection.py
```

---

## Cleanup

```bash
# Remove virtual environments when done
rm -rf sensitivity_env/
rm -rf model_selection_env/

# Remove output directories
rm -rf sensitivity_output/
rm -rf model_selection_output/
```

---

## For Google Colab

If running on Colab, you don't need venvs (each notebook is isolated):

```python
# In Colab cell:
!git clone https://github.com/Amelia3141/mphil.git
%cd mphil
!git checkout claude/convert-to-jupyter-01GY8iZvAixjYs4t3VyLsWHf

!pip install torch numpy scipy matplotlib pandas seaborn scikit-learn tqdm pathos tabulate

# Run sensitivity
!python sensitivity_analysis.py

# Run model selection
!python model_selection.py --k-max 5
```

---

## For GBSH Servers

```bash
# SSH to GBSH
ssh your_username@gbsh.server

# Load CUDA module (if needed)
module load cuda/11.0

# Follow standard venv setup above
# Use screen or tmux for long runs
screen -S sensitivity
source sensitivity_env/bin/activate
python sensitivity_analysis.py
# Ctrl+A, D to detach

# Check progress
screen -r sensitivity
```

---

## Summary

**Two separate environments:**
1. `sensitivity_env/` for sensitivity analysis
2. `model_selection_env/` for model selection

**Both require same core dependencies:**
- torch, numpy, scipy, matplotlib, pandas, seaborn, scikit-learn, tqdm, pathos, tabulate

**Key differences:**
- Sensitivity: Tests robustness across 5 experiments
- Model selection: Determines optimal k via CVIC

**Combined runtime:**
- GPU: ~1-1.5 hours total
- CPU: ~20-25 hours total

**All outputs are publication-ready figures and tables!**
