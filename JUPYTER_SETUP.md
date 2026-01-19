# Jupyter Notebook Setup Guide

## Quick Start

### 1. Activate Virtual Environment
```bash
cd /Users/kalirajannatarajan/projects/autoencoder
source venv/bin/activate
```

### 2. Start Jupyter
```bash
# Option A: Jupyter Notebook (classic interface)
jupyter notebook

# Option B: JupyterLab (modern interface)
jupyter lab
```

### 3. Select Kernel
When opening a notebook, select **"Python (autoencoder)"** as the kernel.

This kernel uses your virtual environment with all required packages (TensorFlow, NumPy, Pillow).

---

## Notebooks Available

1. **`01_baseline_autoencoder.ipynb`**
   - Model 1: Baseline autoencoder (latent_dim=64)
   - Textbook version implementation

2. **`02_model2_reduced_latent.ipynb`**
   - Model 2: Modified autoencoder (latent_dim=32)
   - Demonstrates compression vs. quality trade-off

---

## Troubleshooting

**Kernel not showing up?**
```bash
source venv/bin/activate
python -m ipykernel install --user --name=autoencoder --display-name "Python (autoencoder)"
```

**Jupyter not starting?**
```bash
source venv/bin/activate
pip install jupyter ipykernel
```

---

## Running Notebooks

1. Open the notebook in Jupyter
2. Select kernel: **"Python (autoencoder)"**
3. Run cells sequentially (Shift+Enter)
4. All outputs (images, summaries) will be saved to `reconstructions/` directory
