# Fashion-MNIST Autoencoders

Implementation of dense autoencoders on **Fashion-MNIST**, based on the textbook notebook from [*Generative Deep Learning*, 2nd Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition) (Chapter 3: Autoencoders).

## Overview

This project contains two autoencoder models for direct comparison:

| Model | Latent Dim | Description |
|-------|------------|-------------|
| **Model 1** | 64 | Baseline autoencoder (textbook version) |
| **Model 2** | 32 | Modified autoencoder with reduced latent space |

Both use the same encoder/decoder architecture, optimizer, loss, and training settings. Model 2 demonstrates how **reducing latent dimension** increases information loss and reconstruction error.

---

## Project Structure

```
autoencoder/
├── 01_baseline_autoencoder.ipynb    # Model 1: latent_dim=64 (Jupyter)
├── 02_model2_reduced_latent.ipynb    # Model 2: latent_dim=32 (Jupyter)
├── baseline_autoencoder.py            # Model 1: latent_dim=64 (Python script)
├── model2_autoencoder.py              # Model 2: latent_dim=32 (Python script)
├── reconstructions/                   # Output images
│   ├── original_1.png .. original_5.png
│   ├── reconstructed_1.png .. reconstructed_5.png      # Model 1
│   ├── reconstructed_model2_1.png .. reconstructed_model2_5.png  # Model 2
│   └── original_images_collage.png
├── requirements.txt
└── README.md
```

---

## Architecture (Both Models)

- **Encoder:** `784 → 256 → 128 → 64 → latent` (ReLU)
- **Decoder:** `latent → 64 → 128 → 256 → 784` (sigmoid on output)
- **Optimizer:** RMSprop (lr=0.001)
- **Loss:** Mean Squared Error (MSE)
- **Training:** 20 epochs, batch_size=128

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kalirajann/autoencoder.git
cd autoencoder
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Option 1: Jupyter Notebooks (Recommended)

1. **Start Jupyter:**
   ```bash
   source venv/bin/activate
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Open notebooks:**
   - `01_baseline_autoencoder.ipynb` - Model 1 (latent_dim=64)
   - `02_model2_reduced_latent.ipynb` - Model 2 (latent_dim=32)

3. **Select kernel:** Choose "Python (autoencoder)" when prompted

4. **Run cells:** Execute cells sequentially (Shift+Enter)

### Option 2: Python Scripts

**Run Model 1 (Baseline, latent_dim=64):**
```bash
source venv/bin/activate
python baseline_autoencoder.py
```

**Run Model 2 (Reduced latent, latent_dim=32):**
```bash
source venv/bin/activate
python model2_autoencoder.py
```

Both scripts:
- Train on Fashion-MNIST
- Save original and reconstructed images in `reconstructions/`
- Print encoder/decoder/autoencoder summaries, training loss per epoch, and final test reconstruction loss

---

## Sample Results

| Model | Latent Dim | Final Train Loss | Final Test Loss |
|-------|------------|------------------|-----------------|
| Model 1 | 64 | ~0.018 | ~0.018 |
| Model 2 | 32 | ~0.019 | ~0.018 |

Model 2’s higher reconstruction loss illustrates the **compression vs. quality** trade-off when shrinking the latent bottleneck.

---

## Outputs

- **`reconstructions/original_*.png`** — Original Fashion-MNIST test images (indices 0–4)
- **`reconstructions/reconstructed_*.png`** — Reconstructions from Model 1
- **`reconstructions/reconstructed_model2_*.png`** — Reconstructions from Model 2
- **`reconstructions/original_images_collage.png`** — Collage of the 5 original images

---

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pillow (PIL)
- Jupyter (for notebooks)
- ipykernel (for Jupyter kernel)

---

## Reference

- Notebook: [autoencoder.ipynb](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/03_vae/01_autoencoder/autoencoder.ipynb)
- Book: *Generative Deep Learning*, 2nd Ed. — David Foster (O’Reilly)

---

## License

MIT

