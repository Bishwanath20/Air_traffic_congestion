import os
import numpy as np

SPATIAL_TENSOR_DIR = "processed/spatial_tensors"
OUT_DIR = "processed/ml_tensors"
os.makedirs(OUT_DIR, exist_ok=True)

# Pick one representative day
DAY_FILE = "2019-01-14.npy"
tensor_path = os.path.join(SPATIAL_TENSOR_DIR, DAY_FILE)

print(f"📂 Loading spatial tensor: {DAY_FILE}")
tensor = np.load(tensor_path, mmap_mode="r")

# tensor shape: (24, H, W, Z)
# We build a small temporal window (e.g., hours 12–18)
X_sample = tensor[12:18]          # (6, H, W, Z)

# Reduce altitude dimension (mean over Z)
X_sample = X_sample.mean(axis=-1) # (6, H, W)

# Downsample for XAI clarity + speed
X_sample = X_sample[:, ::12, ::12]  # (6, 60, 60)

# Add batch + channel dims → (1, 6, 60, 60, 1)
X_sample = X_sample[np.newaxis, ..., np.newaxis].astype("float32")

out_path = os.path.join(OUT_DIR, "X_sample.npy")
np.save(out_path, X_sample)

print(f"✅ XAI sample saved to: {out_path}")
print("Final shape:", X_sample.shape)