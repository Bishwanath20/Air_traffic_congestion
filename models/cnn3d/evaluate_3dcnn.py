import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# ======================
# CONFIG
# ======================
DEVICE = "cpu"
MODEL_PATH = r"d:\projects\data\models\cnn3d\cnn3d_model.pth"
DATA_PATH = r"d:\projects\data\processed\ml_tensors\X.dat"
OUT_PLOT = r"d:\projects\data\models\cnn3d\cnn3d_prediction.png"

# ======================
# MODEL DEFINITION
# ======================
class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 8, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

# ======================
# LOAD MODEL
# ======================
model = CNN3D().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======================
# LOAD DATA
# ======================
import json
META_PATH = r"d:\projects\data\processed\ml_tensors\shapes.json"
with open(META_PATH) as f:
    meta = json.load(f)

# Load only first sample and extract a patch for evaluation
X = np.memmap(DATA_PATH, dtype=np.float32, mode="r", shape=(meta["num_samples"],) + tuple(meta["input_shape"]))

# Shape of X is (num_samples, 6, 720, 1440, 8, 1)
# Extract a small patch: (samples=5, time=6, height_patch=60, width_patch=60, channels=8, drop_last=1)
X_patch = X[:5, :, :60, :60, :, 0]  # Remove last dimension, shape becomes (5, 6, 60, 60, 8)

# Make writable copy and reshape for 3D CNN
X_patch_copy = np.array(X_patch, copy=True)  # Shape: (5, 6, 60, 60, 8)

# For Conv3d input should be (batch, channels, depth, height, width)
# Current shape is (batch, time, height, width, channels)
# Permute to (batch, channels, time, height, width)
X_tensor = torch.from_numpy(X_patch_copy).float().to(DEVICE)
X_tensor = X_tensor.permute(0, 4, 1, 2, 3)  # (5, 8, 6, 60, 60)

# ======================
# PREDICT
# ======================
with torch.no_grad():
    pred = model(X_tensor)

# ======================
# METRICS
# ======================
target = X_tensor
mse = torch.mean((pred - target) ** 2).item()
mae = torch.mean(torch.abs(pred - target)).item()

print(f"[OK] 3D CNN MSE: {mse:.6f}")
print(f"[OK] 3D CNN MAE: {mae:.6f}")

# ======================
# VISUALIZATION
# ======================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Ground Truth")
plt.imshow(target[0,0,3].cpu().numpy(), cmap="hot")
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Prediction")
plt.imshow(pred[0,0,3].cpu().numpy(), cmap="hot")
plt.colorbar()

plt.tight_layout()
plt.savefig(OUT_PLOT)
plt.close()

print(f"[DONE] Saved plot to {OUT_PLOT}")
