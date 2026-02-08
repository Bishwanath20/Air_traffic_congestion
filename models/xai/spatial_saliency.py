import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "convlstm", "convlstm_model.pt")
)

TENSOR_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "processed", "ml_tensors", "X_sample.npy")
)

OUT_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "processed", "xai", "spatial_saliency")
)
os.makedirs(OUT_DIR, exist_ok=True)

assert os.path.exists(MODEL_PATH), f"[ERROR] Model not found: {MODEL_PATH}"

# -----------------------------
# ConvLSTM MODEL (same as training)
# -----------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )
    
    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.convlstm = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    
    def forward(self, x):
        """
        x: (batch, time_steps, channels, height, width)
        """
        batch_size, time_steps, _, height, width = x.shape
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        
        for t in range(time_steps):
            h, c = self.convlstm(x[:, t], h, c)
        
        h = self.bn(h)
        out = self.conv_out(h)
        return out

# -----------------------------
# LOAD MODEL
# -----------------------------
model = ConvLSTM(input_channels=1, hidden_channels=16, kernel_size=3).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print("[OK] ConvLSTM model loaded")

# -----------------------------
# LOAD INPUT SAMPLE
# Load from ml_tensors and extract a small sample
# Shape: (samples, time, height, width, channels)
# -----------------------------
import json

META_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "processed", "ml_tensors", "shapes.json")
)
X_DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "processed", "ml_tensors", "X.dat")
)

with open(META_PATH) as f:
    meta = json.load(f)

# Load first sample with crop
X_memmap = np.memmap(X_DATA_PATH, dtype=np.float32, mode="r", shape=(meta["num_samples"],) + tuple(meta["input_shape"]))
LAT_CROP = slice(300, 360)
LON_CROP = slice(600, 660)
ALT_IDX = 3

# Extract sample: (1, 6, 60, 60, 1)
X_sample = X_memmap[0:1, :, LAT_CROP, LON_CROP, ALT_IDX, 0]
X_sample = np.array(X_sample, copy=True)[..., np.newaxis]  # Add channel dim: (1, 6, 60, 60, 1)

# Reshape to (batch, time, channels, height, width)
X_sample = X_sample.transpose(0, 1, 4, 2, 3)  # (1, 6, 1, 60, 60)

X = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True).to(DEVICE)

print("[OK] Input tensor shape:", X.shape)

# -----------------------------
# FORWARD + BACKWARD
# -----------------------------
output = model(X)
loss = output.mean()
loss.backward()

# -----------------------------
# SALIENCY MAP
# -----------------------------
saliency = X.grad.abs().mean(dim=1).squeeze().detach().cpu().numpy()

# -----------------------------
# VISUALIZATION
# -----------------------------
plt.figure(figsize=(10, 6))
plt.imshow(saliency, cmap="hot")
plt.colorbar(label="Saliency Intensity")
plt.title("Spatial Saliency Map (ConvLSTM)")
plt.xlabel("Longitude Grid")
plt.ylabel("Latitude Grid")

out_path = os.path.join(OUT_DIR, "spatial_saliency.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"[DONE] Spatial Saliency XAI completed")
print(f"[OK] Saved to: {out_path}")