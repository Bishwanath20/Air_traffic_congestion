import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# STEP 1 — CONFIGURATION
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "convlstm", "convlstm_model.pt")
)

TENSOR_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "processed", "ml_tensors", "X_sample.npy")
)

OUT_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "processed", "xai", "temporal_saliency")
)
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# STEP 2 — EXACT ConvLSTM MODEL (MATCHES TRAINING)
# ==========================================================
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
        
        outputs = []
        for t in range(time_steps):
            h, c = self.convlstm(x[:, t], h, c)
            outputs.append(h.unsqueeze(1))
        
        h = self.bn(h)
        out = self.conv_out(h)
        return out.unsqueeze(1)  # (batch, 1, 1, height, width)

# ==========================================================
# STEP 3 — LOAD MODEL
# ==========================================================
assert os.path.exists(MODEL_PATH), f"[ERROR] Model not found: {MODEL_PATH}"

model = ConvLSTM(input_channels=1, hidden_channels=16, kernel_size=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("[OK] ConvLSTM model loaded")

# ==========================================================
# STEP 4 — LOAD INPUT TENSOR
# ==========================================================
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

print("[OK] Input tensor shape:", X.shape)

# ==========================================================
# STEP 5 — FORWARD + BACKWARD (SALiency)
# ==========================================================
output = model(X)
loss = output.mean()
loss.backward()

# ==========================================================
# STEP 6 — TEMPORAL SALIENCY COMPUTATION
# ==========================================================
# Gradient importance across time dimension
grads = X.grad.abs()   # (1, T, 1, H, W)

temporal_importance = grads.mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()

# ==========================================================
# STEP 7 — VISUALIZATION
# ==========================================================
plt.figure(figsize=(10, 5))
plt.plot(range(len(temporal_importance)), temporal_importance, marker="o")
plt.xlabel("Hour Index")
plt.ylabel("Importance Score")
plt.title("Temporal Saliency — Hour-wise Congestion Importance")
plt.grid(True)

out_path = os.path.join(OUT_DIR, "temporal_saliency.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()

print("[DONE] Temporal Saliency XAI completed")
print("[OK] Saved to:", out_path)