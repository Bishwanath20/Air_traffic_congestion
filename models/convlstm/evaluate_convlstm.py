import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# ======================
# CONFIG
# ======================
DEVICE = "cpu"
MODEL_PATH = r"d:\projects\data\models\convlstm\convlstm_model.pt"
DATA_PATH = r"d:\projects\data\processed\ml_tensors\X.dat"
OUT_PLOT = r"d:\projects\data\models\convlstm\convlstm_prediction.png"

# ======================
# MODEL DEFINITION (same as training)
# ======================
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


class ConvLSTMModel(nn.Module):
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

# ======================
# LOAD MODEL
# ======================
model = ConvLSTMModel(input_channels=1, hidden_channels=16, kernel_size=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ======================
# LOAD DATA (small batch)
# ======================
import json
META_PATH = r"d:\projects\data\processed\ml_tensors\shapes.json"
with open(META_PATH) as f:
    meta = json.load(f)

X = np.memmap(DATA_PATH, dtype=np.float32, mode="r", shape=(meta["num_samples"],) + tuple(meta["input_shape"]))
Y = np.memmap(r"d:\projects\data\processed\ml_tensors\Y.dat", dtype=np.float32, mode="r", shape=(meta["num_samples"],) + tuple(meta["output_shape"]))

# Use first 5 samples for evaluation
LAT_CROP = slice(300, 360)
LON_CROP = slice(600, 660)
ALT_IDX = 3

X_eval = X[:5, :, LAT_CROP, LON_CROP, ALT_IDX, 0]
Y_eval = Y[:5, LAT_CROP, LON_CROP, ALT_IDX]

# Reshape to (batch, time, channels, height, width)
X_eval_copy = np.array(X_eval, copy=True)
# X_eval shape is (5, 6, 60, 60) - add channel dimension
X_eval_copy = X_eval_copy[..., np.newaxis]  # (5, 6, 60, 60, 1)
# Transpose to (5, 6, 1, 60, 60)
X_eval_copy = X_eval_copy.transpose(0, 1, 4, 2, 3)

X_tensor = torch.from_numpy(X_eval_copy).float().to(DEVICE)
Y_eval_copy = np.array(Y_eval, copy=True)
Y_tensor = torch.from_numpy(Y_eval_copy).float().to(DEVICE)

# ======================
# PREDICT
# ======================
with torch.no_grad():
    pred = model(X_tensor)

# ======================
# METRICS
# ======================
pred_flat = pred.squeeze(1)
mse = torch.mean((pred_flat - Y_tensor) ** 2).item()
mae = torch.mean(torch.abs(pred_flat - Y_tensor)).item()

print(f"[OK] ConvLSTM MSE: {mse:.6f}")
print(f"[OK] ConvLSTM MAE: {mae:.6f}")

# ======================
# VISUALIZATION
# ======================
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Ground Truth")
plt.imshow(Y_tensor[0].cpu().numpy(), cmap="hot")
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Prediction")
plt.imshow(pred_flat[0].cpu().numpy(), cmap="hot")
plt.colorbar()

plt.tight_layout()
plt.savefig(OUT_PLOT)
plt.close()

print(f"[DONE] Saved plot to {OUT_PLOT}")