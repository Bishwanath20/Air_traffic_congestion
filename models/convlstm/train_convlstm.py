import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# --------------------
# Paths
# --------------------
DATA_DIR = r"D:\projects\data\processed\ml_tensors"
MODEL_DIR = "./"
os.makedirs(MODEL_DIR, exist_ok=True)

X_PATH = os.path.join(DATA_DIR, "X.dat")
Y_PATH = os.path.join(DATA_DIR, "Y.dat")
META_PATH = os.path.join(DATA_DIR, "shapes.json")

# --------------------
# Load metadata
# --------------------
with open(META_PATH) as f:
    meta = json.load(f)

NUM_SAMPLES = meta["num_samples"]
TIME_STEPS = meta["time_steps"]

# Spatial crop (VERY IMPORTANT)
LAT_CROP = slice(300, 360)
LON_CROP = slice(600, 660)
ALT_IDX = 3  # mid-altitude slice

# --------------------
# Memory-mapped loading
# --------------------
X = np.memmap(
    X_PATH,
    dtype=np.float32,
    mode="r",
    shape=(NUM_SAMPLES,) + tuple(meta["input_shape"])
)

Y = np.memmap(
    Y_PATH,
    dtype=np.float32,
    mode="r",
    shape=(NUM_SAMPLES,) + tuple(meta["output_shape"])
)

# --------------------
# Prepare small training set
# --------------------
N_TRAIN = 200   # small but enough to learn
X_train = X[:N_TRAIN, :, LAT_CROP, LON_CROP, ALT_IDX, 0]
Y_train = Y[:N_TRAIN, LAT_CROP, LON_CROP, ALT_IDX]

X_train = X_train[..., np.newaxis]  # channel dim

print("[OK] Training shape:", X_train.shape)

# --------------------
# Build ConvLSTM model
# --------------------
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvLSTMModel(input_channels=1, hidden_channels=16, kernel_size=3)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"Model on device: {device}")

# --------------------
# Train
# --------------------
model.train()
losses = []

# Make a writable copy and reshape to (batch, time, channels, height, width)
X_train_copy = np.array(X_train, copy=True)
Y_train_copy = np.array(Y_train, copy=True)

# X_train shape: (200, 6, 60, 60, 1) -> (200, 6, 1, 60, 60)
X_train_copy = X_train_copy.transpose(0, 1, 4, 2, 3)  # move channel to position 2

X_train_torch = torch.from_numpy(X_train_copy).float().to(device)
Y_train_torch = torch.from_numpy(Y_train_copy).float().to(device)

for epoch in range(5):
    epoch_loss = 0.0
    
    # Create a simple data loader
    dataset = TensorDataset(X_train_torch, Y_train_torch)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        # output shape: (batch, 1, 60, 60), squeeze to (batch, 60, 60)
        loss = criterion(output.squeeze(1), batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.6f}")

# --------------------
# Save outputs
# --------------------
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "convlstm_model.pt"))

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("ConvLSTM Training Loss")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(MODEL_DIR, "losses.png"), dpi=100, bbox_inches="tight")
plt.close()

print("[DONE] ConvLSTM training completed")
