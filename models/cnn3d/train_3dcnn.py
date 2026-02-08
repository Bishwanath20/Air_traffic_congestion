import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# =====================================================
# PATH SETUP (robust, absolute)
# =====================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

TENSOR_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "spatial_tensors")
MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "models", "cnn3d")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# HYPERPARAMETERS
# =====================================================
TIME_WINDOW = 6          # temporal depth
PATCH_SIZE = 60          # spatial patch
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-3

# =====================================================
# DATASET (PATCH-BASED, STREAMING)
# =====================================================
class SpatialPatchDataset(Dataset):
    def __init__(self, tensor_dir):
        self.files = [
            os.path.join(tensor_dir, f)
            for f in os.listdir(tensor_dir)
            if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor = np.load(self.files[idx], mmap_mode="r")
        # tensor shape: (24, H, W, C)

        T, H, W, C = tensor.shape

        # random temporal window
        t0 = np.random.randint(0, T - TIME_WINDOW)

        # random spatial patch
        i = np.random.randint(0, H - PATCH_SIZE)
        j = np.random.randint(0, W - PATCH_SIZE)

        patch = tensor[
            t0:t0 + TIME_WINDOW,
            i:i + PATCH_SIZE,
            j:j + PATCH_SIZE,
            :
        ]  # (T, H, W, C)

        # convert to (C, T, H, W)
        patch = np.transpose(patch, (3, 0, 1, 2)).astype(np.float32)

        x = torch.from_numpy(patch)
        y = x.clone()  # autoencoder-style target

        return x, y


dataset = SpatialPatchDataset(TENSOR_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =====================================================
# 3D CNN MODEL (NO POOLING → SHAPE SAFE)
# =====================================================
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


model = CNN3D().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# =====================================================
# TRAINING LOOP
# =====================================================
print(f"Model on device: {DEVICE}")

for epoch in range(EPOCHS):
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")

# =====================================================
# SAVE MODEL
# =====================================================
model_path = os.path.join(MODEL_DIR, "cnn3d_model.pth")
torch.save(model.state_dict(), model_path)

print("🎉 3D CNN PATCH-BASED TRAINING COMPLETED")
print(f"💾 Model saved to: {model_path}")