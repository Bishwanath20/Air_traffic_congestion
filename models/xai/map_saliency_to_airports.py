import os
import numpy as np
import pandas as pd

# ---------------- PATHS ----------------
BASE_DIR = r"D:\projects\data"

SAL_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "xai",
    "spatial_saliency",
    "spatial_saliency.npy"
)

AIRPORTS_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "metadata",
    "airports_clean.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "processed",
    "xai",
    "airport_explanations"
)

OUT_FILE = os.path.join(OUT_DIR, "top_airport_contributors.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
print("Loading spatial saliency map...")
if os.path.exists(SAL_PATH):
    saliency = np.load(SAL_PATH)
else:
    # try PNG fallback
    png_path = os.path.splitext(SAL_PATH)[0] + ".png"
    if os.path.exists(png_path):
        try:
            from PIL import Image

            img = Image.open(png_path).convert("L")
            saliency = np.array(img, dtype=np.float32)
            # normalize to 0..1
            if saliency.max() > 0:
                saliency = saliency / saliency.max()
            print(f"Loaded saliency from PNG fallback: {png_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load saliency PNG fallback: {e}")
    else:
        raise FileNotFoundError(
            f"Saliency file not found: {SAL_PATH} (also tried {png_path})"
        )

print("Loading airport metadata...")
if not os.path.exists(AIRPORTS_PATH):
    raise FileNotFoundError(f"Airports metadata not found: {AIRPORTS_PATH}")
airports = pd.read_csv(AIRPORTS_PATH)

# ---------------- GRID DEFINITION ----------------
# Must match tensor grid used earlier
LAT_MIN, LAT_MAX = -90, 90
LON_MIN, LON_MAX = -180, 180

H, W = saliency.shape

def latlon_to_grid(lat, lon):
    """Convert lat/lon to saliency grid index"""
    i = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * (H - 1))
    j = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * (W - 1))
    return i, j

# ---------------- MAP AIRPORTS ----------------
rows = []

print("Mapping airports to saliency regions...")

for _, row in airports.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    i, j = latlon_to_grid(lat, lon)

    if 0 <= i < H and 0 <= j < W:
        score = saliency[i, j]
        rows.append({
            "airport": row["name"],
            "city": row["city"],
            "country": row["country"],
            "iata": row["iata"],
            "icao": row["icao"],
            "latitude": lat,
            "longitude": lon,
            "saliency_score": score
        })

df_scores = pd.DataFrame(rows)

# ---------------- TOP CONTRIBUTORS ----------------
df_top = (
    df_scores
    .sort_values("saliency_score", ascending=False)
    .head(30)
)

df_top.to_csv(OUT_FILE, index=False)

print("Airport explanations generated")
print(f"Saved to: {OUT_FILE}")
print("Top contributing airports:")
print(df_top[["airport", "iata", "country", "saliency_score"]].head(10))