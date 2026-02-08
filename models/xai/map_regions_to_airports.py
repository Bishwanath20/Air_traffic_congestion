import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import os

BASE_DIR = r"D:\projects\data"

AIRPORTS_PATH = os.path.join(BASE_DIR, "processed", "metadata", "airports_clean.csv")
REGIONS_PATH = os.path.join(
    BASE_DIR, "processed", "xai", "region_explanations", "top_congested_regions.csv"
)

OUT_PATH = os.path.join(
    BASE_DIR, "processed", "xai", "airport_explanations", "top_airports_from_regions.csv"
)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

print("📂 Loading airports...")
airports = pd.read_csv(AIRPORTS_PATH)

# Remove missing coords
airports = airports.dropna(subset=["latitude", "longitude"])

# Build BallTree (Haversine)
coords_rad = np.radians(airports[["latitude", "longitude"]].values)
tree = BallTree(coords_rad, metric="haversine")

print("📂 Loading top congested regions...")
if os.path.exists(REGIONS_PATH):
    regions = pd.read_csv(REGIONS_PATH)
else:
    # Fallback: derive top regions from spatial saliency map
    print("Regions file not found, deriving regions from saliency map...")
    SAL_PATH = os.path.join(BASE_DIR, "processed", "xai", "spatial_saliency", "spatial_saliency.npy")
    PNG_FALLBACK = os.path.splitext(SAL_PATH)[0] + ".png"

    if os.path.exists(SAL_PATH):
        sal = np.load(SAL_PATH)
    elif os.path.exists(PNG_FALLBACK):
        from PIL import Image
        img = Image.open(PNG_FALLBACK).convert("L")
        sal = np.array(img, dtype=np.float32)
        if sal.max() > 0:
            sal = sal / sal.max()
    else:
        raise FileNotFoundError(f"No regions file and no saliency map found ({REGIONS_PATH}, {SAL_PATH})")

    # Build top-k region list from saliency values
    H, W = sal.shape
    # number of regions to extract
    N_TOP = 200
    flat_idx = np.argsort(sal.ravel())[::-1][:N_TOP]
    rows = []

    # Use global grid shape from preprocessing metadata so we map crop indices
    # to global lat/lon correctly. The training preprocessing used a crop
    # (LAT_OFFSET=300, LON_OFFSET=600) producing a 60x60 patch.
    # If your preprocessing changed, update LAT_OFFSET/LON_OFFSET accordingly.
    try:
        import json
        META_PATH = os.path.join(BASE_DIR, "..", "..", "processed", "ml_tensors", "shapes.json")
        with open(META_PATH) as f:
            meta = json.load(f)
        full_h = meta.get("output_shape", [None, None, None])[0] or meta.get("input_shape", [None, None, None, None, None])[1]
        full_w = meta.get("output_shape", [None, None, None])[1] or meta.get("input_shape", [None, None, None, None, None])[2]
    except Exception:
        full_h, full_w = 720, 1440

    # Crop offsets used by the saliency generation
    LAT_OFFSET = 300
    LON_OFFSET = 600

    LAT_MIN, LAT_MAX = -90, 90
    LON_MIN, LON_MAX = -180, 180

    for rid, idx in enumerate(flat_idx):
        i_local = idx // W
        j_local = idx % W
        # map to global grid indices
        i = int(LAT_OFFSET + i_local)
        j = int(LON_OFFSET + j_local)
        # convert global grid to lat/lon (center of cell)
        lat = LAT_MAX - (i / (full_h - 1)) * (LAT_MAX - LAT_MIN)
        lon = LON_MIN + (j / (full_w - 1)) * (LON_MAX - LON_MIN)
        rows.append({
            "region_id": rid,
            "lat": float(lat),
            "lon": float(lon),
            "importance": float(sal[i_local, j_local])
        })

    regions = pd.DataFrame(rows)
    # save fallback regions for future runs
    os.makedirs(os.path.dirname(REGIONS_PATH), exist_ok=True)
    regions.to_csv(REGIONS_PATH, index=False)

# k-NN settings
K_NEIGHBORS = 5
EARTH_KM = 6371.0

# Radii to try (km). The script will try each radius in order and stop when
# we find at least one unique airport (by IATA or ICAO). Adjust as needed.
RADII_KM = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]

def map_regions_with_radius(radius_km):
    matched = []
    for _, row in regions.iterrows():
        lat, lon = row["lat"], row["lon"]
        query_point = np.radians([[lat, lon]])
        dist, idx = tree.query(query_point, k=K_NEIGHBORS)
        dist_km = dist[0] * EARTH_KM
        idxs = idx[0]

        found_any = False
        for dkm, ai in zip(dist_km, idxs):
            if ai is None or np.isnan(ai):
                continue
            if dkm <= radius_km:
                airport = airports.iloc[int(ai)]
                matched.append({
                    "region_id": row["region_id"],
                    "lat": lat,
                    "lon": lon,
                    "icao": airport.get("icao", ""),
                    "iata": airport.get("iata", ""),
                    "airport_name": airport.get("name", ""),
                    "country": airport.get("country", ""),
                    "importance": row.get("importance", 0.0),
                    "distance_km": float(dkm),
                    "radius_km": float(radius_km),
                    "fallback": False,
                })
                found_any = True

        # fallback to nearest neighbor if nothing within radius
        if not found_any and len(idxs) > 0:
            ai = int(idxs[0])
            dkm = float(dist_km[0])
            airport = airports.iloc[ai]
            matched.append({
                "region_id": row["region_id"],
                "lat": lat,
                "lon": lon,
                "icao": airport.get("icao", ""),
                "iata": airport.get("iata", ""),
                "airport_name": airport.get("name", ""),
                "country": airport.get("country", ""),
                "importance": row.get("importance", 0.0),
                "distance_km": dkm,
                "radius_km": float(radius_km),
                "fallback": True,
            })

    return pd.DataFrame(matched)

final_df = None
for r in RADII_KM:
    df_try = map_regions_with_radius(r)
    # count unique IATA and ICAO (ignore empty strings)
    iata_unique = df_try["iata"].replace("", np.nan).dropna().nunique()
    icao_unique = df_try["icao"].replace("", np.nan).dropna().nunique()
    print(f"Tried radius={r} km -> unique iata: {iata_unique}, unique icao: {icao_unique}")
    final_df = df_try
    if iata_unique > 0 or icao_unique > 0:
        print(f"Stopping at radius {r} km (found airports)")
        break

# save final result (last attempt)
final_out = OUT_PATH
final_df.to_csv(final_out, index=False)
print(f"Airport mapping completed: {final_out}")
print(f"Unique IATA codes: {final_df['iata'].replace('', np.nan).dropna().nunique()}")
print(f"Unique ICAO codes: {final_df['icao'].replace('', np.nan).dropna().nunique()}")