"""
Image A: USA map with state boundaries, airports as red/orange circles, and routes as connecting lines.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import os

BASE_DIR = "D:/projects/data"
CONTRIB = os.path.join(BASE_DIR, "processed/xai/airport_explanations/top_airport_contributors.csv")
ROUTES = os.path.join(BASE_DIR, "processed/metadata/routes_clean.csv")
OUT = os.path.join(BASE_DIR, "processed/xai/visualizations/image_a_usa_routes_map.png")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Load data
print("📂 Loading airports and routes...")
contrib = pd.read_csv(CONTRIB)
routes = pd.read_csv(ROUTES)

# Filter to USA airports by country and reasonable geographic bounds
usa_airports = contrib[
    (contrib['country'] == 'United States') |
    ((contrib['latitude'] >= 20) & 
     (contrib['latitude'] <= 50) &
     (contrib['longitude'] >= -130) & 
     (contrib['longitude'] <= -60))
].copy()

usa_airports = usa_airports.dropna(subset=['latitude', 'longitude'])
print(f"✈️ USA airports: {len(usa_airports)}")

# Get top airports by saliency for routes
top_iata = set(usa_airports[usa_airports['iata'].notna()]['iata'].unique())
routes_filtered = routes[
    (routes['src_iata'].isin(top_iata)) &
    (routes['dst_iata'].isin(top_iata))
].copy()
print(f"🔗 Routes between top airports: {len(routes_filtered)}")

# ---- Create figure ----
fig, ax = plt.subplots(figsize=(16, 10), dpi=150)

# Set USA bounds
ax.set_xlim(-130, -60)
ax.set_ylim(20, 50)
ax.set_facecolor('#ADD8E6')  # Light blue background

# Add grid
ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)

# Draw USA state boundaries (simplified using basic rectangles for major states)
# Light green for land area
ax.add_patch(plt.Rectangle((-130, 20), 70, 30, alpha=0.2, fc='lightgreen', ec='darkgreen', linewidth=1.5))

# Add state boundary lines (simplified grid)
for lon in np.arange(-130, -60, 5):
    ax.axvline(lon, color='darkgreen', alpha=0.15, linewidth=0.5, linestyle=':')
for lat in np.arange(20, 51, 5):
    ax.axhline(lat, color='darkgreen', alpha=0.15, linewidth=0.5, linestyle=':')

# Draw routes (gray lines)
for _, r in routes_filtered.iterrows():
    src_iata, dst_iata = r['src_iata'], r['dst_iata']
    src = usa_airports[usa_airports['iata'] == src_iata]
    dst = usa_airports[usa_airports['iata'] == dst_iata]
    
    if len(src) > 0 and len(dst) > 0:
        src_lon, src_lat = src.iloc[0]['longitude'], src.iloc[0]['latitude']
        dst_lon, dst_lat = dst.iloc[0]['longitude'], dst.iloc[0]['latitude']
        ax.plot([src_lon, dst_lon], [src_lat, dst_lat], 'gray', alpha=0.2, linewidth=0.5, zorder=1)

# Plot airports (red circles, sized by saliency)
saliency = usa_airports['saliency_score'].values
saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
sizes = 50 + saliency_norm * 250

scatter = ax.scatter(
    usa_airports['longitude'], 
    usa_airports['latitude'],
    s=sizes,
    c='red',
    alpha=0.6,
    edgecolors='darkred',
    linewidth=1.5,
    zorder=5,
    label='Airports'
)

# Labels for top airports
for idx, row in usa_airports.head(15).iterrows():
    code = row['iata'] if pd.notna(row['iata']) else row['icao']
    ax.text(row['longitude'], row['latitude'] + 1, code, fontsize=7, ha='center', 
            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6), zorder=6)

ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
ax.set_title('USA Airport Congestion Network - Geographic Distribution', fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ Image A saved: {OUT}")
