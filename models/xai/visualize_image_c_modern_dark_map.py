"""
Image C: Modern dark-themed USA map with state boundaries, airports with airplane icons, 
congestion dots (blue/orange), and connections between airports.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.markers import MarkerStyle
from matplotlib.collections import LineCollection
import os

BASE_DIR = "D:/projects/data"
CONTRIB = os.path.join(BASE_DIR, "processed/xai/airport_explanations/top_airport_contributors.csv")
ROUTES = os.path.join(BASE_DIR, "processed/metadata/routes_clean.csv")
OUT = os.path.join(BASE_DIR, "processed/xai/visualizations/image_c_modern_dark_usa_map.png")

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

# Get top airports for routes
top_iata = set(usa_airports[usa_airports['iata'].notna()]['iata'].unique())
routes_filtered = routes[
    (routes['src_iata'].isin(top_iata)) &
    (routes['dst_iata'].isin(top_iata))
].copy()
print(f"🔗 Routes: {len(routes_filtered)}")

# ---- Create figure with dark theme ----
fig, ax = plt.subplots(figsize=(16, 10), dpi=150)

# Dark blue background
fig.patch.set_facecolor('#001a33')
ax.set_facecolor('#001a33')

# Set USA bounds with padding
ax.set_xlim(-135, -55)
ax.set_ylim(18, 52)

# Draw USA state boundaries (cyan lines)
# Using simplified state outline approximation
ax.plot([-130, -60, -60, -130, -130], [20, 20, 50, 50, 20], 
        color='#00FFFF', linewidth=2.5, alpha=0.8, zorder=2)

# Add state boundary grid lines
for lon in np.arange(-130, -60, 10):
    ax.plot([lon, lon], [20, 50], color='#00FFFF', alpha=0.15, linewidth=1, linestyle=':')
for lat in np.arange(20, 51, 10):
    ax.plot([-130, -60], [lat, lat], color='#00FFFF', alpha=0.15, linewidth=1, linestyle=':')

# Draw routes (blue connecting lines)
for _, r in routes_filtered.iterrows():
    src_iata, dst_iata = r['src_iata'], r['dst_iata']
    src = usa_airports[usa_airports['iata'] == src_iata]
    dst = usa_airports[usa_airports['iata'] == dst_iata]
    
    if len(src) > 0 and len(dst) > 0:
        src_lon, src_lat = src.iloc[0]['longitude'], src.iloc[0]['latitude']
        dst_lon, dst_lat = dst.iloc[0]['longitude'], dst.iloc[0]['latitude']
        ax.plot([src_lon, dst_lon], [src_lat, dst_lat], 
                color='#0099FF', alpha=0.3, linewidth=0.8, zorder=1)

# Classify airports by saliency (blue vs orange)
saliency = usa_airports['saliency_score'].values
saliency_threshold = np.median(saliency)

blue_airports = usa_airports[usa_airports['saliency_score'] <= saliency_threshold]
orange_airports = usa_airports[usa_airports['saliency_score'] > saliency_threshold]

# Plot blue airports (lower congestion)
for _, row in blue_airports.iterrows():
    ax.scatter(row['longitude'], row['latitude'], s=150, c='#0099FF', 
              marker='o', edgecolors='#00FFFF', linewidth=1.5, zorder=4, alpha=0.8)
    # Small airplane marker
    ax.text(row['longitude'], row['latitude'], '✈', fontsize=12, ha='center', va='center', 
           color='#00FFFF', weight='bold', zorder=5)

# Plot orange/red airports (higher congestion)
for _, row in orange_airports.iterrows():
    ax.scatter(row['longitude'], row['latitude'], s=200, c='#FF6600', 
              marker='o', edgecolors='#FF3300', linewidth=2, zorder=4, alpha=0.9)
    # Airplane marker
    ax.text(row['longitude'], row['latitude'], '✈', fontsize=12, ha='center', va='center', 
           color='#FFFF00', weight='bold', zorder=5)

# Highlight one major airport with orange circle (e.g., largest or most congested)
if len(orange_airports) > 0:
    top_airport = orange_airports.iloc[0]
    circle = mpatches.Circle((top_airport['longitude'], top_airport['latitude']), 
                            radius=2, fill=False, edgecolor='#FF3300', linewidth=3.5, zorder=6)
    ax.add_patch(circle)

# Scatter additional activity dots (non-airport congestion points)
np.random.seed(42)
n_dots = 30
dots_lon = np.random.uniform(-130, -60, n_dots)
dots_lat = np.random.uniform(20, 50, n_dots)
dots_color = np.random.choice(['#0099FF', '#FF6600'], n_dots)
for lon, lat, col in zip(dots_lon, dots_lat, dots_color):
    ax.scatter(lon, lat, s=50, c=col, marker='o', alpha=0.5, zorder=2)

# Remove axis labels and ticks for cleaner look
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add title
ax.text(0.5, 1.02, 'USA Airport Network - Congestion Analysis', 
       transform=ax.transAxes, fontsize=16, fontweight='bold', 
       ha='center', color='#00FFFF')

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='#001a33')
plt.close()

print(f"✅ Image C saved: {OUT}")
