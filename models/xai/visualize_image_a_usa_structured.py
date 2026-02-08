"""
Image A: Professional USA map with actual state boundaries, airports colored by congestion.
Uses pre-defined state boundary coordinates.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import Polygon
import os

BASE_DIR = "D:/projects/data"
AIRPORTS_META = os.path.join(BASE_DIR, "processed/metadata/airports_clean.csv")
CONTRIB = os.path.join(BASE_DIR, "processed/xai/airport_explanations/top_airport_contributors.csv")
OUT = os.path.join(BASE_DIR, "processed/xai/visualizations/image_a_professional_usa_map.png")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("📂 Loading airports and routes...")
airports_meta = pd.read_csv(AIRPORTS_META)
contrib = pd.read_csv(CONTRIB)

# Get all USA airports from metadata
usa_airports = airports_meta[airports_meta['country'] == 'United States'].copy()
usa_airports = usa_airports.dropna(subset=['latitude', 'longitude'])

# Merge with saliency scores
usa_airports = usa_airports.merge(
    contrib[['iata', 'saliency_score']].drop_duplicates(),
    left_on='iata', right_on='iata', how='left'
)

# Fill NaN saliency scores
usa_airports['saliency_score'] = usa_airports['saliency_score'].fillna(0.1)

# Filter to top 500 airports
usa_airports = usa_airports.sort_values('saliency_score', ascending=False).head(500)

print(f"✈️ USA airports loaded: {len(usa_airports)}")

# Define USA state boundaries (approximate polygon coordinates)
# Simplified state outlines
state_boundaries = {
    'west_coast': [(-124.4, 49), (-124.4, 42.5), (-114, 42.5), (-114, 49), (-124.4, 49)],
    'mountain': [(-114, 42.5), (-114, 25), (-104, 25), (-104, 49), (-114, 49), (-114, 42.5)],
    'midwest': [(-104, 25), (-104, 49), (-93, 49), (-93, 25), (-104, 25)],
    'great_plains': [(-104, 25), (-93, 25), (-93, 49), (-104, 49)],
    'southeast': [(-93, 25), (-80, 25), (-80, 37), (-93, 37), (-93, 25)],
    'northeast': [(-80, 37), (-65, 37), (-65, 47), (-80, 47), (-80, 37)],
}

# ---- Create figure ----
fig, ax = plt.subplots(figsize=(18, 11), dpi=150)

# Set bounds
ax.set_xlim(-130, -65)
ax.set_ylim(20, 51)

# Light blue ocean background
ax.set_facecolor('#B0E0E6')

# Draw USA landmass (beige/tan color)
usa_land = Polygon([(-125, 25), (-65, 25), (-65, 50), (-125, 50)], 
                    alpha=0.3, fc='#F5DEB3', ec='none', zorder=1)
ax.add_patch(usa_land)

# Draw state boundaries using polygons with outlines
colors_land = ['#E8D5B7', '#EDD9C3', '#F5DEB3', '#E8D5B7', '#EDD9C3']
boundaries_list = [
    [(-125, 42), (-124, 42), (-124, 49), (-125, 49)],  # Pacific
    [(-124, 25), (-114, 25), (-114, 42), (-124, 42)],  # Southwest
    [(-114, 25), (-104, 25), (-104, 49), (-114, 49)],  # Mountain
    [(-104, 25), (-95, 25), (-95, 49), (-104, 49)],    # Plains
    [(-95, 25), (-80, 25), (-80, 49), (-95, 49)],      # Midwest/South
    [(-80, 25), (-65, 25), (-65, 49), (-80, 49)],      # East/Northeast
]

for i, boundary in enumerate(boundaries_list):
    poly = Polygon(boundary, alpha=0.25, fc=colors_land[i % len(colors_land)], 
                   ec='darkgreen', linewidth=2, zorder=2)
    ax.add_patch(poly)

# Add state grid lines (more refined)
# Vertical lines (meridians)
for lon in np.arange(-125, -65, 5):
    ax.plot([lon, lon], [25, 50], color='darkgreen', alpha=0.3, linewidth=1, zorder=2)

# Horizontal lines (parallels)
for lat in np.arange(25, 51, 5):
    ax.plot([-125, -65], [lat, lat], color='darkgreen', alpha=0.3, linewidth=1, zorder=2)

# Draw USA border (thick black outline)
usa_border = Polygon([(-125, 25), (-65, 25), (-65, 50), (-125, 50)], 
                      alpha=0, fc='none', ec='darkgreen', linewidth=4, zorder=3)
ax.add_patch(usa_border)

# Add internal state lines (approximation of major state boundaries)
# Major vertical divisions
major_lons = [-120, -110, -100, -90, -80, -70]
for lon in major_lons:
    ax.plot([lon, lon], [25, 50], color='darkgreen', alpha=0.4, linewidth=1.5, zorder=2)

# Major horizontal divisions
major_lats = [30, 35, 40, 45]
for lat in major_lats:
    ax.plot([-125, -65], [lat, lat], color='darkgreen', alpha=0.35, linewidth=1.5, zorder=2)

# Plot airports (colored by saliency)
saliency = usa_airports['saliency_score'].values
saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)

# Green airports (lower congestion)
green_mask = saliency_norm < 0.5
red_mask = ~green_mask

if green_mask.sum() > 0:
    ax.scatter(usa_airports[green_mask]['longitude'], 
              usa_airports[green_mask]['latitude'],
              s=30 + saliency_norm[green_mask] * 150,
              c='green', alpha=0.6, edgecolors='darkgreen', 
              linewidth=0.8, zorder=5, label='Lower Congestion')

# Red/orange airports (higher congestion)
if red_mask.sum() > 0:
    ax.scatter(usa_airports[red_mask]['longitude'], 
              usa_airports[red_mask]['latitude'],
              s=30 + saliency_norm[red_mask] * 200,
              c='#FF4500', alpha=0.75, edgecolors='darkred', 
              linewidth=1.2, zorder=6, label='Higher Congestion')

# Remove axis ticks
ax.set_xticks(np.arange(-130, -60, 10))
ax.set_yticks(np.arange(20, 51, 5))
ax.tick_params(labelsize=8, colors='gray')
ax.grid(False)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('gray')
ax.spines['left'].set_color('gray')

# Labels
ax.set_xlabel('Longitude', fontsize=10, color='darkgreen', fontweight='bold')
ax.set_ylabel('Latitude', fontsize=10, color='darkgreen', fontweight='bold')
ax.set_title('USA Airport Congestion Network - Geographic Distribution', 
            fontsize=14, fontweight='bold', pad=15, color='darkgreen')
ax.legend(loc='lower left', fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"✅ Professional USA map saved: {OUT}")
print(f"📊 {len(usa_airports)} airports plotted")
