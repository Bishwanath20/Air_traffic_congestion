"""
Image A: USA map with contextily real map tiles showing airport congestion.
Displays North America context with USA boundaries highlighted.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import os

BASE_DIR = "D:/projects/data"
AIRPORTS_META = os.path.join(BASE_DIR, "processed/metadata/airports_clean.csv")
CONTRIB = os.path.join(BASE_DIR, "processed/xai/airport_explanations/top_airport_contributors.csv")
OUT = os.path.join(BASE_DIR, "processed/xai/visualizations/image_a_professional_usa_map.png")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("📂 Loading airports...")
airports_meta = pd.read_csv(AIRPORTS_META)
contrib = pd.read_csv(CONTRIB)

# USA geographic bounds (lat/lon)
USA_LON_MIN, USA_LON_MAX = -125, -66
USA_LAT_MIN, USA_LAT_MAX = 24, 50

# Filter to USA only
usa_airports = airports_meta[
    (airports_meta['latitude'] >= USA_LAT_MIN) &
    (airports_meta['latitude'] <= USA_LAT_MAX) &
    (airports_meta['longitude'] >= USA_LON_MIN) &
    (airports_meta['longitude'] <= USA_LON_MAX)
].copy()

# Merge with saliency scores
usa_airports = usa_airports.merge(
    contrib[['iata', 'saliency_score']].drop_duplicates(),
    left_on='iata', right_on='iata', how='left'
)

# Fill missing saliency
usa_airports['saliency_score'] = usa_airports['saliency_score'].fillna(0.1)

# Get top airports
usa_airports = usa_airports.sort_values('saliency_score', ascending=False).head(300)

print(f"✈️ USA airports in bounds: {len(usa_airports)}")

# Function to convert lat/lon to Web Mercator (EPSG:3857)
def lonlat_to_webmercator(lon, lat):
    """Convert lon/lat to Web Mercator coordinates"""
    x = lon * 20037508.34 / 180.0
    y = np.log(np.tan((90 + lat) * np.pi / 360)) * 20037508.34 / 180.0
    return x, y

# Convert USA bounds to Web Mercator
wm_min = lonlat_to_webmercator(USA_LON_MIN, USA_LAT_MIN)
wm_max = lonlat_to_webmercator(USA_LON_MAX, USA_LAT_MAX)

west, south = wm_min
east, north = wm_max

print(f"📍 Fetching map tiles for Web Mercator bounds: ({west:.0f}, {south:.0f}) to ({east:.0f}, {north:.0f})")

# ---- Create figure ----
fig, ax = plt.subplots(figsize=(18, 11), dpi=150)

# Fetch and display OpenStreetMap tiles
try:
    print("🗺️ Adding OpenStreetMap tiles...")
    ctx.add_basemap(
        ax,
        crs='EPSG:3857',
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom=5,
        attribution_size=8
    )
    print("✅ Tiles loaded successfully")
except Exception as e:
    print(f"⚠️ Basemap loading issue: {e}")

# Set axis limits (Web Mercator)
ax.set_xlim(west, east)
ax.set_ylim(south, north)

# Convert airports to Web Mercator for plotting
airport_coords_wm = [lonlat_to_webmercator(row['longitude'], row['latitude']) 
                     for _, row in usa_airports.iterrows()]
airport_x = [c[0] for c in airport_coords_wm]
airport_y = [c[1] for c in airport_coords_wm]

# Normalize saliency scores for visualization
saliency = usa_airports['saliency_score'].values
if len(saliency) > 0:
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
    
    # Green airports (lower congestion)
    green_mask = saliency_norm < 0.5
    if green_mask.sum() > 0:
        ax.scatter(
            np.array(airport_x)[green_mask], 
            np.array(airport_y)[green_mask],
            s=50 + saliency_norm[green_mask] * 200,
            c='#00AA00', alpha=0.7, edgecolors='darkgreen', 
            linewidth=1.5, zorder=6, label='Lower Congestion'
        )
    
    # Red/orange airports (higher congestion)
    red_mask = ~green_mask
    if red_mask.sum() > 0:
        ax.scatter(
            np.array(airport_x)[red_mask], 
            np.array(airport_y)[red_mask],
            s=50 + saliency_norm[red_mask] * 250,
            c='#FF4500', alpha=0.85, edgecolors='darkred', 
            linewidth=2, zorder=7, label='Higher Congestion'
        )

# USA border rectangle (Web Mercator)
usa_border_rect = plt.Rectangle(
    (west, south), east - west, north - south,
    fill=False, edgecolor='#FF0000', linewidth=5, alpha=0.8, zorder=10
)
ax.add_patch(usa_border_rect)

# Formatting
ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
ax.set_title('USA Airport Congestion Network Map\n(OpenStreetMap Geographic Tiles)', 
            fontsize=14, fontweight='bold', pad=15)

ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ USA map with contextily tiles saved: {OUT}")
print(f"📊 {len(usa_airports)} USA airports plotted with real map background")
