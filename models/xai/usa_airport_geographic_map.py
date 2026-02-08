"""
Image A: USA map showing North America context with USA boundaries clearly marked.
Only plots congestion data for airports within USA bounds.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle
import os

BASE_DIR = "D:/projects/data"
AIRPORTS_META = os.path.join(BASE_DIR, "processed/metadata/airports_clean.csv")
CONTRIB = os.path.join(BASE_DIR, "processed/xai/airport_explanations/top_airport_contributors.csv")
OUT = os.path.join(BASE_DIR, "processed/xai/visualizations/image_a_professional_usa_map.png")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("📂 Loading airports...")
airports_meta = pd.read_csv(AIRPORTS_META)
contrib = pd.read_csv(CONTRIB)

# USA geographic bounds
USA_LON_MIN, USA_LON_MAX = -125, -66
USA_LAT_MIN, USA_LAT_MAX = 24, 50

# Filter to USA only - strict geographic bounds
usa_airports = airports_meta[
    (airports_meta['latitude'] >= USA_LAT_MIN) &
    (airports_meta['latitude'] <= USA_LAT_MAX) &
    (airports_meta['longitude'] >= USA_LON_MIN) &
    (airports_meta['longitude'] <= USA_LON_MAX)
].copy()

# Merge with saliency scores from contribution analysis
usa_airports = usa_airports.merge(
    contrib[['iata', 'saliency_score']].drop_duplicates(),
    left_on='iata', right_on='iata', how='left'
)

# Fill missing saliency with low value
usa_airports['saliency_score'] = usa_airports['saliency_score'].fillna(0.1)

# Sort and get top airports
usa_airports = usa_airports.sort_values('saliency_score', ascending=False).head(400)

print(f"✈️ USA airports in bounds: {len(usa_airports)}")

# ---- Create figure ----
fig, ax = plt.subplots(figsize=(18, 11), dpi=150)

# Set bounds to show North America + USA clearly
ax.set_xlim(-140, -55)
ax.set_ylim(15, 55)

# Light blue ocean background (North America context)
ax.set_facecolor('#ADD8E6')  # Light blue

# Draw Canada outline (simple)
canada = Polygon([(-140, 50), (-50, 50), (-50, 85), (-140, 85)], 
                  alpha=0.15, fc='#D3D3D3', ec='gray', linewidth=1.5, zorder=1)
ax.add_patch(canada)

# Draw Mexico outline (simple)
mexico = Polygon([(-117, 15), (-87, 15), (-87, 33), (-117, 33)], 
                  alpha=0.15, fc='#D3D3D3', ec='gray', linewidth=1.5, zorder=1)
ax.add_patch(mexico)

# Draw USA landmass prominently
usa_land = Rectangle((USA_LON_MIN, USA_LAT_MIN), 
                      USA_LON_MAX - USA_LON_MIN, USA_LAT_MAX - USA_LAT_MIN,
                      alpha=0.25, fc='#F5DEB3', ec='none', zorder=2)
ax.add_patch(usa_land)

# Draw USA border - THICK and PROMINENT
usa_border_x = [USA_LON_MIN, USA_LON_MAX, USA_LON_MAX, USA_LON_MIN, USA_LON_MIN]
usa_border_y = [USA_LAT_MIN, USA_LAT_MIN, USA_LAT_MAX, USA_LAT_MAX, USA_LAT_MIN]
ax.plot(usa_border_x, usa_border_y, color='#FF0000', linewidth=5, alpha=0.8, zorder=10, label='USA Border')

# Draw USA state boundaries (grid-based approximation of state borders)
# Vertical state boundary lines
state_lon_lines = np.arange(USA_LON_MIN, USA_LON_MAX, 2.5)
for lon in state_lon_lines:
    ax.plot([lon, lon], [USA_LAT_MIN, USA_LAT_MAX], 
            color='darkgreen', alpha=0.4, linewidth=1.5, zorder=3)

# Horizontal state boundary lines
state_lat_lines = np.arange(USA_LAT_MIN, USA_LAT_MAX, 2.5)
for lat in state_lat_lines:
    ax.plot([USA_LON_MIN, USA_LON_MAX], [lat, lat], 
            color='darkgreen', alpha=0.4, linewidth=1.5, zorder=3)

# Major state divisions (thicker lines)
major_lons = [-120, -110, -100, -90, -80, -70]
for lon in major_lons:
    if USA_LON_MIN <= lon <= USA_LON_MAX:
        ax.plot([lon, lon], [USA_LAT_MIN, USA_LAT_MAX], 
                color='darkgreen', alpha=0.6, linewidth=2.5, zorder=3)

major_lats = [28, 33, 38, 43, 48]
for lat in major_lats:
    if USA_LAT_MIN <= lat <= USA_LAT_MAX:
        ax.plot([USA_LON_MIN, USA_LON_MAX], [lat, lat], 
                color='darkgreen', alpha=0.6, linewidth=2.5, zorder=3)

# Plot USA airports ONLY (inside USA bounds)
saliency = usa_airports['saliency_score'].values
if len(saliency) > 0:
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
    
    # Green airports (lower congestion)
    green_mask = saliency_norm < 0.5
    if green_mask.sum() > 0:
        ax.scatter(usa_airports[green_mask]['longitude'], 
                  usa_airports[green_mask]['latitude'],
                  s=40 + saliency_norm[green_mask] * 200,
                  c='#00AA00', alpha=0.7, edgecolors='darkgreen', 
                  linewidth=1, zorder=6, label='Lower Congestion')
    
    # Red/orange airports (higher congestion)
    red_mask = ~green_mask
    if red_mask.sum() > 0:
        ax.scatter(usa_airports[red_mask]['longitude'], 
                  usa_airports[red_mask]['latitude'],
                  s=40 + saliency_norm[red_mask] * 250,
                  c='#FF4500', alpha=0.8, edgecolors='darkred', 
                  linewidth=1.5, zorder=7, label='Higher Congestion')

# Add text labels for major regions
ax.text(-125, 52, 'CANADA', fontsize=12, color='gray', alpha=0.6, fontweight='bold', ha='center')
ax.text(-95, 10, 'MEXICO', fontsize=12, color='gray', alpha=0.6, fontweight='bold', ha='center')

# Grid and formatting
ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, color='gray')
ax.set_xlabel('Longitude', fontsize=11, fontweight='bold', color='darkgreen')
ax.set_ylabel('Latitude', fontsize=11, fontweight='bold', color='darkgreen')
ax.set_title('USA Airport Congestion Network Map\n(North America Context)', 
            fontsize=14, fontweight='bold', pad=15, color='darkgreen')

# Legend
ax.legend(loc='lower left', fontsize=10, framealpha=0.95, edgecolor='darkgreen')

# Spine styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('darkgreen')
ax.spines['left'].set_color('darkgreen')
ax.tick_params(labelsize=9, colors='darkgreen')

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print(f"✅ USA map with North America context saved: {OUT}")
print(f"📍 USA Bounds: LON [{USA_LON_MIN}, {USA_LON_MAX}], LAT [{USA_LAT_MIN}, {USA_LAT_MAX}]")
print(f"📊 {len(usa_airports)} USA airports plotted with congestion data")