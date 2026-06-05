"""
Image A: Professional USA map with state boundaries, airports colored by congestion level,
and connecting routes. Matches the reference image style.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

BASE_DIR = "D:/projects/data"
AIRPORTS_META = os.path.join(BASE_DIR, "processed/metadata/airports_clean.csv")
CONTRIB = os.path.join(BASE_DIR, "processed/xai/airport_explanations/top_airport_contributors.csv")
ROUTES = os.path.join(BASE_DIR, "processed/metadata/routes_clean.csv")
OUT = os.path.join(BASE_DIR, "processed/xai/visualizations/image_a_professional_usa_map.png")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("📂 Loading airports and routes...")
airports_meta = pd.read_csv(AIRPORTS_META)
contrib = pd.read_csv(CONTRIB)
routes = pd.read_csv(ROUTES)

# Get all USA airports from metadata
usa_airports = airports_meta[airports_meta['country'] == 'United States'].copy()
usa_airports = usa_airports.dropna(subset=['latitude', 'longitude'])

# Merge with saliency scores (from contribution analysis)
contrib['code'] = contrib['iata'].fillna(contrib['icao'])
contrib_scores = contrib[['code', 'saliency_score']].drop_duplicates()

# Try to merge on IATA code
usa_airports = usa_airports.merge(
    contrib[['iata', 'saliency_score']].rename(columns={'iata': 'iata'}),
    left_on='iata', right_on='iata', how='left'
)

# Fill NaN saliency scores with a low value (0.1)
usa_airports['saliency_score'] = usa_airports['saliency_score'].fillna(0.1)

all_airports = usa_airports
print(f"✈️ USA airports loaded: {len(all_airports)}")

# Filter to North American bounds (includes Mexico, Canada, USA)
# This matches the reference image extent
na_airports = all_airports[
    (all_airports['latitude'] >= 15) & 
    (all_airports['latitude'] <= 60) &
    (all_airports['longitude'] >= -170) & 
    (all_airports['longitude'] <= -55)
].copy()

# Sample top airports by saliency and keep ALL others for background
top_n = 500  # Show top 500 airports by saliency
na_airports_sorted = na_airports.sort_values('saliency_score', ascending=False)
na_airports_to_plot = na_airports_sorted.head(top_n)

print(f"✈️ North American airports: {len(na_airports)}")
print(f"✈️ Airports to plot (top {top_n}): {len(na_airports_to_plot)}")

# Get top airports for routes
top_iata = set(na_airports_to_plot[na_airports_to_plot['iata'].notna()]['iata'].unique())

if len(top_iata) > 0:
    routes_filtered = routes[
        (routes['src_iata'].isin(top_iata)) &
        (routes['dst_iata'].isin(top_iata))
    ].copy()
else:
    routes_filtered = pd.DataFrame()

print(f"🔗 Routes found: {len(routes_filtered)}")

# Use sampled airports for plotting
na_airports = na_airports_to_plot

# ---- Create figure ----
fig, ax = plt.subplots(figsize=(16, 10), dpi=150)

# Set bounds (continental + Mexico + parts of Canada)
ax.set_xlim(-170, -55)
ax.set_ylim(15, 60)

# Light blue ocean background
ax.set_facecolor('#B0E0E6')  # Powder blue

# Add cream/beige background for land (USA extent)
land_rect = patches.Rectangle((-130, 20), 70, 35, 
                              linewidth=0, facecolor='#F5DEB3', alpha=0.4, zorder=0)
ax.add_patch(land_rect)

# Add state grid (simplified USA state boundaries approximation)
# Major state latitude lines (more visible)
for lat in np.arange(20, 51, 2):
    ax.axhline(lat, color='darkgreen', alpha=0.25, linewidth=0.8, linestyle='-')

# Major state longitude lines (more visible)
for lon in np.arange(-130, -60, 2):
    ax.axvline(lon, color='darkgreen', alpha=0.25, linewidth=0.8, linestyle='-')

# Thicker USA border
ax.plot([-125, -66, -66, -125, -125], [25, 25, 49, 49, 25], 
        color='darkgreen', linewidth=3.5, alpha=0.8, zorder=2)

# Add major state boundaries (approximate state outlines)
# Northern boundary (Canada border approximation)
ax.plot([-125, -95, -80, -66], [49, 49, 49, 49], color='darkgreen', linewidth=2.5, alpha=0.6, zorder=2)

# Mississippi River (major boundary)
ax.plot([-94, -94], [25, 49], color='darkgreen', linewidth=2, alpha=0.5, linestyle='--', zorder=2)

# Major state vertical boundaries
# Rocky Mountains region
for lon in [-125, -120, -115, -110, -105, -100]:
    ax.plot([lon, lon], [25, 49], color='darkgreen', alpha=0.3, linewidth=1, zorder=1)

# Eastern region
for lon in [-95, -90, -85, -80, -75, -70]:
    ax.plot([lon, lon], [25, 49], color='darkgreen', alpha=0.3, linewidth=0.8, zorder=1)

# Major state horizontal boundaries
for lat in [30, 35, 40, 45]:
    ax.plot([-125, -66], [lat, lat], color='darkgreen', alpha=0.25, linewidth=0.8, zorder=1)

# Draw routes (gray connecting lines) - only for top airports to avoid overload
# Skip routes for now to speed up rendering
routes_to_draw = routes_filtered.head(500)  # Limit to 500 routes

print(f"📊 Drawing {len(routes_to_draw)} routes...")
for idx, (_, r) in enumerate(routes_to_draw.iterrows()):
    if idx % 100 == 0:
        print(f"  Processing route {idx}/{len(routes_to_draw)}...")
    src_iata, dst_iata = r['src_iata'], r['dst_iata']
    src = na_airports[na_airports['iata'] == src_iata]
    dst = na_airports[na_airports['iata'] == dst_iata]
    
    if len(src) > 0 and len(dst) > 0:
        src_lon, src_lat = src.iloc[0]['longitude'], src.iloc[0]['latitude']
        dst_lon, dst_lat = dst.iloc[0]['longitude'], dst.iloc[0]['latitude']
        ax.plot([src_lon, dst_lon], [src_lat, dst_lat], 
                color='gray', alpha=0.15, linewidth=0.6, zorder=1)

# Classify airports by saliency (green vs red)
if len(na_airports) > 0:
    saliency = na_airports['saliency_score'].values
    saliency_threshold = np.percentile(saliency, 50)  # Median
    
    # Green airports (lower congestion)
    green_airports = na_airports[na_airports['saliency_score'] <= saliency_threshold]
    # Red airports (higher congestion)
    red_airports = na_airports[na_airports['saliency_score'] > saliency_threshold]
    
    # Plot green airports
    if len(green_airports) > 0:
        sizes_green = 50 + (saliency_threshold - green_airports['saliency_score'].values) * 20
        ax.scatter(green_airports['longitude'], green_airports['latitude'],
                  s=sizes_green, c='green', alpha=0.6, edgecolors='darkgreen', 
                  linewidth=0.8, zorder=4, label='Lower Congestion')
    
    # Plot red/orange airports
    if len(red_airports) > 0:
        sizes_red = 50 + (red_airports['saliency_score'].values - saliency_threshold) * 30
        ax.scatter(red_airports['longitude'], red_airports['latitude'],
                  s=sizes_red, c='#FF4500', alpha=0.7, edgecolors='darkred', 
                  linewidth=1.2, zorder=5, label='Higher Congestion')

# Remove axis ticks for cleaner look
ax.set_xticks([])
ax.set_yticks([])

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Styling
ax.set_xlabel('', fontsize=0)
ax.set_ylabel('', fontsize=0)
ax.set_title('USA Airport Congestion Network Map', 
            fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ Professional USA map saved: {OUT}")
