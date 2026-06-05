import os
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
# from matplotlib.transforms import Transformer  # Not needed for this script

BASE_DIR = r"D:\projects\data"
CONTRIB_PATH = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "top_airport_contributors.csv")
MAPPED_PATH = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "top_airports_from_regions.csv")
OUT_PNG = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "airport_congestion_network_geographical_map.png")
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# Load airports
contrib = pd.read_csv(CONTRIB_PATH)
mapped = pd.read_csv(MAPPED_PATH)

contrib['key'] = contrib.apply(lambda r: r['iata'] if pd.notna(r.get('iata')) and r.get('iata')!='' else r.get('icao'), axis=1)
contrib = contrib.dropna(subset=['key']).set_index('key')

mapped['key'] = mapped.apply(lambda r: r['iata'] if pd.notna(r.get('iata')) and r.get('iata')!='' else r.get('icao'), axis=1)

all_nodes = pd.concat([
    contrib.reset_index()[['key','airport','latitude','longitude','saliency_score']].rename(columns={'saliency_score':'score','airport':'name'}).set_index('key'),
    mapped.set_index('key')[['airport_name','lat','lon','importance']].rename(columns={'lat':'latitude','lon':'longitude','importance':'score','airport_name':'name'})
], axis=0, sort=False)
all_nodes = all_nodes[~all_nodes.index.duplicated(keep='first')]
all_nodes = all_nodes.dropna(subset=['latitude','longitude'])

# Select top 25
top_nodes = all_nodes.sort_values('score', ascending=False).head(25)

# Build network (kNN edges)
coords = np.radians(top_nodes[['latitude','longitude']].values.astype(float))
tree = BallTree(coords, metric='haversine')

K_NEIGH = 8
G = nx.Graph()
for key, row in top_nodes.iterrows():
    G.add_node(key, latitude=float(row['latitude']), longitude=float(row['longitude']), score=float(row['score']), name=row.get('name', key))

keys = list(top_nodes.index)
for i, key in enumerate(keys):
    point = coords[i:i+1]
    kq = min(K_NEIGH+1, len(keys))
    dist, idx = tree.query(point, k=kq)
    dist_km = dist[0] * 6371.0
    neighbors = idx[0]
    for d, ni in zip(dist_km, neighbors):
        if ni == i:
            continue
        other = keys[int(ni)]
        weight = (top_nodes.loc[key,'score'] + top_nodes.loc[other,'score']) / 2.0
        G.add_edge(key, other, distance_km=float(d), weight=float(weight))

print(f"[OK] Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# USA extent (Web Mercator projection for contextily)
# Convert lon/lat bounds to Web Mercator (EPSG:3857)
def lonlat_to_webmercator(lon, lat):
    """Convert lon/lat to Web Mercator coordinates"""
    x = lon * 20037508.34 / 180.0
    y = np.log(np.tan((90 + lat) * np.pi / 360)) * 20037508.34 / 180.0
    return x, y

USA_LON_MIN, USA_LON_MAX = -130, -60
USA_LAT_MIN, USA_LAT_MAX = 20, 50

x_min, y_min = lonlat_to_webmercator(USA_LON_MIN, USA_LAT_MIN)
x_max, y_max = lonlat_to_webmercator(USA_LON_MAX, USA_LAT_MAX)

print(f"[OK] Web Mercator bounds: ({x_min:.0f}, {y_min:.0f}) to ({x_max:.0f}, {y_max:.0f})")

# Create figure in Web Mercator projection
fig, ax = plt.subplots(figsize=(18, 12))

# Compute bbox from node coordinates (lon/lat) and add padding
lon_vals = top_nodes['longitude'].astype(float).values
lat_vals = top_nodes['latitude'].astype(float).values
lon_min_nodes, lon_max_nodes = lon_vals.min(), lon_vals.max()
lat_min_nodes, lat_max_nodes = lat_vals.min(), lat_vals.max()
pad_deg = 3.0

# Clamp bbox to continental USA bounds to avoid fetching global tiles
lon_min_pad = max(lon_min_nodes - pad_deg, USA_LON_MIN)
lon_max_pad = min(lon_max_nodes + pad_deg, USA_LON_MAX)
lat_min_pad = max(lat_min_nodes - pad_deg, USA_LAT_MIN)
lat_max_pad = min(lat_max_nodes + pad_deg, USA_LAT_MAX)

# Convert padded bbox to Web Mercator
wx_min, wy_min = lonlat_to_webmercator(lon_min_pad, lat_min_pad)
wx_max, wy_max = lonlat_to_webmercator(lon_max_pad, lat_max_pad)

print(f"[OK] Node bbox (lon/lat): ({lon_min_pad:.2f}, {lat_min_pad:.2f}) - ({lon_max_pad:.2f}, {lat_max_pad:.2f})")

OUT_PNG_V2 = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "airport_congestion_network_geographical_map_v2.png")
os.makedirs(os.path.dirname(OUT_PNG_V2), exist_ok=True)

# Fetch raster tiles for the bbox at a reasonable zoom using contextily.bounds2img
try:
    zoom_level = 6
    img, extent = ctx.bounds2img(wx_min, wy_min, wx_max, wy_max, zoom=zoom_level, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.imshow(img, extent=extent, origin='upper', zorder=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    print(f"[OK] Basemap raster fetched (zoom={zoom_level}) and rendered")
except Exception as e:
    print(f"[WARN] bounds2img basemap failed: {e}")
    # Fallback to add_basemap using axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    try:
        ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik, zoom=5)
        print("[OK] Basemap added (fallback)")
    except Exception as e2:
        print(f"[ERROR] Basemap fallback failed: {e2}")

# Draw edges (converted to Web Mercator)
for u, v in G.edges():
    lon1, lat1 = G.nodes[u]['longitude'], G.nodes[u]['latitude']
    lon2, lat2 = G.nodes[v]['longitude'], G.nodes[v]['latitude']
    x1, y1 = lonlat_to_webmercator(lon1, lat1)
    x2, y2 = lonlat_to_webmercator(lon2, lat2)
    ax.plot([x1, x2], [y1, y2], 'white', alpha=0.5, linewidth=1.5, zorder=2)

# Draw nodes (converted to Web Mercator)
node_x = []
node_y = []
node_scores = []
node_keys = []

for n in G.nodes():
    lon, lat = G.nodes[n]['longitude'], G.nodes[n]['latitude']
    x, y = lonlat_to_webmercator(lon, lat)
    node_x.append(x)
    node_y.append(y)
    node_scores.append(G.nodes[n]['score'])
    node_keys.append(n)

node_x = np.array(node_x)
node_y = np.array(node_y)
node_scores = np.array(node_scores)

# Normalize scores for visualization
score_min, score_max = node_scores.min(), node_scores.max()
score_range = score_max - score_min if score_max > score_min else 1.0
norm_scores = (node_scores - score_min) / score_range

# Node sizes (30-400)
node_sizes = 30 + norm_scores * 370

# Scatter plot with colormap
scatter = ax.scatter(node_x, node_y, s=node_sizes, c=norm_scores, 
                     cmap='RdYlGn_r', alpha=0.85, edgecolors='darkred', linewidth=2, zorder=5)

# Add airport labels
for i, key in enumerate(node_keys):
    ax.text(node_x[i], node_y[i]+50000, key, fontsize=9, fontweight='bold', 
            ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7), zorder=6)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, label='Saliency Score', shrink=0.8, pad=0.02)
cbar.set_label('Saliency Score (Congestion Level)', fontsize=11, fontweight='bold')

# Title and labels
ax.set_title('Airport Congestion Network - USA Geographical Map\n(Geographic Coordinates with OpenStreetMap)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)

# Remove axis spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
save_path = OUT_PNG_V2 if 'OUT_PNG_V2' in globals() else OUT_PNG
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[DONE] Saved geographical map to: {save_path}")
print(f"[INFO] Map shows {len(G.nodes)} airports with proper geographic projection")
print(f"[INFO] OpenStreetMap tiles provide real geographical context (states, cities, roads)")
