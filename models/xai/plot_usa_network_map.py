import os
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

BASE_DIR = r"D:\projects\data"
CONTRIB_PATH = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "top_airport_contributors.csv")
MAPPED_PATH = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "top_airports_from_regions.csv")
OUT_PNG = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "airport_congestion_network_usa_map.png")
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

# Filter to USA airports (approx: lat 20-50, lon -130 to -60)
USA_LAT_MIN, USA_LAT_MAX = 20, 50
USA_LON_MIN, USA_LON_MAX = -130, -60

usa_nodes = {n: G.nodes[n] for n in G.nodes 
             if USA_LAT_MIN <= G.nodes[n]['latitude'] <= USA_LAT_MAX 
             and USA_LON_MIN <= G.nodes[n]['longitude'] <= USA_LON_MAX}

if not usa_nodes:
    print("[WARN] No airports in USA bounds, showing all")
    usa_nodes = dict(G.nodes(data=True))

print(f"[OK] USA airports: {len(usa_nodes)}")

# Plot
fig, ax = plt.subplots(figsize=(16, 10))

# USA extent
ax.set_xlim(USA_LON_MIN, USA_LON_MAX)
ax.set_ylim(USA_LAT_MIN, USA_LAT_MAX)

# Light background
ax.set_facecolor('#e6f2ff')

# Draw grid
ax.grid(True, alpha=0.3, linestyle='--', color='gray')

# Draw edges
lons_edges = []
lats_edges = []
for u, v in G.edges():
    if u in usa_nodes and v in usa_nodes:
        lon1, lat1 = G.nodes[u]['longitude'], G.nodes[u]['latitude']
        lon2, lat2 = G.nodes[v]['longitude'], G.nodes[v]['latitude']
        ax.plot([lon1, lon2], [lat1, lat2], 'gray', alpha=0.4, linewidth=1, zorder=1)

# Draw nodes
node_lons = np.array([G.nodes[n]['longitude'] for n in usa_nodes.keys()])
node_lats = np.array([G.nodes[n]['latitude'] for n in usa_nodes.keys()])
scores = np.array([G.nodes[n]['score'] for n in usa_nodes.keys()])

# Normalize scores for size and color
score_min, score_max = scores.min(), scores.max()
score_range = score_max - score_min if score_max > score_min else 1.0
norm_scores = (scores - score_min) / score_range

# Node sizes (20-300)
node_sizes = 20 + norm_scores * 280

# Color map: red (high) to yellow (low)
colors = plt.cm.YlOrRd(norm_scores)

scatter = ax.scatter(node_lons, node_lats, s=node_sizes, c=norm_scores, 
                     cmap='YlOrRd', alpha=0.8, edgecolors='darkred', linewidth=1.5, zorder=5)

# Add airport labels
for n in usa_nodes.keys():
    lon, lat = G.nodes[n]['longitude'], G.nodes[n]['latitude']
    ax.text(lon+0.5, lat+0.5, n, fontsize=8, fontweight='bold', zorder=6)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, label='Saliency Score', pad=0.02)

# Title and labels
ax.set_title('Airport Congestion Network - USA Geographic Map', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Add legend
red_patch = mpatches.Patch(color='#fee5d9', label='Low Saliency')
orange_patch = mpatches.Patch(color='#fcae91', label='Medium Saliency')
red_dark = mpatches.Patch(color='#a50f15', label='High Saliency')
ax.legend(handles=[red_patch, orange_patch, red_dark], loc='lower left', fontsize=10)

# Tight layout
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
plt.close()

print(f"[DONE] Saved USA network map to: {OUT_PNG}")
print(f"[INFO] Plot shows {len(usa_nodes)} airports on USA map")
