import os
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import networkx as nx
import matplotlib.pyplot as plt

BASE_DIR = r"D:\projects\data"
CONTRIB_PATH = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "top_airport_contributors.csv")
MAPPED_PATH = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "top_airports_from_regions.csv")
OUT_PNG = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "airport_congestion_network_map.png")
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# Build node set similar to build_airport_network
contrib = pd.read_csv(CONTRIB_PATH)
mapped = pd.read_csv(MAPPED_PATH)
contrib['key'] = contrib.apply(lambda r: r['iata'] if pd.notna(r.get('iata')) and r.get('iata')!='' else r.get('icao'), axis=1)
contrib = contrib.dropna(subset=['key']).set_index('key')
mapped['key'] = mapped.apply(lambda r: r['iata'] if pd.notna(r.get('iata')) and r.get('iata')!='' else r.get('icao'), axis=1)

all_nodes = pd.concat([
    contrib.reset_index()[['key','latitude','longitude','saliency_score']].rename(columns={'saliency_score':'score'}).set_index('key'),
    mapped.set_index('key')[['lat','lon','importance']].rename(columns={'lat':'latitude','lon':'longitude','importance':'score'})
], axis=0, sort=False)
all_nodes = all_nodes[~all_nodes.index.duplicated(keep='first')]
all_nodes = all_nodes.dropna(subset=['latitude','longitude'])
all_nodes['is_iata'] = all_nodes.index.map(lambda k: isinstance(k,str) and len(k)==3)

# include airport name where available
if 'airport' in contrib.columns:
    contrib_names = contrib.reset_index()[['key','airport']].set_index('key')
    all_nodes = all_nodes.join(contrib_names, how='left')
if 'airport_name' in mapped.columns:
    mapped_names = mapped.set_index('key')[['airport_name']]
    all_nodes = all_nodes.combine_first(mapped_names.rename(columns={'airport_name':'airport'}))

# choose nodes (prefer IATA when available)
TOP_N = 25
preferred = all_nodes[all_nodes['is_iata']].sort_values('score', ascending=False)
if len(preferred) >= 15:
    nodes = preferred.head(TOP_N)
else:
    nodes = all_nodes.sort_values('score', ascending=False).head(TOP_N)

# create edges via geographic kNN
coords = np.radians(nodes[['latitude','longitude']].values.astype(float))
tree = BallTree(coords, metric='haversine')
K_NEIGH = 8
G = nx.Graph()
for key, row in nodes.iterrows():
    G.add_node(key, latitude=float(row['latitude']), longitude=float(row['longitude']), score=float(row['score']))

keys = list(nodes.index)
for i, key in enumerate(keys):
    point = coords[i:i+1]
    kq = min(K_NEIGH+1, len(nodes))
    dist, idx = tree.query(point, k=kq)
    dist_km = dist[0] * 6371.0
    neighbors = idx[0]
    for d, ni in zip(dist_km, neighbors):
        if ni == i:
            continue
        other = keys[int(ni)]
        weight = (nodes.loc[key,'score'] + nodes.loc[other,'score']) / 2.0
        G.add_edge(key, other, distance_km=float(d), weight=float(weight))

# Try to plot with cartopy if available
use_cartopy = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    use_cartopy = False

plt.figure(figsize=(14,10))
if use_cartopy:
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # Center/zoom to USA (continental)
    ax.set_extent([-130, -60, 20, 55], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    # positions
    lons = np.array([G.nodes[n]['longitude'] for n in G.nodes])
    lats = np.array([G.nodes[n]['latitude'] for n in G.nodes])

    # draw edges as great-circle lines
    for u, v, d in G.edges(data=True):
        lon1, lat1 = G.nodes[u]['longitude'], G.nodes[u]['latitude']
        lon2, lat2 = G.nodes[v]['longitude'], G.nodes[v]['latitude']
        ax.plot([lon1, lon2], [lat1, lat2], transform=ccrs.Geodetic(), color='gray', alpha=0.6, linewidth=1)

    # node sizes
    scores = np.array([G.nodes[n]['score'] for n in G.nodes])
    scores_ptp = np.ptp(scores) if np.ptp(scores) != 0 else 1.0
    node_sizes = 50 + (scores - scores.min())/scores_ptp * 400

    ax.scatter(lons, lats, s=node_sizes, color='red', transform=ccrs.PlateCarree(), zorder=5)
    for n in G.nodes:
        lab = G.nodes[n].get('airport', n)
        ax.text(G.nodes[n]['longitude']+0.2, G.nodes[n]['latitude']+0.2, lab, transform=ccrs.PlateCarree(), fontsize=7)

else:
    # fallback: simple lon/lat scatter and annotate full names
    lons = np.array([G.nodes[n]['longitude'] for n in G.nodes])
    lats = np.array([G.nodes[n]['latitude'] for n in G.nodes])
    for u, v, d in G.edges(data=True):
        lon1, lat1 = G.nodes[u]['longitude'], G.nodes[u]['latitude']
        lon2, lat2 = G.nodes[v]['longitude'], G.nodes[v]['latitude']
        plt.plot([lon1, lon2], [lat1, lat2], color='gray', alpha=0.6, linewidth=1)

    scores = np.array([G.nodes[n]['score'] for n in G.nodes])
    scores_ptp = np.ptp(scores) if np.ptp(scores) != 0 else 1.0
    node_sizes = 50 + (scores - scores.min())/scores_ptp * 400
    plt.scatter(lons, lats, s=node_sizes, color='red')
    for n in G.nodes:
        lab = G.nodes[n].get('airport', n)
        plt.text(G.nodes[n]['longitude']+0.2, G.nodes[n]['latitude']+0.2, lab, fontsize=7)
    plt.xlim(-130, -60)
    plt.ylim(20, 55)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

plt.title('Airport Congestion Network on Geographic Map (USA view)')
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print(f"Saved map to: {OUT_PNG} (cartopy used: {use_cartopy})")
