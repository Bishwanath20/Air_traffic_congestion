import os
import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import networkx as nx
import matplotlib.pyplot as plt

BASE_DIR = r"D:\projects\data"
CONTRIB_PATH = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "top_airport_contributors.csv")
MAPPED_PATH = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "top_airports_from_regions.csv")
os.makedirs(os.path.join(BASE_DIR, "processed", "xai", "airport_explanations"), exist_ok=True)

def build_network(mode='default', top_n=20, out_png=None):
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

    # Modes:
    # default: prefer IATA-bearing airports when possible (existing behavior)
    # include_icao: select top_n purely by score regardless of IATA presence
    # label_icao: same as default but ensure labels show ICAO when IATA missing
    # overlay_routes: produce dense synthetic route edges (no external route file available)

    # identify IATA keys
    all_nodes['is_iata'] = all_nodes.index.map(lambda k: isinstance(k,str) and len(k)==3)

    if mode == 'include_icao':
        nodes = all_nodes.sort_values('score', ascending=False).head(top_n)
    else:
        preferred = all_nodes[all_nodes['is_iata']].sort_values('score', ascending=False)
        if len(preferred) >= 15:
            nodes = preferred.head(top_n)
        else:
            nodes = all_nodes.sort_values('score', ascending=False).head(top_n)

    # prepare coordinates
    coords = np.radians(nodes[['latitude','longitude']].values.astype(float))
    tree = BallTree(coords, metric='haversine')

    # build graph
    K_NEIGH = 8 if mode != 'overlay_routes' else min(len(nodes)-1, 25)
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

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()} (mode={mode})")

    # plot
    if out_png is None:
        out_png = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", f"airport_congestion_network_{mode}.png")

    plt.figure(figsize=(14,10))
    pos = {n: (G.nodes[n]['longitude'], G.nodes[n]['latitude']) for n in G.nodes}

    scores = np.array([G.nodes[n]['score'] for n in G.nodes])
    scores_ptp = np.ptp(scores) if np.ptp(scores) != 0 else 1.0
    node_sizes = 200 + (scores - scores.min())/scores_ptp * 1200

    edge_weights = np.array([d['weight'] for _,_,d in G.edges(data=True)]) if G.number_of_edges()>0 else np.array([1.0])
    edge_ptp = np.ptp(edge_weights) if np.ptp(edge_weights) != 0 else 1.0
    edge_widths = 0.5 + (edge_weights - edge_weights.min())/edge_ptp * 3.5

    nx.draw_networkx_edges(G, pos, alpha=0.6, width=edge_widths, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='red', alpha=0.9)

    if mode == 'label_icao':
        labels = {n: (n if not (isinstance(n,str) and len(n)==3) else n) for n in G.nodes}
    else:
        labels = {n: n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    plt.title(f'Airport Congestion Network (mode={mode})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved network image to: {out_png}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['default','include_icao','label_icao','overlay_routes'], default='default')
    p.add_argument('--top', type=int, default=20)
    args = p.parse_args()
    build_network(mode=args.mode, top_n=args.top)
