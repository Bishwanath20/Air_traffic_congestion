import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
BASE_DIR = r"D:\projects\data"

AIRPORTS_PATH = os.path.join(BASE_DIR, "processed", "metadata", "airports_clean.csv")
ROUTES_PATH = os.path.join(BASE_DIR, "metadata", "routes.dat")
TOP_AIRPORTS_PATH = os.path.join(
    BASE_DIR, "processed", "xai", "airport_explanations", "top_airport_contributors.csv"
)

OUT_DIR = os.path.join(BASE_DIR, "processed", "xai", "route_explanations")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------

print("📂 Loading airport metadata...")
airports = pd.read_csv(AIRPORTS_PATH)

# Build ICAO → IATA map
icao_to_iata = dict(
    zip(
        airports["icao"].astype(str),
        airports["iata"].astype(str)
    )
)

print(f"🔁 ICAO→IATA mappings: {len(icao_to_iata)}")

print("📂 Loading top congested airports...")
top_airports = pd.read_csv(TOP_AIRPORTS_PATH)

# Convert ICAO → IATA
important_iata = set()
for icao in top_airports["icao"]:
    if icao in icao_to_iata:
        iata = icao_to_iata[icao]
        if iata != "\\N":
            important_iata.add(iata)

print(f"✈️ Important airports (IATA): {len(important_iata)}")

# ---------------- LOAD ROUTES ----------------
print("📂 Loading routes.dat...")

routes_cols = [
    "airline", "airline_id",
    "source_airport", "source_id",
    "dest_airport", "dest_id",
    "codeshare", "stops", "equipment"
]

routes = pd.read_csv(
    ROUTES_PATH,
    names=routes_cols,
    usecols=["source_airport", "dest_airport"],
    low_memory=True
)

# Filter relevant routes
routes = routes[
    routes["source_airport"].isin(important_iata) &
    routes["dest_airport"].isin(important_iata)
]

print(f"🔗 Filtered routes: {len(routes)}")

# ---------------- BUILD GRAPH ----------------
print("🧠 Building congestion network...")

G = nx.Graph()

for _, row in routes.iterrows():
    u = row["source_airport"]
    v = row["dest_airport"]

    if G.has_edge(u, v):
        G[u][v]["weight"] += 1
    else:
        G.add_edge(u, v, weight=1)

print(f"📊 Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(14, 14))

if G.number_of_nodes() > 0:
    pos = nx.spring_layout(G, seed=42, k=0.3)
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=120, node_color="red", alpha=0.9)
    nx.draw_networkx_edges(
        G, pos,
        width=[w / max(weights) * 4 for w in weights],
        alpha=0.6
    )

plt.title("Airport-to-Airport Congestion Propagation Network", fontsize=16)
plt.axis("off")

out_path = os.path.join(OUT_DIR, "airport_congestion_network.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"🎉 Network graph saved: {out_path}")