import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = "D:/projects/data"

AIRPORTS = os.path.join(BASE_DIR, "processed/metadata/airports_clean.csv")
ROUTES = os.path.join(BASE_DIR, "processed/metadata/routes_clean.csv")
CONTRIB = os.path.join(BASE_DIR, "processed/xai/airport_explanations/top_airport_contributors.csv")
OUT = os.path.join(BASE_DIR, "processed/xai/route_explanations/us_airport_routes.png")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

print("📂 Loading airport metadata...")
airports = pd.read_csv(AIRPORTS)

print("📂 Loading routes...")
routes = pd.read_csv(ROUTES)

print("📂 Loading top airports from saliency...")
contrib = pd.read_csv(CONTRIB)

# 🇺🇸 Filter US airports only
airports_us = airports[airports["country"] == "United States"]

# Get top airports from saliency (already ranked)
top_from_contrib = contrib.head(25)
top_from_contrib['code'] = top_from_contrib['iata'].fillna(top_from_contrib['icao'])

# Try to merge with airports_clean to get routes
top_iata = set(top_from_contrib[top_from_contrib['iata'].notna()]['iata'].unique())

if len(top_iata) > 0:
    # Keep routes between top congested airports
    routes_filtered = routes[
        (routes["src_iata"].isin(top_iata)) &
        (routes["dst_iata"].isin(top_iata))
    ]
else:
    routes_filtered = pd.DataFrame()  # Empty if no IATA codes

print(f"✈️ Top airports to plot: {len(top_from_contrib)}")
print(f"🔗 Routes found between top airports: {len(routes_filtered)}")

# Use saliency airport data (has lat/lon directly)
plot_airports = top_from_contrib.dropna(subset=['latitude', 'longitude'])

# ---- Plot ----
plt.figure(figsize=(14, 9))

# Plot routes
for _, r in routes_filtered.iterrows():
    src_iata, dst_iata = r["src_iata"], r["dst_iata"]
    src = top_from_contrib[top_from_contrib['iata'] == src_iata]
    dst = top_from_contrib[top_from_contrib['iata'] == dst_iata]
    
    if len(src) > 0 and len(dst) > 0:
        src_lat, src_lon = src.iloc[0]['latitude'], src.iloc[0]['longitude']
        dst_lat, dst_lon = dst.iloc[0]['latitude'], dst.iloc[0]['longitude']
        plt.plot(
            [src_lon, dst_lon],
            [src_lat, dst_lat],
            color="gray",
            alpha=0.25,
            linewidth=0.8
        )

# Plot airports (colored by saliency)
scatter = plt.scatter(
    plot_airports["longitude"],
    plot_airports["latitude"],
    s=plot_airports['saliency_score'] * 100 + 50,
    c=plot_airports['saliency_score'],
    cmap='RdYlGn_r',
    alpha=0.8,
    edgecolors='darkred',
    linewidth=2,
    zorder=5
)

# Labels
for _, row in plot_airports.iterrows():
    code = row['iata'] if pd.notna(row['iata']) else row['icao']
    plt.text(
        row["longitude"] + 1,
        row["latitude"] + 1,
        code,
        fontsize=8,
        fontweight='bold'
    )

plt.colorbar(scatter, label='Saliency Score')
plt.title("Airport-to-Airport Congestion Propagation Map (US)", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(alpha=0.2)

plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()

print(f"🎉 Geographic congestion map saved: {OUT}")