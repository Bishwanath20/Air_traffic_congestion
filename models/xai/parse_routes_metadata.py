import pandas as pd
import os

BASE_DIR = "D:/projects/data"
ROUTES_PATH = os.path.join(BASE_DIR, "metadata", "routes.dat")
OUT_PATH = os.path.join(BASE_DIR, "processed", "metadata", "routes_clean.csv")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

cols = [
    "airline", "airline_id",
    "src_iata", "src_id",
    "dst_iata", "dst_id",
    "codeshare", "stops", "equipment"
]

print("📂 Loading routes.dat ...")
routes = pd.read_csv(ROUTES_PATH, header=None, names=cols)

# Keep only valid IATA routes
routes = routes[
    (routes["src_iata"] != "\\N") &
    (routes["dst_iata"] != "\\N")
]

routes = routes[["src_iata", "dst_iata"]]

routes.to_csv(OUT_PATH, index=False)
print(f"✅ Routes cleaned and saved: {OUT_PATH}")
print(f"🔗 Total routes: {len(routes)}")