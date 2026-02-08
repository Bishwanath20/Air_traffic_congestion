import os
import pandas as pd

# Paths
BASE_DIR = r"D:\projects\data"
AIRPORTS_DAT = os.path.join(BASE_DIR, "metadata", "airports.dat")
OUT_DIR = os.path.join(BASE_DIR, "processed", "metadata")
OUT_FILE = os.path.join(OUT_DIR, "airports_clean.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# OpenFlights airport columns (official spec)
COLUMNS = [
    "airport_id",
    "name",
    "city",
    "country",
    "iata",
    "icao",
    "latitude",
    "longitude",
    "altitude",
    "timezone",
    "dst",
    "tz_db",
    "type",
    "source"
]

print("📂 Loading airports.dat ...")

df = pd.read_csv(
    AIRPORTS_DAT,
    header=None,
    names=COLUMNS
)

# Keep only meaningful & reliable rows
df = df[
    (df["latitude"].notna()) &
    (df["longitude"].notna()) &
    (df["type"] == "airport")
]

# Clean invalid IATA codes
df["iata"] = df["iata"].replace("\\N", None)
df["icao"] = df["icao"].replace("\\N", None)

# Select final columns (professional & compact)
df_clean = df[
    [
        "name",
        "city",
        "country",
        "iata",
        "icao",
        "latitude",
        "longitude"
    ]
]

# Save
df_clean.to_csv(OUT_FILE, index=False)

print(f"✅ Airports metadata cleaned")
print(f"📁 Saved to: {OUT_FILE}")
print(f"✈️ Total airports: {len(df_clean)}")