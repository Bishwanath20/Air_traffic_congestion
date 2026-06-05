import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔑 API KEYS — set these in environment variables or replace with your own
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPENWEATHER_API_KEY   = os.getenv("OPENWEATHER_API_KEY", "your_openweathermap_key_here")
OPENSKY_USERNAME      = os.getenv("OPENSKY_USERNAME", "your_opensky_username_here")
OPENSKY_PASSWORD      = os.getenv("OPENSKY_PASSWORD", "your_opensky_password_here")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY", "your_aviationstack_key_here")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(layout="wide", page_title="✈️ Flight Tracker")
st.title("✈️ Real Time Air Traffic Congestion - Spatiotemporal")

DATA_FILE = Path(__file__).resolve().parent / "data" / "processed" / "daily_merged" / "2019-01-14.csv"
SAMPLE_FILE = Path(__file__).resolve().parent / "data" / "processed" / "daily_merged" / "2019-01-14-sample.csv"

if DATA_FILE.exists():
    selected_file = DATA_FILE
    st.info("Using full dataset `2019-01-14.csv` from the repository.")
elif SAMPLE_FILE.exists():
    selected_file = SAMPLE_FILE
    st.warning(
        "Using sample dataset `2019-01-14-sample.csv` because the full dataset is not in the repo. "
        "For full results, add `data/processed/daily_merged/2019-01-14.csv` to the repository."
    )
else:
    st.error(
        "Dataset not found. Please add `data/processed/daily_merged/2019-01-14.csv` "
        "or the sample file `data/processed/daily_merged/2019-01-14-sample.csv` to the repository."
    )
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE INIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "live_data"           not in st.session_state: st.session_state.live_data           = None
if "live_last_fetched"   not in st.session_state: st.session_state.live_last_fetched   = 0
if "weather_cache"       not in st.session_state: st.session_state.weather_cache       = {}
if "weather_last_fetch"  not in st.session_state: st.session_state.weather_last_fetch  = {}
if "airport_cache"       not in st.session_state: st.session_state.airport_cache       = None
if "airport_last_fetch"  not in st.session_state: st.session_state.airport_last_fetch  = 0

WEATHER_CACHE_TTL = 600   # 10 min
LIVE_ADS_B_TTL    = 300   # 5 min  (rate-limit safe)
AIRPORT_TTL       = 3600  # 1 hr   (AviationStack free tier: 1000/month)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD HISTORICAL DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data
def load_data():
    file_path = str(selected_file)
    chunks = []
    usecols = lambda col: col in {
        "icao24", "time", "timestamp", "lat", "latitude", "lon", "longitude", "velocity"
    }

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=500_000
    ):
        chunk = chunk.rename(columns={"time": "timestamp", "lat": "latitude", "lon": "longitude"})

        if "timestamp" not in chunk.columns:
            raise ValueError("Dataset must contain either 'time' or 'timestamp' column.")
        if "latitude" not in chunk.columns:
            raise ValueError("Dataset must contain either 'lat' or 'latitude' column.")
        if "longitude" not in chunk.columns:
            raise ValueError("Dataset must contain either 'lon' or 'longitude' column.")

        chunk = chunk.dropna()
        chunk = chunk[(chunk["latitude"] != 0) & (chunk["longitude"] != 0)]
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], unit="s")
        chunk = chunk.iloc[::5]
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["icao24", "timestamp"]).reset_index(drop=True)
    return df

df_raw = load_data()
start_time = df_raw["timestamp"].min()
df_raw["frame"] = ((df_raw["timestamp"] - start_time).dt.total_seconds() // 1800).astype(int)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API FETCH FUNCTIONS (with caching + graceful fallback)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_weather(lat, lon, icao):
    """Fetch weather for a lat/lon. Cached per icao for WEATHER_CACHE_TTL seconds."""
    now = time.time()
    if icao in st.session_state.weather_cache:
        if now - st.session_state.weather_last_fetch.get(icao, 0) < WEATHER_CACHE_TTL:
            return st.session_state.weather_cache[icao]
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your_openweathermap_key_here":
        return None
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            d = r.json()
            result = {
                "description": d["weather"][0]["description"].title(),
                "wind_speed":  round(d["wind"]["speed"], 1),
                "wind_deg":    d["wind"].get("deg", 0),
                "visibility":  round(d.get("visibility", 10000) / 1000, 1),
                "temp":        round(d["main"]["temp"], 1),
                "icon":        d["weather"][0]["main"],
            }
            st.session_state.weather_cache[icao]      = result
            st.session_state.weather_last_fetch[icao] = now
            return result
    except Exception:
        pass
    return None


def fetch_live_adsb():
    """Fetch live ADS-B from OpenSky. Cached for LIVE_ADS_B_TTL seconds."""
    now = time.time()
    if (st.session_state.live_data is not None and
            now - st.session_state.live_last_fetched < LIVE_ADS_B_TTL):
        return st.session_state.live_data
    if (not OPENSKY_USERNAME or OPENSKY_USERNAME == "your_opensky_username_here"):
        return None
    try:
        url = "https://opensky-network.org/api/states/all"
        r = requests.get(url, auth=(OPENSKY_USERNAME, OPENSKY_PASSWORD), timeout=10)
        if r.status_code == 200:
            states = r.json().get("states", [])
            rows = []
            for s in states:
                if s[5] is not None and s[6] is not None:
                    rows.append({
                        "icao24":    s[0],
                        "callsign":  (s[1] or "").strip(),
                        "latitude":  s[6],
                        "longitude": s[5],
                        "velocity":  s[9] or 0,
                        "altitude":  s[7] or 0,
                        "on_ground": s[8],
                    })
            df_live = pd.DataFrame(rows)
            df_live = df_live[~df_live["on_ground"]]
            st.session_state.live_data         = df_live
            st.session_state.live_last_fetched = now
            return df_live
    except Exception:
        pass
    return None


def fetch_airport_status():
    """Fetch airport status from AviationStack. Cached for AIRPORT_TTL seconds."""
    now = time.time()
    if (st.session_state.airport_cache is not None and
            now - st.session_state.airport_last_fetch < AIRPORT_TTL):
        return st.session_state.airport_cache
    if (not AVIATIONSTACK_API_KEY or
            AVIATIONSTACK_API_KEY == "your_aviationstack_key_here"):
        return None
    try:
        url = (
            f"http://api.aviationstack.com/v1/flights"
            f"?access_key={AVIATIONSTACK_API_KEY}&flight_status=active&limit=100"
        )
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [])
            airports = {}
            for flight in data:
                for key in ["departure", "arrival"]:
                    ap = flight.get(key, {})
                    iata   = ap.get("iata", "")
                    delay  = ap.get("delay") or 0
                    name   = ap.get("airport", iata)
                    live   = flight.get("live", {}) or {}
                    lat    = live.get("latitude")
                    lon    = live.get("longitude")
                    if iata and lat and lon:
                        if iata not in airports:
                            airports[iata] = {
                                "iata":   iata,
                                "name":   name,
                                "lat":    lat,
                                "lon":    lon,
                                "delay":  delay,
                                "status": "🔴 Closed" if delay > 120
                                          else ("🟡 Delay" if delay > 15 else "🟢 Normal"),
                                "delay_min": delay,
                            }
            result = list(airports.values())
            st.session_state.airport_cache      = result
            st.session_state.airport_last_fetch = now
            return result
    except Exception:
        pass
    return None


def weather_icon_color(icon_str):
    """Map OpenWeatherMap icon category to a marker color."""
    danger = {"Thunderstorm": "red", "Tornado": "red", "Squall": "red"}
    warn   = {"Rain": "orange", "Drizzle": "orange", "Snow": "cyan",
               "Fog": "yellow", "Mist": "yellow", "Haze": "yellow"}
    return danger.get(icon_str, warn.get(icon_str, "deepskyblue"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR — FILTERS + LIVE CONTROLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.sidebar.header("🎛️ Filters")

MAX_ROWS = 50000
rows_per_aircraft = len(df_raw) / df_raw["icao24"].nunique()
max_flights = min(int(MAX_ROWS / rows_per_aircraft), df_raw["icao24"].nunique())
max_flights = max(max_flights, 5)
st.sidebar.markdown(f"**Max recommended:** `{max_flights}` (lag-free)")

num_flights = st.sidebar.slider("Number of Flights", min_value=5, max_value=max_flights,
                                 value=min(10, max_flights), step=5)
top_ids = df_raw["icao24"].value_counts().head(num_flights).index.tolist()

selected_ids = st.sidebar.multiselect("Select Aircraft (icao24)", options=top_ids, default=top_ids)
if not selected_ids:
    st.warning("Please select at least one aircraft.")
    st.stop()

min_vel = float(df_raw["velocity"].min())
max_vel = float(df_raw["velocity"].max())
vel_range = st.sidebar.slider("Velocity Filter (m/s)", min_value=min_vel, max_value=max_vel,
                               value=(min_vel, max_vel), step=1.0)
trail_window = st.sidebar.slider("Trail Length (frames)", min_value=2, max_value=10, value=5, step=1)

speed_options  = ["Very Slow", "Slow", "Normal", "Fast", "Very Fast"]
speed_map_dict = {"Very Slow": 2000, "Slow": 1200, "Normal": 800, "Fast": 400, "Very Fast": 150}
anim_speed     = st.sidebar.select_slider("Animation Speed", options=speed_options, value="Slow")
frame_duration = speed_map_dict[anim_speed]

icon_choice = st.sidebar.radio("Flight Icon", options=["✈️ Airplane", "🛩️ Small Plane"], index=0)
flight_icon = "✈️" if "Airplane" in icon_choice else "🛩️"

st.sidebar.markdown("---")
st.sidebar.header("🌐 Live Data Layers")

show_weather  = st.sidebar.toggle("🌦️ Weather Overlay",     value=True)
show_live     = st.sidebar.toggle("📡 Live ADS-B Positions", value=True)
show_airports = st.sidebar.toggle("🛫 Airport Status Pins",  value=True)

use_live_mode = st.sidebar.radio(
    "Flight Data Source",
    options=["📂 Historical (CSV)", "📡 Live (OpenSky)"],
    index=0
)

if st.sidebar.button("🔄 Refresh Airport Status"):
    st.session_state.airport_cache      = None
    st.session_state.airport_last_fetch = 0
    st.rerun()

if use_live_mode == "📡 Live (OpenSky)" and show_live:
    st.sidebar.caption("⏱ Auto-refreshing every 30s")
    time.sleep(0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FETCH LIVE DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df_live      = fetch_live_adsb()      if show_live     else None
airport_list = fetch_airport_status() if show_airports else None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HISTORICAL DATA PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df_full     = df_raw[df_raw["icao24"].isin(selected_ids)].copy()
df_filtered = df_full[
    (df_full["velocity"] >= vel_range[0]) &
    (df_full["velocity"] <= vel_range[1])
].copy()

df_anim_full     = df_full.groupby(["icao24", "frame"]).tail(1).reset_index(drop=True)
df_anim_filtered = df_filtered.groupby(["icao24", "frame"]).tail(1).reset_index(drop=True)

aircraft_data   = {}
aircraft_status = {}

for icao in selected_ids:
    sub_full = df_anim_full[df_anim_full["icao24"] == icao].sort_values("frame")
    if not sub_full.empty:
        sub_full = sub_full.copy()
        sub_full["status"] = "En Route"
        sub_full["congestion"] = pd.cut(
            sub_full["velocity"],
            bins=[0, 120, 200, 300, 600],
            labels=["Severe", "High", "Medium", "Low"]
        )
        sub_full["alert"] = sub_full["congestion"].apply(
            lambda x: "⚠️ ALERT" if str(x) in ["Severe", "High"] else "Normal"
        )
        aircraft_data[icao] = sub_full

        first_frame    = int(sub_full["frame"].min())
        last_frame     = int(sub_full["frame"].max())
        landing_time   = (start_time + pd.Timedelta(seconds=last_frame  * 1800)).strftime("%H:%M")
        departure_time = (start_time + pd.Timedelta(seconds=first_frame * 1800)).strftime("%H:%M")
        last_pos       = sub_full.iloc[[-1]]

        aircraft_status[icao] = {
            "first_frame":    first_frame,
            "last_frame":     last_frame,
            "landing_time":   landing_time,
            "departure_time": departure_time,
            "last_lat":       float(last_pos["latitude"].values[0]),
            "last_lon":       float(last_pos["longitude"].values[0]),
        }

all_frame_ids = sorted(df_anim_full["frame"].unique())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DISTINCT COLORS PER AIRCRAFT — 40-color palette, cycles if
# more flights are selected than palette length
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AIRCRAFT_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff",
    "#e05252", "#52b0e0", "#e0c452", "#a552e0", "#52e08a",
    "#e07b52", "#5271e0", "#52e0d4", "#c4e052", "#e052b0",
    "#7be052", "#e05279", "#52e052", "#7052e0", "#e0a252",
    "#52a2e0", "#d4e052", "#e052e0", "#52e0a2", "#e06b52",
]

aircraft_color = {
    icao: AIRCRAFT_PALETTE[i % len(AIRCRAFT_PALETTE)]
    for i, icao in enumerate(selected_ids)
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONGESTION CONFIG
# Note: tuples are (colorscale, z_weight) — dot color is now
# handled by aircraft_color above, not by congestion level
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONGESTION_LEVELS = {
    "Low":    ([[0,"rgba(0,0,0,0)"],[0.4,"rgba(0,200,0,0.3)"],[1,"rgba(0,255,0,0.95)"]],    1),
    "Medium": ([[0,"rgba(0,0,0,0)"],[0.4,"rgba(200,200,0,0.3)"],[1,"rgba(255,255,0,0.95)"]], 2),
    "High":   ([[0,"rgba(0,0,0,0)"],[0.4,"rgba(255,140,0,0.3)"],[1,"rgba(255,165,0,0.95)"]], 3),
    "Severe": ([[0,"rgba(0,0,0,0)"],[0.3,"rgba(180,0,0,0.4)"],[0.7,"rgba(255,80,0,0.8)"],[1,"rgba(255,220,0,1.0)"]], 5),
}
LEVEL_ORDER = ["Low", "Medium", "High", "Severe"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEATHER FETCH FOR ACTIVE FLIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
weather_per_icao = {}
if show_weather:
    for icao, status in aircraft_status.items():
        w = fetch_weather(status["last_lat"], status["last_lon"], icao)
        if w:
            weather_per_icao[icao] = w

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANIMATION TRACE BUILDER
# Fixed trace index layout per frame:
#   [0]   Densitymapbox Low
#   [1]   Densitymapbox Medium
#   [2]   Densitymapbox High
#   [3]   Densitymapbox Severe
#   [4…N] Scattermapbox historical flights (per-aircraft color)
# Static traces added after animation (not in frames):
#   • Live ADS-B scatter
#   • Airport pins
#   • Weather markers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_traces(current_frame):
    traces = []

    # ── Congestion density buckets ──
    level_lats    = {lvl: [] for lvl in LEVEL_ORDER}
    level_lons    = {lvl: [] for lvl in LEVEL_ORDER}
    level_weights = {lvl: [] for lvl in LEVEL_ORDER}

    for icao in selected_ids:
        data   = aircraft_data.get(icao, pd.DataFrame())
        status = aircraft_status.get(icao, {})
        if data.empty or not status:
            continue
        if current_frame < status["first_frame"] or current_frame > status["last_frame"]:
            continue
        cur = data[data["frame"] <= current_frame].iloc[[-1]]
        if cur.empty:
            continue
        cong_val = str(cur["congestion"].values[0])
        if cong_val not in CONGESTION_LEVELS:
            continue
        _, z_weight = CONGESTION_LEVELS[cong_val]
        level_lats[cong_val].append(float(cur["latitude"].values[0]))
        level_lons[cong_val].append(float(cur["longitude"].values[0]))
        level_weights[cong_val].append(float(z_weight))

    # ── Densitymapbox layers [0–3] ──
    for lvl in LEVEL_ORDER:
        colorscale, _ = CONGESTION_LEVELS[lvl]
        lats = level_lats[lvl];  lons = level_lons[lvl];  zs = level_weights[lvl]
        if not lats:
            traces.append(go.Densitymapbox(lat=[0], lon=[0], z=[0], radius=20,
                colorscale=colorscale, showscale=False, opacity=0.0,
                hoverinfo="skip", name=f"heat-{lvl}", showlegend=False))
        else:
            traces.append(go.Densitymapbox(lat=lats, lon=lons, z=zs, radius=20,
                colorscale=colorscale, showscale=False, opacity=0.85,
                hoverinfo="skip", name=f"heat-{lvl}", showlegend=False))

    # ── Historical flight dots [4+] — colored by aircraft identity ──
    for icao in selected_ids:
        data   = aircraft_data.get(icao, pd.DataFrame())
        status = aircraft_status.get(icao, {})
        dot_color = aircraft_color.get(icao, "#888888")

        if data.empty or not status:
            traces.append(go.Scattermapbox(
                lat=[], lon=[],
                mode="markers",
                marker=dict(size=13, color=dot_color),
                name=icao, showlegend=False))
            continue

        first_frame = status["first_frame"]
        last_frame  = status["last_frame"]

        if current_frame < first_frame:
            traces.append(go.Scattermapbox(
                lat=[], lon=[],
                mode="markers",
                marker=dict(size=13, color=dot_color),
                name=icao, showlegend=False))
            continue

        if current_frame > last_frame:
            traces.append(go.Scattermapbox(
                lat=[status["last_lat"]], lon=[status["last_lon"]],
                mode="markers", marker=dict(size=9, color="gray", opacity=0.35),
                name=f"🛬 {icao}", showlegend=False))
            continue

        cur = data[data["frame"] <= current_frame].iloc[[-1]]
        if cur.empty:
            traces.append(go.Scattermapbox(
                lat=[], lon=[],
                mode="markers",
                marker=dict(size=13, color=dot_color),
                name=icao, showlegend=False))
            continue

        cong_val = str(cur["congestion"].values[0])

        # Build hover — include weather if available
        w = weather_per_icao.get(icao)
        weather_line = ""
        if w:
            wind_arrow = "↑↗→↘↓↙←↖"[int((w["wind_deg"] + 22.5) / 45) % 8]
            weather_line = (
                f"<br><b>Weather:</b> {w['description']}<br>"
                f"<b>Wind:</b> {wind_arrow} {w['wind_speed']} m/s<br>"
                f"<b>Visibility:</b> {w['visibility']} km<br>"
                f"<b>Temp:</b> {w['temp']}°C"
            )

        traces.append(go.Scattermapbox(
            lat=cur["latitude"], lon=cur["longitude"],
            mode="markers+text",
            marker=dict(size=13, color=dot_color, opacity=1.0),
            text=[f"{flight_icon} {icao}  {cong_val}"],
            textfont=dict(size=10, color="white"),
            textposition="top right",
            name=icao,
            customdata=cur[["icao24","velocity","status","congestion","alert","latitude","longitude"]].values,
            hovertemplate=(
                "<b>ICAO:</b> %{customdata[0]}<br>"
                "<b>Velocity:</b> %{customdata[1]:.1f} m/s<br>"
                "<b>Status:</b> %{customdata[2]}<br>"
                "<b>Congestion:</b> %{customdata[3]}<br>"
                "<b>Alert:</b> %{customdata[4]}<br>"
                "<b>Lat:</b> %{customdata[5]:.4f}<br>"
                "<b>Lon:</b> %{customdata[6]:.4f}"
                + weather_line +
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    return traces


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUILD ANIMATED FIGURE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig = go.Figure(data=make_traces(all_frame_ids[0]))
fig.frames = [go.Frame(data=make_traces(fid), name=str(fid)) for fid in all_frame_ids]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATIC OVERLAY 1 — 📡 LIVE ADS-B DOTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if show_live and df_live is not None and not df_live.empty:
    sample = df_live.sample(min(500, len(df_live)))
    fig.add_trace(go.Scattermapbox(
        lat=sample["latitude"],
        lon=sample["longitude"],
        mode="markers",
        marker=dict(size=5, color="deepskyblue", opacity=0.55),
        name="📡 Live ADS-B",
        hovertemplate=(
            "<b>ICAO:</b> %{customdata[0]}<br>"
            "<b>Callsign:</b> %{customdata[1]}<br>"
            "<b>Velocity:</b> %{customdata[2]:.0f} m/s<br>"
            "<b>Altitude:</b> %{customdata[3]:.0f} m<extra></extra>"
        ),
        customdata=sample[["icao24","callsign","velocity","altitude"]].values,
        showlegend=True,
    ))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATIC OVERLAY 2 — 🛫 AIRPORT STATUS PINS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATUS_COLOR = {"🟢 Normal": "lime", "🟡 Delay": "yellow", "🔴 Closed": "red"}

if show_airports and airport_list:
    for status_label, color in STATUS_COLOR.items():
        grp = [a for a in airport_list if a["status"] == status_label]
        if not grp:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=[a["lat"]  for a in grp],
            lon=[a["lon"]  for a in grp],
            mode="markers",
            marker=dict(size=10, color=color, symbol="airport", opacity=0.9),
            name=f"Airport {status_label}",
            customdata=[[a["iata"], a["name"], a["delay_min"]] for a in grp],
            hovertemplate=(
                "<b>Airport:</b> %{customdata[1]} (%{customdata[0]})<br>"
                "<b>Status:</b> " + status_label + "<br>"
                "<b>Delay:</b> %{customdata[2]} min<extra></extra>"
            ),
            showlegend=True,
        ))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATIC OVERLAY 3 — 🌦️ WEATHER MARKERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if show_weather and weather_per_icao:
    wx_lats, wx_lons, wx_colors, wx_text, wx_custom = [], [], [], [], []
    for icao, w in weather_per_icao.items():
        s = aircraft_status.get(icao)
        if not s:
            continue
        wx_lats.append(s["last_lat"])
        wx_lons.append(s["last_lon"])
        wx_colors.append(weather_icon_color(w["icon"]))
        wx_text.append(w["description"][:12])
        wx_custom.append([
            w["description"], w["wind_speed"], w["visibility"], w["temp"], icao
        ])

    if wx_lats:
        fig.add_trace(go.Scattermapbox(
            lat=wx_lats, lon=wx_lons,
            mode="markers+text",
            marker=dict(size=9, color=wx_colors, opacity=0.75),
            text=wx_text,
            textfont=dict(size=8, color="white"),
            textposition="bottom right",
            name="🌦️ Weather",
            customdata=wx_custom,
            hovertemplate=(
                "<b>Weather near %{customdata[4]}</b><br>"
                "<b>Condition:</b> %{customdata[0]}<br>"
                "<b>Wind:</b> %{customdata[1]} m/s<br>"
                "<b>Visibility:</b> %{customdata[2]} km<br>"
                "<b>Temp:</b> %{customdata[3]}°C<extra></extra>"
            ),
            showlegend=True,
        ))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig.update_layout(
    mapbox_style="carto-darkmatter",
    mapbox_zoom=2,
    mapbox_center={"lat": 40, "lon": -50},
    height=670,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor="#0a0a0a",
    plot_bgcolor="#0a0a0a",
    legend=dict(
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        font=dict(color="white", size=11),
        x=0.01, y=0.15,
    ),
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "bgcolor": "#1c1c1c",
        "bordercolor": "#444",
        "font": {"color": "white", "size": 13},
        "buttons": [
            {"label": "▶ Play",  "method": "animate",
             "args": [None, {"frame": {"duration": frame_duration, "redraw": True},
                             "fromcurrent": True, "transition": {"duration": 0}}]},
            {"label": "⏸ Pause", "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate", "transition": {"duration": 0}}]},
        ],
        "x": 0.01, "y": 0.02, "xanchor": "left", "yanchor": "bottom"
    }],
    sliders=[{
        "active": 0,
        "bgcolor": "#1c1c1c", "bordercolor": "#555", "tickcolor": "#888",
        "font": {"color": "white", "size": 10},
        "currentvalue": {"prefix": "Frame: ", "font": {"color": "white", "size": 12}, "xanchor": "center"},
        "steps": [
            {"args": [[str(fid)], {"frame": {"duration": frame_duration, "redraw": True},
                                   "mode": "immediate", "transition": {"duration": 0}}],
             "label": str(fid), "method": "animate"}
            for fid in all_frame_ids
        ],
        "x": 0.0, "y": 0.0, "len": 1.0
    }]
)

# ── Congestion legend (top-right annotation) ──
fig.add_annotation(
    x=0.99, y=0.99, xref="paper", yref="paper",
    xanchor="right", yanchor="top",
    text=(
        "<b>Congestion Level</b><br>"
        "🔴  Severe  (&lt;120 m/s)<br>"
        "🟠  High      (120–200 m/s)<br>"
        "🟡  Medium  (200–300 m/s)<br>"
        "🟢  Low       (300–600 m/s)"
    ),
    showarrow=False, align="left",
    bgcolor="rgba(0,0,0,0.70)", bordercolor="rgba(255,255,255,0.15)",
    borderwidth=1, font=dict(color="white", size=11),
)

st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR PANELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Congestion alerts ──
combined_df = pd.concat(aircraft_data.values())
alerts = combined_df[combined_df["alert"] == "⚠️ ALERT"]
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Congestion Alerts")
if len(alerts) > 0:
    st.sidebar.error(f"⚠️ High-risk flights: **{alerts['icao24'].nunique()}**")
    st.sidebar.caption(", ".join(alerts["icao24"].unique()))
else:
    st.sidebar.success("✅ No major congestion")

breakdown = (
    combined_df.groupby(combined_df["congestion"].astype(str))["icao24"]
    .nunique()
    .reindex(["Severe","High","Medium","Low"], fill_value=0)
    .reset_index()
)
breakdown.columns = ["Level","Flights"]
level_emoji = {"Severe":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}
breakdown["Level"] = breakdown["Level"].map(lambda x: f"{level_emoji.get(x,'')} {x}")
st.sidebar.markdown("### 📊 Congestion Breakdown")
st.sidebar.dataframe(breakdown, hide_index=True, use_container_width=True)

# ── Live ADS-B panel ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Live ADS-B Feed")
if not show_live:
    st.sidebar.info("Toggle 'Live ADS-B Positions' above to enable.")
elif OPENSKY_USERNAME == "your_opensky_username_here":
    st.sidebar.warning("⚠️ Add your OpenSky credentials in the API KEYS block.")
elif df_live is not None:
    last_fetched_ago = int(time.time() - st.session_state.live_last_fetched)
    st.sidebar.success(f"✅ Live data loaded — {len(df_live):,} airborne flights")
    st.sidebar.caption(f"Last updated: {last_fetched_ago}s ago  |  Next refresh in ~{max(0, LIVE_ADS_B_TTL - last_fetched_ago)}s")
    mode_label = "📡 Live" if use_live_mode == "📡 Live (OpenSky)" else "📂 Historical"
    st.sidebar.caption(f"Display mode: **{mode_label}**")
else:
    st.sidebar.error("❌ Could not fetch live data. Check credentials or network.")

# ── Weather alerts ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌦️ Weather Alerts")
if not show_weather:
    st.sidebar.info("Toggle 'Weather Overlay' above to enable.")
elif OPENWEATHER_API_KEY == "your_openweathermap_key_here":
    st.sidebar.warning("⚠️ Add your OpenWeatherMap API key in the API KEYS block.")
elif not weather_per_icao:
    st.sidebar.success("✅ No severe weather near active flights.")
else:
    storm_flights = [
        icao for icao, w in weather_per_icao.items()
        if w["icon"] in ["Thunderstorm", "Tornado", "Squall"]
    ]
    warn_flights = [
        icao for icao, w in weather_per_icao.items()
        if w["icon"] in ["Rain", "Drizzle", "Snow", "Fog"]
    ]
    if storm_flights:
        st.sidebar.error(f"⛈️ Storm near: {', '.join(storm_flights)}")
    if warn_flights:
        st.sidebar.warning(f"🌧️ Rain/Fog near: {', '.join(warn_flights)}")
    if not storm_flights and not warn_flights:
        st.sidebar.success("✅ Clear conditions near all active flights.")

    wx_rows = []
    for icao, w in weather_per_icao.items():
        wx_rows.append({
            "ICAO":       icao,
            "Condition":  w["description"],
            "Wind (m/s)": w["wind_speed"],
            "Vis (km)":   w["visibility"],
            "Temp (°C)":  w["temp"],
        })
    if wx_rows:
        st.sidebar.dataframe(pd.DataFrame(wx_rows), hide_index=True, use_container_width=True)

# ── Airport status panel ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛫 Airport Status")
if not show_airports:
    st.sidebar.info("Toggle 'Airport Status Pins' above to enable.")
elif AVIATIONSTACK_API_KEY == "your_aviationstack_key_here":
    st.sidebar.warning("⚠️ Add your AviationStack API key in the API KEYS block.")
elif airport_list:
    delayed  = [a for a in airport_list if "Delay"  in a["status"]]
    closed   = [a for a in airport_list if "Closed" in a["status"]]
    normal   = [a for a in airport_list if "Normal" in a["status"]]

    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("🟢 Normal", len(normal))
    col2.metric("🟡 Delay",  len(delayed))
    col3.metric("🔴 Closed", len(closed))

    if closed:
        st.sidebar.error("🔴 Closed airports: " + ", ".join(a["iata"] for a in closed))
    if delayed:
        st.sidebar.warning("🟡 Delayed airports:")
        delay_df = pd.DataFrame([{
            "IATA": a["iata"], "Airport": a["name"], "Delay (min)": a["delay_min"]
        } for a in delayed]).sort_values("Delay (min)", ascending=False)
        st.sidebar.dataframe(delay_df, hide_index=True, use_container_width=True)

    last_ap = int(time.time() - st.session_state.airport_last_fetch)
    st.sidebar.caption(f"Last refreshed: {last_ap//60}m {last_ap%60}s ago  |  Click '🔄 Refresh' to update.")
else:
    st.sidebar.error("❌ Could not fetch airport data. Check API key or network.")