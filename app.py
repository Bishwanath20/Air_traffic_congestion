import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / "data" / "processed" / "daily_merged" / "2019-01-14.csv"

st.set_page_config(layout="wide", page_title="✈️ Flight Tracker")
st.title("✈️ Flight Congestion")

if not DATA_FILE.exists():
    st.error(
        "Dataset not found. Please add `data/processed/daily_merged/2019-01-14.csv` "
        "to the repository before deploying this app."
    )
    st.stop()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD DATA — chunked, safe, cached
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_data
def load_data():
    file_path = str(DATA_FILE)
    chunks = []
    for chunk in pd.read_csv(
        file_path,
        usecols=["icao24", "time", "lat", "lon", "velocity"],
        chunksize=500_000
    ):
        chunk = chunk.rename(columns={
            "time": "timestamp",
            "lat": "latitude",
            "lon": "longitude"
        })
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR FILTERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.sidebar.header("🎛️ Filters")

MAX_ROWS = 50000
rows_per_aircraft = len(df_raw) / df_raw["icao24"].nunique()
max_flights = min(int(MAX_ROWS / rows_per_aircraft), df_raw["icao24"].nunique())
max_flights = max(max_flights, 5)
st.sidebar.markdown(f"**Max recommended:** `{max_flights}` (lag-free)")

num_flights = st.sidebar.slider(
    "Number of Flights",
    min_value=5, max_value=max_flights,
    value=min(10, max_flights), step=5
)
top_ids = df_raw["icao24"].value_counts().head(num_flights).index.tolist()

selected_ids = st.sidebar.multiselect(
    "Select Aircraft (icao24)",
    options=top_ids, default=top_ids
)
if not selected_ids:
    st.warning("Please select at least one aircraft.")
    st.stop()

min_vel = float(df_raw["velocity"].min())
max_vel = float(df_raw["velocity"].max())
vel_range = st.sidebar.slider(
    "Velocity Filter (m/s)",
    min_value=min_vel, max_value=max_vel,
    value=(min_vel, max_vel), step=1.0
)
trail_window = st.sidebar.slider(
    "Trail Length (frames)",
    min_value=2, max_value=10, value=5, step=1
)

speed_options  = ["Very Slow", "Slow", "Normal", "Fast", "Very Fast"]
speed_map      = {"Very Slow": 2000, "Slow": 1200, "Normal": 800, "Fast": 400, "Very Fast": 150}
anim_speed     = st.sidebar.select_slider("Animation Speed", options=speed_options, value="Slow")
frame_duration = speed_map[anim_speed]

icon_choice = st.sidebar.radio("Flight Icon", options=["✈️ Airplane", "🛩️ Small Plane"], index=0)
flight_icon = "✈️" if "Airplane" in icon_choice else "🛩️"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PROCESSING & STATUS LOGIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONGESTION COLOR CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONGESTION_LEVELS = {
    "Low":    (
        [[0, "rgba(0,0,0,0)"], [0.4, "rgba(0,200,0,0.3)"],  [1, "rgba(0,255,0,0.95)"]],
        1, "lime"
    ),
    "Medium": (
        [[0, "rgba(0,0,0,0)"], [0.4, "rgba(200,200,0,0.3)"],[1, "rgba(255,255,0,0.95)"]],
        2, "yellow"
    ),
    "High":   (
        [[0, "rgba(0,0,0,0)"], [0.4, "rgba(255,140,0,0.3)"],[1, "rgba(255,165,0,0.95)"]],
        3, "orange"
    ),
    "Severe": (
        [[0, "rgba(0,0,0,0)"], [0.3, "rgba(180,0,0,0.4)"],  [0.7, "rgba(255,80,0,0.8)"], [1, "rgba(255,220,0,1.0)"]],
        5, "red"
    ),
}

LEVEL_ORDER = ["Low", "Medium", "High", "Severe"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANIMATION TRACE BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_traces(current_frame):
    traces = []

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

        _, z_weight, _ = CONGESTION_LEVELS[cong_val]
        level_lats[cong_val].append(float(cur["latitude"].values[0]))
        level_lons[cong_val].append(float(cur["longitude"].values[0]))
        level_weights[cong_val].append(float(z_weight))

    for lvl in LEVEL_ORDER:
        colorscale, _, _ = CONGESTION_LEVELS[lvl]
        lats = level_lats[lvl]
        lons = level_lons[lvl]
        zs   = level_weights[lvl]

        if not lats:
            traces.append(go.Densitymap(
                lat=[0], lon=[0], z=[0],
                radius=20,
                colorscale=colorscale,
                showscale=False,
                opacity=0.0,
                hoverinfo="skip",
                name=f"heat-{lvl}",
                showlegend=False,
            ))
        else:
            traces.append(go.Densitymap(
                lat=lats, lon=lons, z=zs,
                radius=20,
                colorscale=colorscale,
                showscale=False,
                opacity=0.85,
                hoverinfo="skip",
                name=f"heat-{lvl}",
                showlegend=False,
            ))

    for icao in selected_ids:
        data   = aircraft_data.get(icao, pd.DataFrame())
        status = aircraft_status.get(icao, {})
        if data.empty or not status:
            continue

        first_frame = status["first_frame"]
        last_frame  = status["last_frame"]

        if current_frame < first_frame:
            continue

        if current_frame > last_frame:
            traces.append(go.Scattermap(
                lat=[status["last_lat"]], lon=[status["last_lon"]],
                mode="markers",
                marker=dict(size=9, color="gray", opacity=0.35),
                name=f"🛬 {icao}",
                showlegend=False,
            ))
            continue

        cur = data[data["frame"] <= current_frame].iloc[[-1]]
        if cur.empty:
            continue

        cong_val = str(cur["congestion"].values[0])
        _, _, dot_color = CONGESTION_LEVELS.get(cong_val, (None, None, "cyan"))

        traces.append(go.Scattermap(
            lat=cur["latitude"],
            lon=cur["longitude"],
            mode="markers+text",
            marker=dict(size=13, color=dot_color, opacity=1.0),
            text=[f"{flight_icon} {icao}"],
            textfont=dict(size=10, color="white"),
            textposition="top right",
            name=icao,
            customdata=cur[[
                "icao24", "velocity", "status",
                "congestion", "alert", "latitude", "longitude"
            ]].values,
            hovertemplate=(
                "<b>ICAO:</b> %{customdata[0]}<br>"
                "<b>Velocity:</b> %{customdata[1]:.1f} m/s<br>"
                "<b>Status:</b> %{customdata[2]}<br>"
                "<b>Congestion:</b> %{customdata[3]}<br>"
                "<b>Alert:</b> %{customdata[4]}<br>"
                "<b>Lat:</b> %{customdata[5]:.4f}<br>"
                "<b>Lon:</b> %{customdata[6]:.4f}<extra></extra>"
            ),
            showlegend=False,
        ))

    return traces


fig = go.Figure(data=make_traces(all_frame_ids[0]))
fig.frames = [
    go.Frame(data=make_traces(fid), name=str(fid))
    for fid in all_frame_ids
]

fig.update_layout(
    map=dict(
        style="carto-darkmatter",
        zoom=2,
        center={"lat": 40, "lon": -50},
    ),
    height=670,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor="#0a0a0a",
    plot_bgcolor="#0a0a0a",
    updatemenus=[{
        "type": "buttons",
        "showactive": False,
        "bgcolor": "#1c1c1c",
        "bordercolor": "#444",
        "font": {"color": "white", "size": 13},
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": frame_duration, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0}
                }]
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }]
            }
        ],
        "x": 0.01, "y": 0.02,
        "xanchor": "left", "yanchor": "bottom"
    }],
    sliders=[{
        "active": 0,
        "bgcolor": "#1c1c1c",
        "bordercolor": "#555",
        "tickcolor": "#888",
        "font": {"color": "white", "size": 10},
        "currentvalue": {
            "prefix": "Frame: ",
            "font": {"color": "white", "size": 12},
            "xanchor": "center"
        },
        "steps": [
            {
                "args": [[str(fid)], {
                    "frame": {"duration": frame_duration, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0}
                }],
                "label": str(fid),
                "method": "animate"
            }
            for fid in all_frame_ids
        ],
        "x": 0.0, "y": 0.0, "len": 1.0
    }]
)

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
    showarrow=False,
    align="left",
    bgcolor="rgba(0,0,0,0.70)",
    bordercolor="rgba(255,255,255,0.15)",
    borderwidth=1,
    font=dict(color="white", size=11),
)

st.plotly_chart(fig, width='stretch')

combined_df = pd.concat(aircraft_data.values())
alerts = combined_df[combined_df["alert"] == "⚠️ ALERT"]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 Congestion Alerts")
if len(alerts) > 0:
    st.sidebar.error(f"⚠️ High-risk flights: **{alerts['icao24'].nunique()}**")
    st.sidebar.caption(", ".join(alerts["icao24"].unique()))
else:
    st.sidebar.success("✅ No major congestion")

st.sidebar.markdown("### 📊 Congestion Breakdown")
if aircraft_data:
    breakdown = (
        combined_df
        .groupby(combined_df["congestion"].astype(str))["icao24"]
        .nunique()
        .reindex(["Severe", "High", "Medium", "Low"], fill_value=0)
        .reset_index()
    )
    breakdown.columns = ["Level", "Flights"]
    level_emoji = {"Severe": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    breakdown["Level"] = breakdown["Level"].map(lambda x: f"{level_emoji.get(x, '')} {x}")
    st.sidebar.dataframe(breakdown, hide_index=True, width='stretch')
