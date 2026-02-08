import os
import pandas as pd
import numpy as np
import folium
from folium import plugins

BASE_DIR = r"D:\projects\data"
CONTRIB_PATH = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "top_airport_contributors.csv")
MAPPED_PATH = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "top_airports_from_regions.csv")
OUT_HTML = os.path.join(BASE_DIR, "processed", "xai", "airport_explanations", "airport_congestion_network_folium.html")
os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)

# Load airports
contrib = pd.read_csv(CONTRIB_PATH)
mapped = pd.read_csv(MAPPED_PATH)

# Merge and deduplicate
contrib['key'] = contrib.apply(lambda r: r['iata'] if pd.notna(r.get('iata')) and r.get('iata')!='' else r.get('icao'), axis=1)
contrib = contrib.dropna(subset=['key']).set_index('key')

mapped['key'] = mapped.apply(lambda r: r['iata'] if pd.notna(r.get('iata')) and r.get('iata')!='' else r.get('icao'), axis=1)

all_airports = pd.concat([
    contrib.reset_index()[['key','airport','latitude','longitude','saliency_score']].rename(columns={'saliency_score':'score','airport':'name'}).set_index('key'),
    mapped.set_index('key')[['airport_name','lat','lon','importance']].rename(columns={'lat':'latitude','lon':'longitude','importance':'score','airport_name':'name'})
], axis=0, sort=False)
all_airports = all_airports[~all_airports.index.duplicated(keep='first')]
all_airports = all_airports.dropna(subset=['latitude','longitude'])

# Select top 25 airports by score
top_airports = all_airports.sort_values('score', ascending=False).head(25)

print(f"[OK] Plotting {len(top_airports)} airports on Folium map...")

# Create map centered on USA (continental)
m = folium.Map(
    location=[39, -98],
    zoom_start=4,
    tiles='OpenStreetMap'
)

# Normalize scores for coloring
scores = top_airports['score'].values
score_min, score_max = scores.min(), scores.max()
score_range = score_max - score_min if score_max > score_min else 1.0

# Add markers
for idx, (key, row) in enumerate(top_airports.iterrows()):
    lat = float(row['latitude'])
    lon = float(row['longitude'])
    score = float(row['score'])
    name = row.get('name', key)
    
    # Normalize score to 0-1 for color
    norm_score = (score - score_min) / score_range
    
    # Color: red (high) to yellow (low)
    color = f'hsl({int(60*(1-norm_score))}, 100%, 50%)'
    
    # Size: 5-20 based on score
    radius = 5 + norm_score * 15
    
    # Popup text
    popup_text = f"""
    <b>{key}</b><br>
    {name}<br>
    Saliency: {score:.4f}<br>
    Lat: {lat:.2f}, Lon: {lon:.2f}
    """
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        popup=folium.Popup(popup_text, max_width=300),
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=2,
        opacity=0.8
    ).add_to(m)
    
    # Add label (airport code) on the map
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(prefix='fa', icon='plane', color='gray', icon_color='white'),
        tooltip=f"{key}: {name}"
    ).add_to(m)

# Add a title/legend
title_html = '''
             <div style="position: fixed; 
                     top: 10px; left: 50px; width: 300px; height: auto; 
                     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                     padding: 10px; border-radius: 5px;">
             <b>Airport Congestion Network (Folium)</b><br>
             Top 25 airports by saliency score<br>
             Size & color indicate saliency intensity<br>
             Click markers for details
             </div>
             '''
m.get_root().html.add_child(folium.Element(title_html))

# Save
m.save(OUT_HTML)
print(f"[DONE] Saved interactive map to: {OUT_HTML}")
print(f"[INFO] Open in browser to explore airports interactively")
