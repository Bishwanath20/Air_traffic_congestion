"""
Image B: Time series plot of cancelled flights by time of day for two dates (Oct 20 and Oct 27).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = "D:/projects/data"
OUT = os.path.join(BASE_DIR, "processed/xai/visualizations/image_b_cancelled_flights_timeseries.png")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# Generate synthetic time series data matching the chart pattern
hours = np.arange(13)  # 4am to 4am = 24 hours -> 13 points for every ~2 hours
times = ['4am', '6am', '8am', '10am', '12pm', '2pm', '4pm', '6pm', '8pm', '10pm', '12am', '2am', '4am']

# Oct 27: Shows significant cancellations peaking around 10pm (~140)
oct_27 = np.array([2, 3, 5, 10, 20, 30, 40, 50, 70, 100, 140, 120, 40])

# Oct 20: Shows minimal cancellations (mostly 1-5)
oct_20 = np.array([1, 2, 1, 2, 3, 2, 3, 4, 3, 5, 4, 3, 1])

# Ensure we have proper alignment
x_pos = np.arange(len(times))

# Create figure
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

# Plot Oct 27 (circles, pink/magenta)
ax.plot(x_pos, oct_27, marker='o', markersize=8, color='#E77BC5', linewidth=2.5, label='Oct 27')
ax.scatter(x_pos, oct_27, s=100, c='#E77BC5', marker='o', edgecolors='#D41159', linewidth=1.5, zorder=5)

# Plot Oct 20 (diamonds, green)
ax.plot(x_pos, oct_20, marker='D', markersize=8, color='#2E8B57', linewidth=2.5, label='Oct 20')
ax.scatter(x_pos, oct_20, s=100, c='#2E8B57', marker='D', edgecolors='#1a5c3a', linewidth=1.5, zorder=5)

# Formatting
ax.set_xlabel('Time [EST]', fontsize=12, fontweight='bold')
ax.set_ylabel('Cancelled Flights', fontsize=12, fontweight='bold')
ax.set_title('Cancelled Flights Over Time', fontsize=13, fontweight='bold', pad=15)

ax.set_xticks(x_pos)
ax.set_xticklabels(times, fontsize=10)
ax.set_ylim(0, 200)
ax.set_yticks(np.arange(0, 201, 50))

ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

# White background
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"✅ Image B saved: {OUT}")
