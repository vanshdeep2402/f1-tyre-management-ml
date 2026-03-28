import fastf1
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import numpy as np

# 1. Setup Cache
fastf1.Cache.enable_cache('f1_cache')

# 2. Load Japanese GP 2026 Qualifying
session = fastf1.get_session(2026, 'Suzuka', 'Q')
session.load()

# 3. Get Hamilton's Fastest Lap
lap = session.laps.pick_driver('HAM').pick_fastest()
tel = lap.get_telemetry()

# 4. Prepare data for the heatmap
x = tel['X']
y = tel['Y']
color = tel['Speed']

# Create segments for the line
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 5. Build the Plot
fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
ax.axis('off')

# Create the colored line collection
norm = plt.Normalize(color.min(), color.max())
lc = LineCollection(segments, cmap=cm.magma, norm=norm, linestyle='-', linewidth=5)
lc.set_array(color)
line = ax.add_collection(lc)

# Add a high-tech colorbar
cbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Speed (km/h)', color='white', fontsize=12)
cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

plt.title(f'Lewis Hamilton | Suzuka 2026 | Speed Heatmap', color='white', fontsize=20, pad=20)

# 6. Save and Show
plt.savefig('suzuka_heatmap.png', facecolor='black')
print("✅ Heatmap saved as 'suzuka_heatmap.png'")
plt.show()
