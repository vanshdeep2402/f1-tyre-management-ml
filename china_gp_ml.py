import fastf1
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fastf1.Cache.enable_cache('f1_cache')

# 1. Load Suzuka 2026 Qualifying
session = fastf1.get_session(2026, 'Suzuka', 'Q')
session.load()

# 2. Pick Hamilton's fastest lap
fastest_lap = session.laps.pick_driver('HAM').pick_fastest()
telemetry = fastest_lap.get_telemetry()

# 3. Create the Track Map
x = telemetry['X']
y = telemetry['Y']
color = telemetry['Speed'] # This creates the "Heatmap"

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 4. Plotting
fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
ax.axis('off')

# Create a continuous line with a color gradient
from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap=cm.plasma, norm=plt.Normalize(color.min(), color.max()))
lc.set_array(color)
lc.set_linewidth(5)
line = ax.add_collection(lc)

# Add a Colorbar to show speed
cbar = fig.colorbar(line, ax=ax)
cbar.set_label('Speed (km/h)', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

plt.title(f'Lewis Hamilton: Suzuka 2026 Speed Map', color='white', fontsize=20)
plt.savefig('track_heatmap.png', facecolor='black')
plt.show()