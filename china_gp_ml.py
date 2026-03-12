import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  # <--- This is the "Graph Maker"

# 1. Setup Cache
fastf1.Cache.enable_cache('f1_cache') 

# 2. Load 2024 China Race
session = fastf1.get_session(2024, 'Shanghai', 'R')
session.load()

# 3. Pick Lewis Hamilton
driver_laps = session.laps.pick_driver('HAM')
valuable_laps = driver_laps.pick_accurate()

# 4. Prepare Data
X = valuable_laps[['TyreLife']] 
y = valuable_laps['LapTime'].dt.total_seconds()

# 5. Train Model
model = LinearRegression()
model.fit(X, y)

# 6. --- NEW: THE GRAPH CODE ---
plt.figure(figsize=(10, 6))

# Plot the real data points (dots)
plt.scatter(X, y, color='blue', label='Actual Laps')

# Plot the ML prediction line (red line)
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Degradation Trend')

plt.title('Lewis Hamilton Tyre Degradation - Shanghai')
plt.xlabel('Tyre Age (Laps)')
plt.ylabel('Lap Time (Seconds)')
plt.legend()
plt.grid(True)

# This saves the image to your folder so you can upload it to GitHub
plt.savefig('tyre_plot.png')

# This forces the window to pop up on your screen
plt.show() 

print("Prediction for lap 15:", model.predict([[15]])[0])
