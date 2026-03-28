import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

# 1. Setup Cache
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

# 2. Set Session (Suzuka 2026 Qualifying)
YEAR = 2026
LOCATION = 'Suzuka'
SESSION = 'Q' 

print(f"🏎️ Loading {LOCATION} {YEAR} {SESSION} Data...")
session = fastf1.get_session(YEAR, LOCATION, SESSION)
session.load()

# 3. Define the Grid Leaders + LH44
drivers = ['ANT', 'RUS', 'PIA', 'LEC', 'HAM']
colors = ['cyan', 'silver', 'orange', 'red', 'darkred']

plt.figure(figsize=(12, 7))

print("📊 Analyzing tyre degradation trends...")
for drv, col in zip(drivers, colors):
    # Filter for the driver's accurate laps
    laps = session.laps.pick_driver(drv).pick_accurate()
    
    if not laps.empty:
        # We use TyreLife to see how the car behaves as the rubber wears out
        X = laps[['TyreLife']]
        y = laps['LapTime'].dt.total_seconds()
        
        # Train our AI Model
        model = LinearRegression().fit(X, y)
        deg_rate = model.coef_[0]
        
        # Plot the Trend Line
        plt.plot(X, model.predict(X), color=col, linewidth=3, label=f'{drv} ({deg_rate:.3f}s/lap)')
        plt.scatter(X, y, color=col, alpha=0.3)

# 4. Styling the "Winner Prediction" Graph
plt.title(f'Winner Prediction: {LOCATION} GP 2026 (Tyre Deg Analysis)', fontsize=15)
plt.xlabel('Tyre Age (Laps)')
plt.ylabel('Lap Time (Seconds)')
plt.legend(title="Driver (Degradation Rate)")
plt.grid(True, linestyle='--', alpha=0.6)

# Save for your GitHub
plt.savefig('japan_winner_prediction.png')
print("✅ Analysis Complete! Check 'japan_winner_prediction.png'")
plt.show()