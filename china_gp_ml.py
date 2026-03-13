import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('f1_cache')

def get_driver_data(year, session_type, driver, color):
    print(f"Loading {year} {session_type} data...")
    session = fastf1.get_session(year, 'Shanghai', session_type)
    session.load()
    laps = session.laps.pick_driver(driver).pick_accurate()
    
    # Filter for 'push' laps (within 107% of best time)
    fastest = laps['LapTime'].min()
    laps = laps[laps['LapTime'] < fastest * 1.07]
    
    X = laps[['TyreLife']]
    y = laps['LapTime'].dt.total_seconds()
    
    model = LinearRegression().fit(X, y)
    return X, y, model, color

# Fetch both datasets
X24, y24, model24, col24 = get_driver_data(2024, 'R', 'HAM', 'silver')
X26, y26, model26, col26 = get_driver_data(2026, 'FP1', 'HAM', 'red')

# --- Plotting ---
plt.figure(figsize=(12, 7))

# 2024 Mercedes Data
plt.scatter(X24, y24, color=col24, alpha=0.3, label='2024 Mercedes Laps')
plt.plot(X24, model24.predict(X24), color=col24, linewidth=3, label='2024 Trend (Mercedes)')

# 2026 Ferrari Data
plt.scatter(X26, y26, color=col26, alpha=0.5, label='2026 Ferrari Laps')
plt.plot(X26, model26.predict(X26), color=col26, linewidth=3, label='2026 Trend (Ferrari)')

plt.title('Hamilton Strategy Evolution: Mercedes (2024) vs. Ferrari (2026)')
plt.xlabel('Tyre Age (Laps)')
plt.ylabel('Lap Time (Seconds)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('comparison_plot.png')
plt.show()

print(f"2024 Deg Rate: {model24.coef_[0]:.3f}s/lap")
print(f"2026 Deg Rate: {model26.coef_[0]:.3f}s/lap")