import fastf1
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

fastf1.Cache.enable_cache('f1_cache')

# 1. Load Session
session = fastf1.get_session(2026, 'Shanghai', 'FP1')
session.load()

def get_compound_data(compound_name, color):
    # Filter laps by driver AND compound
    laps = session.laps.pick_driver('HAM').pick_compound(compound_name).pick_accurate()
    
    if len(laps) < 3: return None # Skip if not enough data
    
    X = laps[['TyreLife']]
    y = laps['LapTime'].dt.total_seconds()
    model = LinearRegression().fit(X, y)
    return X, y, model, color

# 2. Extract Soft and Medium data
soft_data = get_compound_data('SOFT', 'red')
med_data = get_compound_data('MEDIUM', 'yellow')

# 3. Plotting the Battle
plt.figure(figsize=(10, 6))

for data in [soft_data, med_data]:
    if data:
        X, y, model, col = data
        plt.scatter(X, y, color=col, alpha=0.5, label=f'Actual {col} Laps')
        plt.plot(X, model.predict(X), color=col, linewidth=3, label=f'{col} Deg Trend')

plt.title('2026 Shanghai: Soft vs Medium Degradation (Hamilton)')
plt.xlabel('Tyre Age (Laps)')
plt.ylabel('Lap Time (Seconds)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('compound_comparison.png')
plt.show()
