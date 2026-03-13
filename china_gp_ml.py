import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Setup Cache - This keeps the data on your laptop
fastf1.Cache.enable_cache('f1_cache') 

print("Fetching LIVE 2026 Chinese GP FP1 Data...")

try:
    # 2. Load the CURRENT 2026 Session (Line 10/11)
    session = fastf1.get_session(2026, 'Shanghai', 'FP1')
    session.load()

    # 3. Pick Lewis Hamilton (now in the Ferrari!)
    driver_laps = session.laps.pick_driver('HAM')
    
    # 4. Clean the data for Practice (FP1)
    # We only want 'accurate' laps and we filter out very slow 
    # out-laps (anything slower than 107% of his best time)
    valuable_laps = driver_laps.pick_accurate()
    if not valuable_laps.empty:
        fastest_lap = valuable_laps['LapTime'].min()
        valuable_laps = valuable_laps[valuable_laps['LapTime'] < fastest_lap * 1.07]

    # 5. Prepare the ML variables
    X = valuable_laps[['TyreLife']] 
    y = valuable_laps['LapTime'].dt.total_seconds()

    # 6. Train the Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', label='Actual Laps (2026 Ferrari)') # Ferrari Red!
    plt.plot(X, model.predict(X), color='black', linewidth=2, label='Degradation Trend')

    plt.title('LIVE ANALYSIS: Lewis Hamilton Tyre Deg - 2026 Chinese GP (FP1)')
    plt.xlabel('Tyre Age (Laps)')
    plt.ylabel('Lap Time (Seconds)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save and Show
    plt.savefig('tyre_plot_2026.png')
    plt.show()

    print(f"Success! Predicted pace for lap 10 of this stint: {model.predict([[10]])[0]:.3f}s")

except Exception as e:
    print(f"Data is likely still processing on F1 servers. Error: {e}")
    print("TIP: If it's a 'DataNotLoadedError', try again in 15-20 minutes!")