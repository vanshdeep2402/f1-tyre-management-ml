import fastf1
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Setup a 'Cache'. F1 data is huge, this saves it on your PC 
# so you don't have to download it every time you run the code.
fastf1.Cache.enable_cache('f1_cache') 

print("Fetching data from the 2024 Chinese GP...")

# 2. Load the 2024 China Race ('R')
session = fastf1.get_session(2024, 'Shanghai', 'R')
session.load()

# 3. Pick a driver to study (Let's use Lewis Hamilton - HAM)
driver_laps = session.laps.pick_driver('HAM')

# 4. Clean the data. 
# We only want 'accurate' laps (no pit stops, no safety cars)
# because those mess up our tyre wear prediction.
valuable_laps = driver_laps.pick_accurate()

# 5. Get the data points for our ML model
# X = Independent variable (Tyre Age)
# y = Dependent variable (Lap Time in seconds)
X = valuable_laps[['TyreLife']] 
y = valuable_laps['LapTime'].dt.total_seconds()

# 6. Initialize and Train the 'Brain' (The Model)
model = LinearRegression()
model.fit(X, y)

# 7. Use the brain to predict!
# If a tyre has lasted 15 laps, how slow will the next lap be?
prediction = model.predict([[15]])

print(f"Predicted lap time for Hamilton on lap 15: {prediction[0]:.3f} seconds")
