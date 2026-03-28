<div align="center">

# 🏎️ F1 Tyre Strategy Predictor: The Hamilton-Ferrari Era
### *Real-Time Machine Learning Analysis for the 2026 Chinese GP*

![F1](https://img.shields.io/badge/Formula_1-FF1801?style=for-the-badge&logo=formula1&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<p align="center">
  <img src="https://www.bing.com/th/id/OGC.b2ea9f99bdbd24038e381c9b9995311a?o=7&pid=1.7&rm=3&rurl=https%3a%2f%2fmedia.tenor.com%2fb2jBft1FOzkAAAAM%2fleweh-lewis-hamilton.gif&ehk=JOZRUDP2DQ5C1KGiQfFvgv3TRe1Q5RAnMCPexrvvM%2bM%3d" width="250" alt="Hamilton Ferrari GIF">
  
## 🗺️ Visual Telemetry: Suzuka Speed Map
![Track Heatmap](suzuka_heatmap.png)
*This heatmap visualizes LH44's speed across the Suzuka International Racing Course. Darker regions indicate heavy braking zones, while bright yellow highlights high-speed sections like the 130R.*

</p>

---

## 📊 Live Comparative Analysis
![Compound Analysis](compound_comparison.png)

*“In F1, speed is a given. Strategy is the variable.”*

---
</div>

## 🧠 Project Overview
As a 3rd-year B.Tech AI student at Gautam Buddha University, I developed this tool to decode the "Tyre Cliff." By leveraging the **FastF1 API** and **Scikit-Learn**, this project analyzes Lewis Hamilton's performance as he transitions into the 2026 Ferrari era.

### 🎯 Key Features
- **Multi-Compound Modeling:** Compares degradation rates between **Soft (Red)** and **Medium (Yellow)** compounds.
- **Cross-Era Benchmarking:** Analyzes 2024 Mercedes telemetry against live 2026 Ferrari FP1 data.
- **Noise Reduction:** Implemented a 107% lap-time filter to eliminate "dirty air" and pit-sequence outliers.

## 🛠️ Tech Stack
- **AI/ML:** Linear Regression (Scikit-Learn)
- **Data:** FastF1 API, Pandas, NumPy
- **Visuals:** Matplotlib (Custom F1 Theme)
- **Version Control:** Git/GitHub

## 🏁 Strategy Insights
| Compound | Initial Pace | Degradation Slope | Stint Potential |
| :--- | :--- | :--- | :--- |
| **Soft (C4)** | High Grip | Steeper (+0.12s/lap) | Aggressive / Short |
| **Medium (C3)** | Balanced | Moderate (+0.05s/lap) | Optimal Race Pace |

---

<details>
<summary><b>📂 Technical Implementation Details</b></summary>

### Data Pipeline:
1. **Cache Management:** Uses `fastf1.Cache` to minimize API calls.
2. **Feature Engineering:** Extracts `TyreLife` and `LapTime` (converted to total seconds).
3. **Linear Fit:** Calculates the coefficient of degradation to predict "the cliff."

### How to Run:
```bash
git clone [https://github.com/vanshdeep2402/f1-tyre-management-ml.git](https://github.com/vanshdeep2402/f1-tyre-management-ml.git)
pip install fastf1 scikit-learn matplotlib pandas
python china_gp_ml.py

