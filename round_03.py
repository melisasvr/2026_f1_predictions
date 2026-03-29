"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 3: JAPANESE GP
 Suzuka International Racing Course | March 30, 2026
 
 CHANGES FROM ORIGINAL:
 - Removed fastf1 dependency (no 2025 data loading)
 - Circuit scores derived from 2026 AUS + CHN race form averages
 - Sector times synthetically derived from qualifying pace ratios
 - Race pace estimated at quali * 1.07 (7% slower)
 - IsRookie flag applied to Hadjar AND Lindblad (+0.45s penalty)
 - Model trains on the 2026 grid itself (quali→race pace mapping)
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

print("=" * 62)
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 3: JAPANESE GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 20    # °C
RACE_TEMP        = 23    # °C
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # +3°C
RAIN_PROBABILITY = 0.20  # 20% precipitation
HUMIDITY         = 71    # % race day humidity
WIND_SPEED       = 11    # km/h

print(f"\n🌡️  Weather: Qualifying {QUALIFYING_TEMP}°C → Race {RACE_TEMP}°C (Δ+{TEMP_DELTA}°C)")
print(f"🌧️  Rain: {int(RAIN_PROBABILITY*100)}%  |  💧 Humidity: {HUMIDITY}%  |  💨 Wind: {WIND_SPEED}km/h")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
# ══════════════════════════════════════════════════════════
POLE_TIME = 88.778  # Antonelli 1:28.778

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Kimi Antonelli",
        "George Russell",
        "Oscar Piastri",
        "Charles Leclerc",
        "Lando Norris",
        "Lewis Hamilton",
        "Pierre Gasly",
        "Isack Hadjar",
        "Gabriel Bortoleto",
        "Arvid Lindblad",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        88.778,   # 1:28.778
        89.076,   # 1:29.076
        89.132,   # 1:29.132
        89.405,   # 1:29.405
        89.409,   # 1:29.409
        89.567,   # 1:29.567
        89.691,   # 1:29.691
        89.978,   # 1:29.978
        90.274,   # 1:30.274
        90.319,   # 1:30.319
    ],
    "Team": [
        "Mercedes", "Mercedes", "McLaren",
        "Ferrari",  "McLaren",  "Ferrari",
        "Alpine",   "Red Bull",
        "Audi",     "RB",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # CHANGED: Hadjar (P8) also flagged as rookie alongside Lindblad (P10)
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    "SprintWinnerBoost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Kimi Antonelli":    "ANT",
    "George Russell":    "RUS",
    "Oscar Piastri":     "PIA",
    "Charles Leclerc":   "LEC",
    "Lando Norris":      "NOR",
    "Lewis Hamilton":    "HAM",
    "Pierre Gasly":      "GAS",
    "Isack Hadjar":      "HAD",
    "Gabriel Bortoleto": "BOR",
    "Arvid Lindblad":    "LIN",
}
qualifying_2026["DriverCode"] = qualifying_2026["Driver"].map(DRIVER_CODES)
qualifying_2026["GapFromPole (s)"] = (
    qualifying_2026["QualifyingTime (s)"] - POLE_TIME
)

# ══════════════════════════════════════════════════════════
# 3. TEAM COLOURS
# ══════════════════════════════════════════════════════════
TEAM_COLORS = {
    "Mercedes":        "#00D2BE",
    "McLaren":         "#FF8000",
    "Ferrari":         "#DC0000",
    "Red Bull":        "#3671C6",
    "RB":              "#6692FF",
    "Alpine":          "#FF87BC",
    "Aston Martin":    "#358C75",
    "Williams":        "#64C4FF",
    "Haas":            "#B6BABD",
    "Audi":            "#B8B8B8",
    "Cadillac":        "#C8102E",
}

# ══════════════════════════════════════════════════════════
# 4. ADJUSTED TEAM SCORE
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.8,
    "Ferrari":         7.8,
    "McLaren":         7.0,
    "Red Bull":        4.5,
    "Alpine":          5.5,
    "RB":              4.5,
    "Haas":            5.0,
    "Aston Martin":    4.0,
    "Williams":        4.0,
    "Audi":            3.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "ANT": 0.974,
    "RUS": 0.969,
    "PIA": 0.977,
    "LEC": 0.976,
    "NOR": 0.978,
    "HAM": 0.976,
    "GAS": 0.979,
    "HAD": 0.984,
    "BOR": 0.983,
    "LIN": 0.990,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. ERS DEPENDENCY SCORE
# ══════════════════════════════════════════════════════════
ERS_DEPENDENCY = {
    "Mercedes":        8.5,
    "McLaren":         8.5,
    "Ferrari":         7.0,
    "Red Bull":        6.0,
    "Alpine":          6.5,
    "RB":              6.0,
    "Haas":            7.0,
    "Aston Martin":    8.5,
    "Williams":        8.5,
    "Audi":            7.5,
    "Cadillac":        7.0,
}
qualifying_2026["ERSDependencyScore"] = qualifying_2026["Team"].map(
    ERS_DEPENDENCY
)

# ══════════════════════════════════════════════════════════
# 7. CIRCUIT SCORE — CHANGED
# Previously loaded from fastf1 2025 Japanese GP results.
# Now derived from 2026 AUS (R1) + CHN (R2) finishing positions.
#
# AUS: ANT 1, RUS 2, NOR 3, PIA 4, LEC 5, HAM 6, GAS 8, HAD 10, BOR 12, LIN 15
# CHN: PIA 1, NOR 2, ANT 3, LEC 4, HAM 5, RUS 6, GAS 7, HAD 9,  BOR 11, LIN 14
# CircuitScore = average finish position across both races
# ══════════════════════════════════════════════════════════
form_scores = {
    "ANT": (1 + 3)  / 2,   # 2.0
    "RUS": (2 + 6)  / 2,   # 4.0
    "PIA": (4 + 1)  / 2,   # 2.5
    "LEC": (5 + 4)  / 2,   # 4.5
    "NOR": (3 + 2)  / 2,   # 2.5
    "HAM": (6 + 5)  / 2,   # 5.5
    "GAS": (8 + 7)  / 2,   # 7.5
    "HAD": (10 + 9) / 2,   # 9.5
    "BOR": (12 + 11)/ 2,   # 11.5
    "LIN": (15 + 14)/ 2,   # 14.5
}
qualifying_2026["CircuitScore"] = qualifying_2026["DriverCode"].map(form_scores)
# Normalise to 1–5 scale (matching original's normalisation logic)
raw_max = max(form_scores.values())
qualifying_2026["CircuitScore"] = (
    1 + (qualifying_2026["CircuitScore"] - 1) * (4 / (raw_max - 1))
)

# ══════════════════════════════════════════════════════════
# 8. SECTOR TIMES — CHANGED
# Previously sourced from fastf1 2025 lap data.
# Now synthetically derived from qualifying pace using
# approximate Suzuka sector proportions:
#   S1 ≈ 30.5%,  S2 ≈ 42.8%,  S3 ≈ 26.7%  of lap time
# Race pace estimated at quali × 1.07 (~7% slower on race tyres)
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.305
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.428
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.267
qualifying_2026["LapTime (s)"]     = qualifying_2026["QualifyingTime (s)"] * 1.07

# ══════════════════════════════════════════════════════════
# 9. WEATHER FEATURES
# ══════════════════════════════════════════════════════════
qualifying_2026["RainProbability"] = RAIN_PROBABILITY
qualifying_2026["Temperature"]     = RACE_TEMP
qualifying_2026["TempDelta"]       = TEMP_DELTA
qualifying_2026["Humidity"]        = HUMIDITY
qualifying_2026["WindSpeed"]       = WIND_SPEED

# ══════════════════════════════════════════════════════════
# 10. FEATURE COLUMNS
# ══════════════════════════════════════════════════════════
FEATURE_COLS = [
    "QualifyingTime (s)",
    "GapFromPole (s)",
    "AdjustedTeamScore",
    "GridPenalty (s)",
    "WetPerformanceFactor",
    "RainProbability",
    "Temperature",
    "TempDelta",
    "Humidity",
    "WindSpeed",
    "ERSDependencyScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",
    "SprintWinnerBoost",
]
TARGET = "LapTime (s)"

# ══════════════════════════════════════════════════════════
# 11. TRAIN MODEL — CHANGED
# Previously trained on fastf1 2025 sector/lap data merged
# with 2026 qualifying. Now trains directly on the 2026 grid
# using synthetic race pace targets (quali × 1.07).
# ══════════════════════════════════════════════════════════
X = qualifying_2026[FEATURE_COLS].fillna(0)
y = qualifying_2026[TARGET]

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.04,
    max_depth=3,
    random_state=38,
)
model.fit(X, y)

# ══════════════════════════════════════════════════════════
# 12. PREDICT 2026 RACE FINISH
# ══════════════════════════════════════════════════════════
qualifying_2026["PredictedLapTime (s)"] = model.predict(X)

# CHANGED: penalty reduced to +0.45s (was +0.5s) and applied to
# both rookies (Hadjar P8 + Lindblad P10)
qualifying_2026.loc[qualifying_2026["IsRookie"] == 1, "PredictedLapTime (s)"] += 0.45

data = qualifying_2026.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 13. PRINT RESULTS
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  🚦  2026 JAPANESE GP — STARTING GRID (TOP 10)")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<16} {'Qual Time'}")
print("  " + "─" * 57)
for _, row in qualifying_2026.sort_values("GridPosition").iterrows():
    mins = int(row['QualifyingTime (s)'] // 60)
    secs = row['QualifyingTime (s)'] % 60
    print(f"  P{int(row['GridPosition']):<4} {row['Driver']:<22} {row['Team']:<16} {mins}:{secs:06.3f}")

medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 JAPANESE GP — PREDICTED RACE RESULT")
print("  📊  Suzuka quali pace + AUS/CHN form + team scores")
print("=" * 62)
print(f"  {'Pos':<6} {'Driver':<22} {'Team':<16} {'Pred Lap (s)':>12}")
print("  " + "─" * 57)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos:<2}")
    print(f"  {icon:<6} {row['Driver']:<22} {row['Team']:<16}"
          f" {row['PredictedLapTime (s)']:>12.3f}")

print("=" * 62)
print()
print("  🏆  PODIUM PREDICTION")
print("  " + "─" * 40)
for _, row in data[data["PredictedPosition"] <= 3].sort_values("PredictedPosition").iterrows():
    pos = int(row["PredictedPosition"])
    print(f"  {medals[pos]}  {row['Driver']} ({row['Team']})")

# ══════════════════════════════════════════════════════════
# 14. VISUALISATIONS (UNCHANGED from original)
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"
driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 3: JAPANESE GP\nSUZUKA INTERNATIONAL RACING COURSE  |  MARCH 30, 2026",
    fontsize=18, fontweight="bold", color="white", fontfamily=FONT, y=0.98
)
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
ax1.barh(data["Driver"][::-1], data["PredictedLapTime (s)"][::-1],
         color=driver_colors[::-1], edgecolor="white", linewidth=0.4, height=0.7)
ax1.set_facecolor("#1a1a1a")

ax2 = fig.add_subplot(gs[1, 0])
qual_sorted = qualifying_2026.sort_values("GapFromPole (s)")
ax2.barh(qual_sorted["Driver"][::-1], qual_sorted["GapFromPole (s)"][::-1],
         color=[TEAM_COLORS.get(t, "#FFF") for t in qual_sorted["Team"]][::-1],
         edgecolor="white", linewidth=0.4, height=0.65)
ax2.set_facecolor("#1a1a1a")

ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(data["ERSDependencyScore"], data["AdjustedTeamScore"],
            c=[TEAM_COLORS.get(t, "#FFF") for t in data["Team"]],
            s=200, edgecolors="white", linewidth=0.5, zorder=5)
ax3.set_facecolor("#1a1a1a")

ax4 = fig.add_subplot(gs[2, 0])
feat_import = model.feature_importances_
sorted_idx  = np.argsort(feat_import)
ax4.barh(np.array(FEATURE_COLS)[sorted_idx], feat_import[sorted_idx],
         color=plt.cm.plasma(np.linspace(0.2, 0.9, len(FEATURE_COLS))),
         edgecolor="white", linewidth=0.3, height=0.6)
ax4.set_facecolor("#1a1a1a")

ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor("#1a1a1a")
ax5.axis("off")
podium     = data[data["PredictedPosition"] <= 3].sort_values("PredictedPosition")
podium_y   = [0.75, 0.47, 0.19]
podium_icon = ["🥇", "🥈", "🥉"]
for i, (_, row) in enumerate(podium.iterrows()):
    ax5.text(0.5, podium_y[i] + 0.08, podium_icon[i],
             ha="center", va="center", fontsize=20, transform=ax5.transAxes)
    ax5.text(0.5, podium_y[i], row["Driver"],
             ha="center", va="center", fontsize=12, fontweight="bold",
             color=TEAM_COLORS.get(row["Team"], "#FFFFFF"),
             fontfamily=FONT, transform=ax5.transAxes)

plt.savefig("round_03_japan_prediction.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
print("\n✅ Chart saved → round_03_japan_prediction.png")
plt.show()