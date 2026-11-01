"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 19: MEXICO CITY GP
 Autodromo Hermanos Rodriguez | Race Date: November 1, 2026
============================================================
 Models   : Gradient Boosting Regressor + XGBoost (averaged)
 Target   : Race pace = qualifying * 1.07
 Notes    : NO grid-position penalty feature
            NO rookie penalty
            Normal weekend (no Sprint) -> SprintWinnerBoost = 0
 Track map: save your Mexico map screenshot as
            "mexico_track_map.jpeg" in the same folder as this script
 Install  : pip install xgboost   (already in requirements.txt)
 Author   : Melisa Sever
============================================================
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed -> running Gradient Boosting only.")
    print("Install it with:  pip install xgboost\n")

print("=" * 62)
print("  F1 PREDICTIONS 2026 — ROUND 19: MEXICO CITY GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER  (qualifying Sat 27°C / race Sun 26°C)
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 27
RACE_TEMP        = 26
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # -1°C
RAIN_PROBABILITY = 0.05    # race forecast <5%
HUMIDITY         = 49      # % race day
WIND_SPEED       = 10      # km/h race day

print(f"\nQualifying: {QUALIFYING_TEMP}C -> Race: {RACE_TEMP}C (delta {TEMP_DELTA}C)")
print(f"Rain: <5% | Humidity: {HUMIDITY}% | Wind: {WIND_SPEED} km/h")

# ══════════════════════════════════════════════════════════
# 2. GP QUALIFYING (official timing screen)
# ══════════════════════════════════════════════════════════
POLE_TIME = 76.514

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Lewis Hamilton", "George Russell", "Charles Leclerc",
        "Max Verstappen", "Kimi Antonelli", "Lando Norris",
        "Isack Hadjar", "Liam Lawson", "Arvid Lindblad",
        "Pierre Gasly",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        76.514,   # Hamilton   POLE
        76.523,   # +0.009  Russell
        76.633,   # +0.119  Leclerc
        76.692,   # +0.178  Verstappen
        76.707,   # +0.193  Antonelli
        76.824,   # +0.310  Norris
        76.900,   # +0.386  Hadjar
        77.029,   # +0.515  Lawson
        77.117,   # +0.603  Lindblad
        77.153,   # +0.639  Gasly
    ],
    "Team": [
        "Ferrari", "Mercedes", "Ferrari",
        "Red Bull Racing", "Mercedes", "McLaren",
        "Red Bull Racing", "Racing Bulls", "Racing Bulls",
        "Alpine",
    ],
    "GridPenalty (s)":   [0] * 10,
    "SprintWinnerBoost": [0] * 10,
})

DRIVER_CODES = {
    "Lewis Hamilton":  "HAM", "George Russell":  "RUS",
    "Charles Leclerc": "LEC", "Max Verstappen":  "VER",
    "Kimi Antonelli":  "ANT", "Lando Norris":    "NOR",
    "Isack Hadjar":    "HAD", "Liam Lawson":     "LAW",
    "Arvid Lindblad":  "LIN", "Pierre Gasly":    "GAS",
}
qualifying_2026["DriverCode"] = qualifying_2026["Driver"].map(DRIVER_CODES)
qualifying_2026["GapFromPole (s)"] = (
    qualifying_2026["QualifyingTime (s)"] - POLE_TIME
)

# ══════════════════════════════════════════════════════════
# 3. TEAM COLOURS
# ══════════════════════════════════════════════════════════
TEAM_COLORS = {
    "Mercedes": "#00D2BE", "McLaren": "#FF8000", "Ferrari": "#DC0000",
    "Red Bull Racing": "#3671C6", "Racing Bulls": "#6692FF",
    "Alpine": "#FF87BC", "Aston Martin": "#358C75", "Williams": "#64C4FF",
    "Haas": "#B6BABD", "Audi": "#B8B8B8", "Cadillac": "#C8102E",
}

# ══════════════════════════════════════════════════════════
# 4. TEAM-LEVEL SCORES
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes": 9.3, "Ferrari": 9.1, "Red Bull Racing": 9.0, "McLaren": 8.6,
    "Alpine": 7.0, "Racing Bulls": 5.8, "Audi": 6.0, "Williams": 5.5,
    "Haas": 5.0, "Aston Martin": 4.5, "Cadillac": 2.5,
}
TYRE_DEG = {
    "Mercedes": 2.0, "Ferrari": 2.0, "McLaren": 1.5, "Red Bull Racing": 2.0,
    "Alpine": 3.0, "Racing Bulls": 3.0, "Audi": 3.5, "Williams": 3.5,
    "Haas": 3.5, "Aston Martin": 3.0, "Cadillac": 4.5,
}
ERS_DEPENDENCY = {
    "Mercedes": 9.0, "McLaren": 9.0, "Ferrari": 6.5, "Red Bull Racing": 5.5,
    "Alpine": 6.5, "Racing Bulls": 5.5, "Williams": 9.0, "Audi": 7.0,
    "Haas": 6.5, "Aston Martin": 8.0, "Cadillac": 6.5,
}
RELIABILITY_RISK = {
    "Mercedes": 4.0, "McLaren": 2.5, "Ferrari": 2.5, "Red Bull Racing": 3.0,
    "Alpine": 4.0, "Racing Bulls": 3.5, "Audi": 5.0, "Williams": 4.0,
    "Haas": 4.0, "Aston Martin": 4.0, "Cadillac": 5.0,
}
qualifying_2026["AdjustedTeamScore"]    = qualifying_2026["Team"].map(ADJUSTED_TEAM_SCORE)
qualifying_2026["TyreDegScore"]         = qualifying_2026["Team"].map(TYRE_DEG)
qualifying_2026["ERSDependencyScore"]   = qualifying_2026["Team"].map(ERS_DEPENDENCY)
qualifying_2026["ReliabilityRiskScore"] = qualifying_2026["Team"].map(RELIABILITY_RISK)
qualifying_2026["HomeRaceBoost"]        = 0.0

# ══════════════════════════════════════════════════════════
# 5. WET FACTOR
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "HAM": 0.964, "RUS": 0.966, "LEC": 0.974, "VER": 0.966, "ANT": 0.972,
    "NOR": 0.976, "HAD": 0.980, "LAW": 0.979, "LIN": 0.983, "GAS": 0.977,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(WET_PERFORMANCE)

POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY
    if (p == 1 and RAIN_PROBABILITY >= 0.60) else 0.0
)

# ══════════════════════════════════════════════════════════
# 6. CIRCUIT SCORE
# ══════════════════════════════════════════════════════════
nan = np.nan
RESULTS_2026 = {
    "HAM": [7, 3, 6, 7, 2, 2, 1, 5, 3, 4, 5, 4, 6, 20, 6, 3, nan, 4],
    "RUS": [1, 2, 4, 4, 20, 20, 2, 1, 2, 20, 7, 3, 2, 5, 1, 20, nan, 8],
    "LEC": [3, 4, 3, 6, 4, 20, 20, 20, 1, 2, 4, 5, 20, 4, 4, 4, 2, 2],
    "VER": [20, 20, 8, 5, 3, 20, 5, 2, 20, 3, 2, 20, 3, 2, 2, 1, 1, 3],
    "ANT": [2, 1, 1, 1, 1, 1, 4, 3, 20, 1, 3, 2, 1, 1, 5, 2, 3, 1],
    "NOR": [5, 20, 5, 2, 20, 20, 3, 20, 4, 7, 1, 1, 4, 3, 5, 9, nan, 5],
    "HAD": [20, 8, 9, 20, 5, 3, 6, 20, 5, 6, 6, 20, 20, 20, 3, 5, nan, 6],
    "LAW": [20, 20, 20, 20, 7, 5, 20, 20, 6, 20, 8, 7, 20, 6, 20, 7, nan, 9],
    "LIN": [8, 20, 14, 20, 6, 6, 20, 20, 7, 9, 10, 10, 8, 9, 7, 10, nan, 20],
    "GAS": [10, 6, 10, 20, 8, 7, 20, 20, 20, 20, 12, 10, 7, 20, 20, 20, nan, 10],
}
circuit_scores = {
    code: round(1 + (np.nanmean(res) - 1) * (4 / 19), 3)
    for code, res in RESULTS_2026.items()
}
qualifying_2026["CircuitScore"] = qualifying_2026["DriverCode"].map(circuit_scores).fillna(3.5)

# ══════════════════════════════════════════════════════════
# 7. SECTOR TIMES + TARGET
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.34
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.36
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.30
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.07

qualifying_2026["RainProbability"] = RAIN_PROBABILITY
qualifying_2026["Temperature"]     = RACE_TEMP
qualifying_2026["TempDelta"]       = TEMP_DELTA
qualifying_2026["Humidity"]        = HUMIDITY
qualifying_2026["WindSpeed"]       = WIND_SPEED

# ══════════════════════════════════════════════════════════
# 8. FEATURES & MODEL TRAINING
# ══════════════════════════════════════════════════════════
FEATURE_COLS = [
    "QualifyingTime (s)", "GapFromPole (s)", "AdjustedTeamScore",
    "GridPenalty (s)", "WetPerformanceFactor", "PoleWetBonus",
    "RainProbability", "Temperature", "TempDelta", "Humidity", "WindSpeed",
    "ERSDependencyScore", "TyreDegScore", "HomeRaceBoost",
    "ReliabilityRiskScore", "Sector1Time (s)", "Sector2Time (s)",
    "Sector3Time (s)", "CircuitScore", "SprintWinnerBoost",
]
TARGET = "RacePace (s)"

X = qualifying_2026[FEATURE_COLS].fillna(0)
y = qualifying_2026[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
gb.fit(X_train, y_train)
mae_gb = mean_absolute_error(y_test, gb.predict(X_test))

if HAS_XGB:
    xgb = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        random_state=42, objective="reg:squarederror",
    )
    xgb.fit(X_train, y_train)
    mae_xgb = mean_absolute_error(y_test, xgb.predict(X_test))
    mae_ens = mean_absolute_error(
        y_test, (gb.predict(X_test) + xgb.predict(X_test)) / 2
    )
    mae = mae_ens
else:
    mae = mae_gb

# ══════════════════════════════════════════════════════════
# 9. PREDICTION CALCULATIONS
# ══════════════════════════════════════════════════════════
data = qualifying_2026.copy()
data["GB (s)"] = gb.predict(X)
if HAS_XGB:
    data["XGB (s)"] = xgb.predict(X)
    data["PredictedLapTime (s)"] = (data["GB (s)"] + data["XGB (s)"]) / 2
else:
    data["PredictedLapTime (s)"] = data["GB (s)"]

data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.02
data["WetBonus"] = (1 - data["WetPerformanceFactor"]) * RAIN_PROBABILITY * 100
data["PredictedLapTime (s)"] -= data["WetBonus"]
data["PredictedLapTime (s)"] -= data["PoleWetBonus"]

data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 10. PLOTTING FIGURE WITH EMBEDDED MAP
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"
driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

script_dir = os.path.dirname(os.path.abspath(__file__))
track_img = None
for name in ("mexico_track_map.jpeg", "mexico_track_map.jpg", "mexico_track_map.png"):
    path = os.path.join(script_dir, name)
    if os.path.exists(path):
        track_img = plt.imread(path)
        break

fig = plt.figure(figsize=(18, 30), facecolor="#0f0f0f")
fig.suptitle(
    "F1 2026 — ROUND 19: MEXICO CITY GP\n"
    "AUTODROMO HERMANOS RODRIGUEZ  |  NOV 1, 2026  |  26°C  |  49% HUMIDITY",
    fontsize=17, fontweight="bold", color="white", fontfamily=FONT, y=0.982
)
gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.0, 1.0, 1.0],
              hspace=0.45, wspace=0.35,
              left=0.10, right=0.95, top=0.93, bottom=0.045)

TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 13, 10, 10

def style_axis(ax):
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white", labelsize=TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

# Row 1: Track map image insertion (Clean, no text fallbacks)
ax_map = fig.add_subplot(gs[0, :])
ax_map.set_facecolor("#0f0f0f")
ax_map.axis("off")
if track_img is not None:
    ax_map.imshow(track_img, aspect="auto")
ax_map.set_title("Autodromo Hermanos Rodriguez", fontsize=TITLE_SIZE,
                 fontweight="bold", color="white", fontfamily=FONT, pad=18)

# Row 2: Predicted order
ax1 = fig.add_subplot(gs[1, :])
ax1.barh(data["Driver"][::-1], data["PredictedLapTime (s)"][::-1],
         color=driver_colors[::-1], edgecolor="white", linewidth=0.4, height=0.65)
ax1.set_title("Predicted Race Finishing Order", fontsize=TITLE_SIZE,
              fontweight="bold", color="white", fontfamily=FONT, pad=16)
ax1.set_xlabel("Predicted Avg Lap Time (s) — lower = faster",
               color="#AAAAAA", fontsize=LABEL_SIZE, fontfamily=FONT, labelpad=10)
style_axis(ax1)
seen, patches = set(), []
for _, row in data.iterrows():
    if row["Team"] not in seen:
        seen.add(row["Team"])
        patches.append(mpatches.Patch(color=TEAM_COLORS.get(row["Team"], "#FFF"),
                                      label=row["Team"]))
ax1.legend(handles=patches, loc="lower right", fontsize=9,
           facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")

# Row 3 Left: Qualifying gap
ax2 = fig.add_subplot(gs[2, 0])
qs = qualifying_2026.sort_values("GapFromPole (s)")
ax2.barh(qs["Driver"][::-1], qs["GapFromPole (s)"][::-1],
         color=[TEAM_COLORS.get(t, "#FFF") for t in qs["Team"]][::-1],
         edgecolor="white", linewidth=0.4, height=0.65)
ax2.set_title("Qualifying Gap to Pole", fontsize=TITLE_SIZE, fontweight="bold",
              color="white", fontfamily=FONT, pad=16)
ax2.set_xlabel("Gap (s)", color="#AAAAAA", fontsize=LABEL_SIZE,
               fontfamily=FONT, labelpad=10)
ax2.set_xlim(0, qs["GapFromPole (s)"].max() * 1.25)
style_axis(ax2)
for i, (_, row) in enumerate(qs[::-1].iterrows()):
    gap = row["GapFromPole (s)"]
    ax2.text(gap + 0.01, i, f"+{gap:.3f}s", va="center",
             fontsize=9, color="white", fontfamily=FONT)

# Row 3 Right: Feature Importance (Top 10)
ax3 = fig.add_subplot(gs[2, 1])
imp = gb.feature_importances_ / gb.feature_importances_.sum()
title = "Feature Importance (Top 10 - Gradient Boosting)"
if HAS_XGB:
    imp_x = xgb.feature_importances_ / xgb.feature_importances_.sum()
    imp = (imp + imp_x) / 2
    title = "Feature Importance (Top 10 - GB + XGB)"

labels = [c.replace(" (s)", "") for c in FEATURE_COLS]
order = np.argsort(imp)[-10:]
ax3.barh([labels[i] for i in order], imp[order],
         color=plt.cm.Blues(np.linspace(0.4, 0.95, len(order))),
         edgecolor="white", linewidth=0.3, height=0.65)
ax3.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", color="white",
              fontfamily=FONT, pad=16)
ax3.tick_params(colors="white", labelsize=9)
ax3.set_facecolor("#1a1a1a")
for spine in ax3.spines.values():
    spine.set_edgecolor("#333333")

# Row 4 Left: Predicted Podium
ax4 = fig.add_subplot(gs[3, 0])
ax4.set_facecolor("#1a1a1a")
ax4.axis("off")
ax4.set_title("Predicted Podium", fontsize=TITLE_SIZE, fontweight="bold",
              color="white", fontfamily=FONT, pad=16)
podium = data[data["PredictedPosition"] <= 3].sort_values("PredictedPosition")
y_positions = [0.74, 0.47, 0.20]
for i, (_, row) in enumerate(podium.iterrows()):
    y0 = y_positions[i]
    ax4.text(0.5, y0 + 0.08, ["1st", "2nd", "3rd"][i], ha="center", va="center",
             fontsize=12, color="#CCCCCC", fontfamily=FONT, transform=ax4.transAxes)
    ax4.text(0.5, y0, row["Driver"], ha="center", va="center", fontsize=11,
             fontweight="bold", color=TEAM_COLORS.get(row["Team"], "#FFF"),
             fontfamily=FONT, transform=ax4.transAxes)
    ax4.text(0.5, y0 - 0.07, row["Team"], ha="center", va="center", fontsize=9,
             color="#888888", fontfamily=FONT, transform=ax4.transAxes)

# Row 4 Right: Model Comparison Table
ax5 = fig.add_subplot(gs[3, 1])
ax5.set_facecolor("#1a1a1a")
ax5.axis("off")
ax5.set_title("Model Comparison (Top 5)", fontsize=TITLE_SIZE,
              fontweight="bold", color="white", fontfamily=FONT, pad=16)
if HAS_XGB:
    gb_top  = data.sort_values("GB (s)")["Driver"].head(5).tolist()
    xgb_top = data.sort_values("XGB (s)")["Driver"].head(5).tolist()
    ens_top = data["Driver"].head(5).tolist()
    
    table_data = [["Pos", "GradBoost", "XGBoost", "Ensemble"]]
    for i in range(5):
        table_data.append([
            str(i+1),
            gb_top[i].split()[-1],
            xgb_top[i].split()[-1],
            ens_top[i].split()[-1]
        ])
    
    table = ax5.table(cellText=table_data, loc="center", cellLoc="center", bbox=[0.05, 0.32, 0.9, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for key, cell in table.get_celld().items():
        cell.set_facecolor("#1a1a1a")
        cell.set_edgecolor("#333333")
        cell.set_text_props(color="white", fontfamily=FONT)
    for j in range(4):
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(weight="bold", color="white")

    ax5.text(0.05, 0.12,
             f"MAE | GB: {mae_gb:.3f}s | XGB: {mae_xgb:.3f}s | Ensemble: {mae_ens:.3f}s",
             va="center", ha="left", fontsize=9, color="#AAAAAA",
             fontfamily=FONT, transform=ax5.transAxes)
else:
    ax5.text(0.5, 0.5, "xgboost not installed\npip install xgboost",
             ha="center", va="center", color="#888888", fontsize=11,
             fontfamily=FONT, transform=ax5.transAxes)

fig.text(0.5, 0.012,
         f"MAE: {mae:.3f}s  |  Race: {RACE_TEMP}°C  |  Rain: <5%  |  "
         f"Pole: Hamilton  |  No grid penalty",
         ha="center", fontsize=9, color="#888888", fontfamily=FONT)

plt.savefig("round_19_mexico_prediction.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
print("Picture saved -> round_19_mexico_prediction.png")
plt.show()