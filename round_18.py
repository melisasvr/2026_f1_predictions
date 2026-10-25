"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 18: UNITED STATES GP
 Circuit of the Americas, Austin | Race Date: October 25, 2026
============================================================
 Models   : Gradient Boosting Regressor + XGBoost (averaged)
 Target   : Race pace = qualifying * 1.07
 Track map: Native Vector-based COTA map with S1, S2, S3 sectors
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
from scipy.interpolate import interp1d
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
print("  F1 PREDICTIONS 2026-ROUND 18: UNITED STATES GP (COTA)")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER  (qualifying Sat 29°C / race Sun 28°C)
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 29
RACE_TEMP        = 28
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # -1°C
RAIN_PROBABILITY = 0.20    # 20% race day
HUMIDITY         = 61      # % race day
WIND_SPEED       = 13      # km/h race day

print(f"\nQualifying: {QUALIFYING_TEMP}C -> Race: {RACE_TEMP}C (delta {TEMP_DELTA}C)")
print(f"Rain: {int(RAIN_PROBABILITY*100)}% | Humidity: {HUMIDITY}% | Wind: {WIND_SPEED} km/h")

# ══════════════════════════════════════════════════════════
# 2. GP Q3 QUALIFYING (official timing screen)
# ══════════════════════════════════════════════════════════
POLE_TIME = 94.035

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Kimi Antonelli", "Max Verstappen", "Isack Hadjar",
        "Charles Leclerc", "Lewis Hamilton", "George Russell",
        "Oscar Piastri", "Lando Norris", "Arvid Lindblad",
        "Liam Lawson",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        94.035,   # Antonelli POLE
        94.214,   # +0.179
        94.250,   # +0.215
        94.267,   # +0.232
        94.311,   # +0.276
        94.476,   # +0.441
        94.803,   # +0.768
        95.049,   # +1.014
        96.049,   # +2.014
        96.349,   # Lawson
    ],
    "Team": [
        "Mercedes", "Red Bull Racing", "Red Bull Racing",
        "Ferrari",  "Ferrari",         "Mercedes",
        "McLaren",  "McLaren",         "Racing Bulls",
        "Racing Bulls",
    ],
    "GridPenalty (s)":   [0] * 10,
    "SprintWinnerBoost": [0] * 10,
})

DRIVER_CODES = {
    "Kimi Antonelli":  "ANT", "Max Verstappen":  "VER",
    "Isack Hadjar":    "HAD", "Charles Leclerc": "LEC",
    "Lewis Hamilton":  "HAM", "George Russell":  "RUS",
    "Oscar Piastri":   "PIA", "Lando Norris":    "NOR",
    "Arvid Lindblad":  "LIN", "Liam Lawson":     "LAW",
}
qualifying_2026["DriverCode"] = qualifying_2026["Driver"].map(DRIVER_CODES)
qualifying_2026["GapFromPole (s)"] = qualifying_2026["QualifyingTime (s)"] - POLE_TIME

# ══════════════════════════════════════════════════════════
# 3. TEAM COLOURS & SCORES
# ══════════════════════════════════════════════════════════
TEAM_COLORS = {
    "Mercedes": "#00D2BE", "McLaren": "#FF8000", "Ferrari": "#DC0000",
    "Red Bull Racing": "#3671C6", "Racing Bulls": "#6692FF",
    "Alpine": "#FF87BC", "Aston Martin": "#358C75", "Williams": "#64C4FF",
    "Haas": "#B6BABD", "Audi": "#B8B8B8", "Cadillac": "#C8102E",
}

ADJUSTED_TEAM_SCORE = {
    "Mercedes": 9.3, "McLaren": 9.0, "Ferrari": 9.0, "Red Bull Racing": 9.0,
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

WET_PERFORMANCE = {
    "ANT": 0.972, "VER": 0.966, "HAD": 0.980, "LEC": 0.974, "HAM": 0.964,
    "RUS": 0.966, "PIA": 0.975, "NOR": 0.976, "LIN": 0.983, "LAW": 0.979,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(WET_PERFORMANCE)
qualifying_2026["PoleWetBonus"] = 0.0

nan = np.nan
RESULTS_2026 = {
    "ANT": [2, 1, 1, 1, 1, 1, 4, 3, 20, 1, 3, 2, 1, 1, 5, 2, 3],
    "VER": [20, 20, 8, 5, 3, 20, 5, 2, 20, 3, 2, 20, 3, 2, 2, 1, 1],
    "HAD": [20, 8, 9, 20, 5, 3, 6, 20, 5, 6, 6, 20, 20, 20, 3, 5, nan],
    "LEC": [3, 4, 3, 6, 4, 20, 20, 20, 1, 2, 4, 5, 20, 4, 4, 4, 2],
    "HAM": [7, 3, 6, 7, 2, 2, 1, 5, 3, 4, 5, 4, 6, 20, 6, 3, nan],
    "RUS": [1, 2, 4, 4, 20, 20, 2, 1, 2, 20, 7, 3, 2, 5, 1, 20, nan],
    "PIA": [22, 2, 2, 3, 20, 4, 20, 4, 20, 5, 20, 6, 5, 8, 20, 6, nan],
    "NOR": [5, 20, 5, 2, 20, 20, 3, 20, 4, 7, 1, 1, 4, 3, 5, 9, nan],
    "LIN": [8, 20, 14, 20, 6, 6, 20, 20, 7, 9, 10, 10, 8, 9, 7, 10, nan],
    "LAW": [20, 20, 20, 20, 7, 5, 20, 20, 6, 20, 8, 7, 20, 6, 20, 7, nan],
}
circuit_scores = {
    code: round(1 + (np.nanmean(res) - 1) * (4 / 19), 3)
    for code, res in RESULTS_2026.items()
}
qualifying_2026["CircuitScore"] = qualifying_2026["DriverCode"].map(circuit_scores).fillna(3.5)

qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.33
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.37
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.30
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.07

qualifying_2026["RainProbability"] = RAIN_PROBABILITY
qualifying_2026["Temperature"]     = RACE_TEMP
qualifying_2026["TempDelta"]       = TEMP_DELTA
qualifying_2026["Humidity"]        = HUMIDITY
qualifying_2026["WindSpeed"]       = WIND_SPEED

# ══════════════════════════════════════════════════════════
# 4. MODEL TRAINING & PREDICTIONS
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

gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
mae_gb = mean_absolute_error(y_test, gb.predict(X_test))

if HAS_XGB:
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42, objective="reg:squarederror")
    xgb.fit(X_train, y_train)
    mae_xgb = mean_absolute_error(y_test, xgb.predict(X_test))
    mae_ens = mean_absolute_error(y_test, (gb.predict(X_test) + xgb.predict(X_test)) / 2)
    mae = mae_ens
else:
    mae = mae_gb

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
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 5. COTA VECTOR MAP GENERATOR FUNCTION (Cleaned Layout)
# ══════════════════════════════════════════════════════════
def draw_cota_sectors(ax):
    waypoints = np.array([
        [0.0, 4.0],   # T20 / Start Straight
        [3.0, 0.0],   # T1 (Uphill hairpin)
        [4.5, 1.0],   # T2
        [5.2, 1.8],   # T3
        [6.0, 2.7],   # T4 (Esses)
        [6.6, 3.2],   # T5
        [7.2, 3.0],   # T6
        [7.5, 2.5],   # T7
        [8.3, 3.0],   # T8
        [9.2, 3.7],   # T9
        [10.8, 3.8],  # T10
        [12.8, 6.2],  # T11 (Hairpin)
        [8.0, 5.5],   # Back Straight (to T12)
        [5.0, 5.0],   # T12
        [4.8, 4.2],   # T13
        [4.2, 3.7],   # T14
        [3.6, 4.4],   # T15
        [2.6, 4.2],   # T16
        [2.0, 3.2],   # T17/T18
        [1.3, 4.6],   # T19
        [0.0, 4.0],   # T20 Back to start
    ])
    
    t = np.linspace(0, 1, len(waypoints))
    t_smooth = np.linspace(0, 1, 500)
    fx = interp1d(t, waypoints[:, 0], kind='cubic')
    fy = interp1d(t, waypoints[:, 1], kind='cubic')
    x = fx(t_smooth)
    y = fy(t_smooth)

    idx_s1_end = int(len(x) * 0.36)
    idx_s2_end = int(len(x) * 0.72)

    ax.plot(x, y, color='#111111', lw=16, zorder=1)
    ax.plot(x, y, color='#262626', lw=9, zorder=2)

    color_s1 = "#B066FF"
    color_s2 = "#00E5A3"
    color_s3 = "#FFD000"

    ax.plot(x[:idx_s1_end+1], y[:idx_s1_end+1], color=color_s1, lw=4.5, zorder=3, label="S1")
    ax.plot(x[idx_s1_end:idx_s2_end+1], y[idx_s1_end:idx_s2_end+1], color=color_s2, lw=4.5, zorder=3, label="S2")
    ax.plot(x[idx_s2_end:], y[idx_s2_end:], color=color_s3, lw=4.5, zorder=3, label="S3")

    ax.scatter([x[idx_s1_end], x[idx_s2_end]], [y[idx_s1_end], y[idx_s2_end]],
               color="#00FFFF", s=50, zorder=4, edgecolor='white', linewidth=1)
    ax.plot([x[0]-0.1, x[0]+0.1], [y[0]-0.2, y[0]+0.2], color="white", lw=4, zorder=5)

    ax.text(0.74, 0.88, "S1", color=color_s1, transform=ax.transAxes, fontsize=11, fontweight='bold', fontfamily='monospace')
    ax.text(0.82, 0.88, "S2", color=color_s2, transform=ax.transAxes, fontsize=11, fontweight='bold', fontfamily='monospace')
    ax.text(0.90, 0.88, "S3", color=color_s3, transform=ax.transAxes, fontsize=11, fontweight='bold', fontfamily='monospace')

    ax.text(0.02, 0.88, "CIRCUIT OF THE AMERICAS — SECTORS", color="white", transform=ax.transAxes,
            fontsize=12, fontweight='bold', fontfamily='monospace')
    
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(x.min() - 1.5, x.max() + 1.5)
    ax.set_ylim(y.min() - 1.5, y.max() + 1.5)

# ══════════════════════════════════════════════════════════
# 6. DASHBOARD & PICTURE GENERATION
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"
driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 32), facecolor="#0f0f0f")
fig.suptitle(
    "F1 2026 — ROUND 18: UNITED STATES GP\n"
    "CIRCUIT OF THE AMERICAS, AUSTIN  |  OCT 25, 2026  |  28°C  |  61% HUMIDITY",
    fontsize=16, fontweight="bold", color="white", fontfamily=FONT, y=0.985
)
gs = GridSpec(4, 2, figure=fig, height_ratios=[1.3, 1, 1, 1], hspace=0.42, wspace=0.28)

# Row 1: Vector Sector Track Map
ax_map = fig.add_subplot(gs[0, :])
ax_map.set_facecolor("#0f0f0f")
draw_cota_sectors(ax_map)

# Row 2: Predicted Order
ax1 = fig.add_subplot(gs[1, :])
ax1.barh(data["Driver"][::-1], data["PredictedLapTime (s)"][::-1],
         color=driver_colors[::-1], edgecolor="white", linewidth=0.4, height=0.7)
ax1.set_title("Predicted Race Finishing Order", fontsize=13, fontweight="bold",
              color="white", fontfamily=FONT, pad=12)
ax1.set_xlabel("Predicted Avg Lap Time (s) — lower = faster", color="#AAAAAA", fontsize=9, fontfamily=FONT)
ax1.tick_params(colors="white", labelsize=9)
ax1.set_facecolor("#1a1a1a")
for spine in ax1.spines.values():
    spine.set_edgecolor("#333333")
seen, patches = set(), []
for _, row in data.iterrows():
    if row["Team"] not in seen:
        seen.add(row["Team"])
        patches.append(mpatches.Patch(color=TEAM_COLORS.get(row["Team"], "#FFF"), label=row["Team"]))
ax1.legend(handles=patches, loc="lower right", fontsize=8, facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")

# Row 3: Qualifying Gap + Feature Importance
ax2 = fig.add_subplot(gs[2, 0])
qs = qualifying_2026.sort_values("GapFromPole (s)")
ax2.barh(qs["Driver"][::-1], qs["GapFromPole (s)"][::-1],
         color=[TEAM_COLORS.get(t, "#FFF") for t in qs["Team"]][::-1],
         edgecolor="white", linewidth=0.4, height=0.65)
ax2.set_title("Qualifying Gap to Pole", fontsize=11, fontweight="bold", color="white", fontfamily=FONT, pad=10)
ax2.set_xlabel("Gap (s)", color="#AAAAAA", fontsize=9, fontfamily=FONT)
ax2.tick_params(colors="white", labelsize=8)
ax2.set_facecolor("#1a1a1a")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333333")
for i, (_, row) in enumerate(qs[::-1].iterrows()):
    gap = row["GapFromPole (s)"]
    note = " (est.)" if row["Driver"] == "Liam Lawson" else ""
    ax2.text(gap + 0.02, i, f"+{gap:.3f}s{note}", va="center", fontsize=7.5, color="white", fontfamily=FONT)

ax3 = fig.add_subplot(gs[2, 1])
imp = gb.feature_importances_ / gb.feature_importances_.sum()
title = "Feature Importance (Gradient Boosting)"
if HAS_XGB:
    imp_x = xgb.feature_importances_ / xgb.feature_importances_.sum()
    imp = (imp + imp_x) / 2
    title = "Feature Importance (GB + XGBoost average)"
labels = [c.replace(" (s)", "") for c in FEATURE_COLS]
order = np.argsort(imp)
ax3.barh([labels[i] for i in order], imp[order],
         color=plt.cm.Blues(np.linspace(0.3, 0.95, len(imp))),
         edgecolor="white", linewidth=0.3, height=0.6)
ax3.set_title(title, fontsize=11, fontweight="bold", color="white", fontfamily=FONT, pad=10)
ax3.tick_params(colors="white", labelsize=7)
ax3.set_facecolor("#1a1a1a")
for spine in ax3.spines.values():
    spine.set_edgecolor("#333333")

# Row 4: Clean, Non-Jumbled Predicted Podium + Model Comparison
ax4 = fig.add_subplot(gs[3, 0])
ax4.set_facecolor("#1a1a1a")
ax4.axis("off")
ax4.set_title("Predicted Podium", fontsize=11, fontweight="bold", color="white", fontfamily=FONT, pad=10)
podium = data[data["PredictedPosition"] <= 3].sort_values("PredictedPosition")

y_positions = [0.78, 0.50, 0.22]
labels_podium = ["1st", "2nd", "3rd"]

for i, (_, row) in enumerate(podium.iterrows()):
    y0 = y_positions[i]
    # Position badge
    ax4.text(0.15, y0, labels_podium[i], ha="left", va="center", fontsize=11, fontweight="bold", 
             color="#FFD700" if i==0 else "#C0C0C0" if i==1 else "#CD7F32", fontfamily=FONT, transform=ax4.transAxes)
    # Driver Name
    ax4.text(0.38, y0, row["Driver"], ha="left", va="center", fontsize=10, fontweight="bold", 
             color=TEAM_COLORS.get(row["Team"], "#FFF"), fontfamily=FONT, transform=ax4.transAxes)
    # Team Name
    ax4.text(0.82, y0, row["Team"], ha="right", va="center", fontsize=8.5, color="#AAAAAA", fontfamily=FONT, transform=ax4.transAxes)

ax5 = fig.add_subplot(gs[3, 1])
ax5.set_facecolor("#1a1a1a")
ax5.axis("off")
ax5.set_title("Model Comparison (top 5 by each model)", fontsize=11, fontweight="bold", color="white", fontfamily=FONT, pad=12)
if HAS_XGB:
    gb_top  = data.sort_values("GB (s)")["Driver"].head(5).tolist()
    xgb_top = data.sort_values("XGB (s)")["Driver"].head(5).tolist()
    ens_top = data["Driver"].head(5).tolist()
    lines = [f"{'#':<3}{'GradBoost':<18}{'XGBoost':<18}{'Final':<16}"]
    for i in range(5):
        lines.append(f"{i+1:<3}{gb_top[i].split()[-1]:<18}"
                     f"{xgb_top[i].split()[-1]:<18}{ens_top[i].split()[-1]:<16}")
    lines.append("")
    lines.append(f"MAE  GB {mae_gb:.3f}s | XGB {mae_xgb:.3f}s | Ens {mae_ens:.3f}s")
    ax5.text(0.05, 0.85, "\n".join(lines), va="top", ha="left", fontsize=10, color="white", fontfamily=FONT, transform=ax5.transAxes)

fig.text(0.5, 0.008,
         f"MAE: {mae:.3f}s  |  Race: {RACE_TEMP}°C  |  Rain: {int(RAIN_PROBABILITY*100)}%  |  "
         f"Pole: Antonelli  |  No grid penalty  |  Lawson time estimated",
         ha="center", fontsize=8, color="#888888", fontfamily=FONT)

plt.savefig("round_18_cota_prediction.png", dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
print("Picture saved -> round_18_cota_prediction.png")
plt.show()