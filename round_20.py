"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 20: SAO PAULO GP
 Autodromo Jose Carlos Pace (Interlagos) | Race Date: November 8, 2026
============================================================
 Models   : Gradient Boosting Regressor + XGBoost (averaged)
 Target   : Race pace = qualifying * 1.07
 Notes    : NO per-position grid penalty feature (removed by request)
            NO rookie penalty
            REAL announced penalties are applied with a small,
            adjustable amount (see PENALTY_SECONDS_PER_PLACE)
            Normal weekend (no Sprint) -> SprintWinnerBoost = 0
 Penalties: RUS -5 places (new ICE), LEC -10 places (engine parts)
            VER + PER back of grid, HAM -10 places: they were NOT in
            the Q3 top 10, so there is no qualifying time for them
            and they are not in this prediction.
 Install  : pip install xgboost   (already in requirements.txt)
 Author   : Melisa Sever
============================================================
"""

import os
import re
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
print("  F1 PREDICTIONS 2026 — ROUND 20: SAO PAULO GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER  (qualifying Sat: 28°C, 90% rain -> race Sun: 16°C, 35% rain)
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 28
RACE_TEMP        = 16
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # -12°C (biggest swing of the season)
RAIN_PROBABILITY = 0.35    # 35% race day (below the 60% PoleWetBonus threshold)
HUMIDITY         = 91      # % race day
WIND_SPEED       = 19      # km/h race day

# Wet bonus strength. 100 = same formula as every previous round.
# Lower it (e.g. 30) if the wet rating overpowers the qualifying gaps.
WET_BONUS_SCALE = 100

print(f"\nQualifying: {QUALIFYING_TEMP}C (wet) -> Race: {RACE_TEMP}C (delta {TEMP_DELTA}C)")
print(f"Rain: {int(RAIN_PROBABILITY*100)}% | Humidity: {HUMIDITY}% | Wind: {WIND_SPEED} km/h")

# ══════════════════════════════════════════════════════════
# 2. GP QUALIFYING (official timing screen, wet session)
#    Gaps converted to absolute seconds from pole 1:27.470
# ══════════════════════════════════════════════════════════
POLE_TIME = 87.470

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Lando Norris", "George Russell", "Kimi Antonelli",
        "Charles Leclerc", "Liam Lawson", "Nico Hulkenberg",
        "Isack Hadjar", "Oliver Bearman", "Fernando Alonso",
        "Alexander Albon",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],   # as qualified, before penalties
    "QualifyingTime (s)": [
        87.470,   # Norris      POLE
        87.674,   # +0.204  Russell
        88.095,   # +0.625  Antonelli
        88.289,   # +0.819  Leclerc
        88.321,   # +0.851  Lawson
        88.668,   # +1.198  Hulkenberg
        88.763,   # +1.293  Hadjar
        89.831,   # +2.361  Bearman
        90.311,   # +2.841  Alonso
        90.456,   # +2.986  Albon
    ],
    "Team": [
        "McLaren", "Mercedes", "Mercedes",
        "Ferrari", "Racing Bulls", "Audi",
        "Red Bull Racing", "Haas", "Aston Martin",
        "Williams",
    ],
    "SprintWinnerBoost": [0] * 10,   # no Sprint this weekend
})

DRIVER_CODES = {
    "Lando Norris":    "NOR", "George Russell":  "RUS",
    "Kimi Antonelli":  "ANT", "Charles Leclerc": "LEC",
    "Liam Lawson":     "LAW", "Nico Hulkenberg": "HUL",
    "Isack Hadjar":    "HAD", "Oliver Bearman":  "BEA",
    "Fernando Alonso": "ALO", "Alexander Albon": "ALB",
}
qualifying_2026["DriverCode"] = qualifying_2026["Driver"].map(DRIVER_CODES)
qualifying_2026["GapFromPole (s)"] = (
    qualifying_2026["QualifyingTime (s)"] - POLE_TIME
)

# ══════════════════════════════════════════════════════════
# 2b. REAL GRID PENALTIES (announced after qualifying)
#     Only drivers who are in the Q3 top 10 can be modelled.
#     Not modelled (no Q3 time): Verstappen (back of grid),
#     Perez (back of grid), Hamilton (-10 places)
# ══════════════════════════════════════════════════════════
PENALTY_PLACES = {"RUS": 5, "LEC": 10}
# Small adjustment per place lost (seconds on predicted average lap).
# Set to 0.0 to turn penalties off completely.
PENALTY_SECONDS_PER_PLACE = 0.02

qualifying_2026["PlacesLost"] = qualifying_2026["DriverCode"].map(PENALTY_PLACES).fillna(0)
qualifying_2026["GridPenalty (s)"] = qualifying_2026["PlacesLost"] * PENALTY_SECONDS_PER_PLACE

print("\nGrid penalties applied:")
for _, row in qualifying_2026[qualifying_2026["PlacesLost"] > 0].iterrows():
    print(f"  {row['Driver']:<18} -{int(row['PlacesLost'])} places "
          f"(+{row['GridPenalty (s)']:.2f}s on predicted lap)")
print("  Not modelled (not in Q3 top 10): Verstappen, Perez, Hamilton")

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
# 4. TEAM-LEVEL SCORES (hand-set estimates)
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes": 9.3, "Ferrari": 9.1, "Red Bull Racing": 9.0, "McLaren": 8.8,
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
qualifying_2026["HomeRaceBoost"]        = 0.0   # Bortoleto (Brazil) is not in the top 10

# ══════════════════════════════════════════════════════════
# 5. WET FACTOR (lower = better in the wet) — hand-set estimates
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "NOR": 0.976, "RUS": 0.966, "ANT": 0.972, "LEC": 0.974, "LAW": 0.979,
    "HUL": 0.975, "HAD": 0.980, "BEA": 0.979, "ALO": 0.968, "ALB": 0.977,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(WET_PERFORMANCE)

# PoleWetBonus only activates above 60% rain (35% here -> stays 0)
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY
    if (p == 1 and RAIN_PROBABILITY >= 0.60) else 0.0
)

# ══════════════════════════════════════════════════════════
# 6. CIRCUIT SCORE — 2026 finishing positions through Mexico (R19)
#    20 = DNF / outside top 10.  np.nan = result not known.
#    Drivers with no record here (HUL, BEA, ALO, ALB) get the neutral 3.5.
#    AUS CHN JPN MIA CAN MON ESP AUT GBR BEL HUN NED ITA MAD AZE MYS SGP USA MEX
# ══════════════════════════════════════════════════════════
nan = np.nan
RESULTS_2026 = {
    "NOR": [5, 20, 5, 2, 20, 20, 3, 20, 4, 7, 1, 1, 4, 3, 5, 9, nan, 5, 5],
    "RUS": [1, 2, 4, 4, 20, 20, 2, 1, 2, 20, 7, 3, 2, 5, 1, 20, nan, 8, 6],
    "ANT": [2, 1, 1, 1, 1, 1, 4, 3, 20, 1, 3, 2, 1, 1, 5, 2, 3, 1, 3],
    "LEC": [3, 4, 3, 6, 4, 20, 20, 20, 1, 2, 4, 5, 20, 4, 4, 4, 2, 2, 4],
    "LAW": [20, 20, 20, 20, 7, 5, 20, 20, 6, 20, 8, 7, 20, 6, 20, 7, nan, 9, 9],
    "HAD": [20, 8, 9, 20, 5, 3, 6, 20, 5, 6, 6, 20, 20, 20, 3, 5, nan, 6, 20],
}
circuit_scores = {
    code: round(1 + (np.nanmean(res) - 1) * (4 / 19), 3)
    for code, res in RESULTS_2026.items()
}
qualifying_2026["CircuitScore"] = qualifying_2026["DriverCode"].map(circuit_scores).fillna(3.5)

# ══════════════════════════════════════════════════════════
# 7. SYNTHETIC SECTOR TIMES + RACE-PACE TARGET
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.30
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.38
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.32
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.07

qualifying_2026["RainProbability"] = RAIN_PROBABILITY
qualifying_2026["Temperature"]     = RACE_TEMP
qualifying_2026["TempDelta"]       = TEMP_DELTA
qualifying_2026["Humidity"]        = HUMIDITY
qualifying_2026["WindSpeed"]       = WIND_SPEED

print("\nFeature set:")
print(qualifying_2026[[
    "Driver", "QualifyingTime (s)", "GapFromPole (s)",
    "AdjustedTeamScore", "WetPerformanceFactor", "CircuitScore",
]].to_string(index=False))

# ══════════════════════════════════════════════════════════
# 8. FEATURES (no per-position grid penalty feature)
#    GridPenalty (s) holds the small REAL-penalty adjustment above.
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

# ══════════════════════════════════════════════════════════
# 9. MODELS: Gradient Boosting + XGBoost, averaged
# ══════════════════════════════════════════════════════════
gb = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
gb.fit(X_train, y_train)
mae_gb = mean_absolute_error(y_test, gb.predict(X_test))
print(f"\nGradient Boosting MAE: {mae_gb:.3f} s")

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
    print(f"XGBoost MAE:           {mae_xgb:.3f} s")
    print(f"Ensemble MAE:          {mae_ens:.3f} s")
    mae = mae_ens
else:
    mae = mae_gb

# ══════════════════════════════════════════════════════════
# 10. PREDICT
# ══════════════════════════════════════════════════════════
data = qualifying_2026.copy()
data["GB (s)"] = gb.predict(X)
if HAS_XGB:
    data["XGB (s)"] = xgb.predict(X)
    data["PredictedLapTime (s)"] = (data["GB (s)"] + data["XGB (s)"]) / 2
else:
    data["PredictedLapTime (s)"] = data["GB (s)"]

data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.02
data["PredictedLapTime (s)"] += data["GridPenalty (s)"]          # real penalties only
data["WetBonus"] = (1 - data["WetPerformanceFactor"]) * RAIN_PROBABILITY * WET_BONUS_SCALE
data["PredictedLapTime (s)"] -= data["WetBonus"]
data["PredictedLapTime (s)"] -= data["PoleWetBonus"]

data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

medals = {1: "1st", 2: "2nd", 3: "3rd"}
print("\n" + "=" * 62)
print("  2026 SAO PAULO GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos = int(row["PredictedPosition"])
    mark = "  *" if row["PlacesLost"] > 0 else ""
    print(f"  {medals.get(pos, f'P{pos}'):<5} {row['Driver']:<22} "
          f"{row['Team']:<18} {row['PredictedLapTime (s)']:>12.3f}{mark}")
print("=" * 62)
print("  * = driver with a grid penalty")
print(f"\n  Race: {RACE_TEMP}C | Rain: {int(RAIN_PROBABILITY*100)}% | "
      f"Humidity: {HUMIDITY}% | Wind: {WIND_SPEED} km/h")
print("  Pole: Lando Norris | Verstappen, Hamilton, Perez not modelled.\n")

# ══════════════════════════════════════════════════════════
# 11. PICTURE (saved as a PNG — open the file, no pop-up window)
#     The track map panel only appears if saopaulo_track_map.jpeg
#     is next to this script. No placeholder text is drawn.
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"
driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

script_dir = os.path.dirname(os.path.abspath(__file__))
track_img = None
for name in ("saopaulo_track_map.jpeg", "saopaulo_track_map.jpg", "saopaulo_track_map.png"):
    path = os.path.join(script_dir, name)
    if os.path.exists(path):
        track_img = plt.imread(path)
        break
HAS_MAP = track_img is not None
if not HAS_MAP:
    print("No track map file found -> picture will be made without a map.")

TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 13, 10, 10

n_rows = 4 if HAS_MAP else 3
height_ratios = ([1.0] if HAS_MAP else []) + [1.15, 1.0, 1.0]
fig = plt.figure(figsize=(16, 7 * n_rows + 2), facecolor="#0f0f0f")
fig.suptitle(
    "F1 2026 — ROUND 20: SAO PAULO GP\n"
    "INTERLAGOS  |  NOV 8, 2026  |  16°C  |  91% HUMIDITY",
    fontsize=16, fontweight="bold", color="white", fontfamily=FONT, y=0.985
)
gs = GridSpec(n_rows, 2, figure=fig, height_ratios=height_ratios,
              hspace=0.75, wspace=0.50,
              left=0.17, right=0.96, top=0.935, bottom=0.05)

def style_axis(ax):
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white", labelsize=TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

r = 0
if HAS_MAP:
    ax_map = fig.add_subplot(gs[0, :])
    ax_map.set_facecolor("#0f0f0f")
    ax_map.axis("off")
    ax_map.imshow(track_img, aspect="equal")
    r = 1

# Predicted order (zoomed axis so the differences are visible)
ax1 = fig.add_subplot(gs[r, :])
vals = data["PredictedLapTime (s)"]
labels_y = [f"{d} *" if p > 0 else d
            for d, p in zip(data["Driver"], data["PlacesLost"])]
ax1.barh(labels_y[::-1], vals[::-1], color=driver_colors[::-1],
         edgecolor="white", linewidth=0.4, height=0.65)
ax1.set_xlim(vals.min() - 0.3, vals.max() + 0.6)
ax1.set_title("Predicted Race Finishing Order   (* = grid penalty)",
              fontsize=TITLE_SIZE, fontweight="bold", color="white",
              fontfamily=FONT, pad=16)
ax1.set_xlabel("Predicted Avg Lap Time (s) — lower = faster",
               color="#AAAAAA", fontsize=LABEL_SIZE, fontfamily=FONT, labelpad=10)
style_axis(ax1)
for i, v in enumerate(vals[::-1]):
    ax1.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9,
             color="white", fontfamily=FONT)
seen, patches = set(), []
for _, row in data.iterrows():
    if row["Team"] not in seen:
        seen.add(row["Team"])
        patches.append(mpatches.Patch(color=TEAM_COLORS.get(row["Team"], "#FFF"),
                                      label=row["Team"]))
ax1.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.22),
           ncol=len(patches), fontsize=9, facecolor="#1a1a1a",
           edgecolor="#444", labelcolor="white")

# Qualifying gap
ax2 = fig.add_subplot(gs[r + 1, 0])
qs = qualifying_2026.sort_values("GapFromPole (s)")
ax2.barh(qs["Driver"][::-1], qs["GapFromPole (s)"][::-1],
         color=[TEAM_COLORS.get(t, "#FFF") for t in qs["Team"]][::-1],
         edgecolor="white", linewidth=0.4, height=0.65)
ax2.set_title("Qualifying Gap to Pole (wet)", fontsize=TITLE_SIZE, fontweight="bold",
              color="white", fontfamily=FONT, pad=16)
ax2.set_xlabel("Gap (s)", color="#AAAAAA", fontsize=LABEL_SIZE,
               fontfamily=FONT, labelpad=10)
ax2.set_xlim(0, qs["GapFromPole (s)"].max() * 1.30)
style_axis(ax2)
for i, (_, row) in enumerate(qs[::-1].iterrows()):
    gap = row["GapFromPole (s)"]
    ax2.text(gap + 0.02, i, f"+{gap:.3f}s", va="center",
             fontsize=9, color="white", fontfamily=FONT)

# Feature importance — top 8 only, readable names
ax3 = fig.add_subplot(gs[r + 1, 1])
imp = gb.feature_importances_ / gb.feature_importances_.sum()
title = "Top Features (Gradient Boosting)"
if HAS_XGB:
    imp_x = xgb.feature_importances_ / xgb.feature_importances_.sum()
    imp = (imp + imp_x) / 2
    title = "Top Features (GB + XGBoost)"

def nice(name):
    name = name.replace(" (s)", "")
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)

TOP_N = 8
top_idx = np.argsort(imp)[-TOP_N:]
ax3.barh([nice(FEATURE_COLS[i]) for i in top_idx], imp[top_idx],
         color=plt.cm.Blues(np.linspace(0.45, 0.95, TOP_N)),
         edgecolor="white", linewidth=0.3, height=0.6)
ax3.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", color="white",
              fontfamily=FONT, pad=16)
ax3.set_xlabel("Importance", color="#AAAAAA", fontsize=LABEL_SIZE,
               fontfamily=FONT, labelpad=10)
ax3.set_xlim(0, imp[top_idx].max() * 1.20)
style_axis(ax3)
for i, idx in enumerate(top_idx):
    ax3.text(imp[idx] + 0.005, i, f"{imp[idx]:.2f}", va="center",
             fontsize=9, color="white", fontfamily=FONT)

# Podium
ax4 = fig.add_subplot(gs[r + 2, 0])
ax4.set_facecolor("#1a1a1a")
ax4.axis("off")
ax4.set_title("Predicted Podium", fontsize=TITLE_SIZE, fontweight="bold",
              color="white", fontfamily=FONT, pad=16)
podium = data[data["PredictedPosition"] <= 3].sort_values("PredictedPosition")
for i, (_, row) in enumerate(podium.iterrows()):
    y0 = [0.78, 0.46, 0.14][i]
    ax4.text(0.5, y0 + 0.12, ["1st", "2nd", "3rd"][i], ha="center", va="center",
             fontsize=14, color="white", fontfamily=FONT, transform=ax4.transAxes)
    ax4.text(0.5, y0, row["Driver"], ha="center", va="center", fontsize=12,
             fontweight="bold", color=TEAM_COLORS.get(row["Team"], "#FFF"),
             fontfamily=FONT, transform=ax4.transAxes)
    ax4.text(0.5, y0 - 0.08, row["Team"], ha="center", va="center", fontsize=10,
             color="#AAAAAA", fontfamily=FONT, transform=ax4.transAxes)

# Model comparison — one text block so nothing overlaps
ax5 = fig.add_subplot(gs[r + 2, 1])
ax5.set_facecolor("#1a1a1a")
ax5.axis("off")
ax5.set_title("Model Comparison (top 5)", fontsize=TITLE_SIZE,
              fontweight="bold", color="white", fontfamily=FONT, pad=16)
if HAS_XGB:
    gb_top  = data.sort_values("GB (s)")["Driver"].head(5).tolist()
    xgb_top = data.sort_values("XGB (s)")["Driver"].head(5).tolist()
    ens_top = data["Driver"].head(5).tolist()
    lines = [f"{'#':<3}{'GradBoost':<13}{'XGBoost':<13}{'Final':<12}"]
    for i in range(5):
        lines.append(f"{i+1:<3}{gb_top[i].split()[-1]:<13}"
                     f"{xgb_top[i].split()[-1]:<13}{ens_top[i].split()[-1]:<12}")
    lines += ["", f"MAE  GradBoost {mae_gb:.3f}s",
              f"     XGBoost   {mae_xgb:.3f}s",
              f"     Average   {mae_ens:.3f}s"]
    ax5.text(0.06, 0.90, "\n".join(lines), va="top", ha="left", fontsize=10,
             color="white", fontfamily=FONT, linespacing=1.9,
             transform=ax5.transAxes)
else:
    ax5.text(0.5, 0.5, "xgboost not installed\npip install xgboost",
             ha="center", va="center", color="#888888", fontsize=11,
             fontfamily=FONT, transform=ax5.transAxes)

fig.text(0.5, 0.015,
         f"MAE: {mae:.3f}s  |  Race: {RACE_TEMP}°C  |  Rain: {int(RAIN_PROBABILITY*100)}%  |  "
         f"Pole: Norris  |  Penalties: RUS -5, LEC -10  |  VER, HAM, PER not modelled",
         ha="center", fontsize=9, color="#888888", fontfamily=FONT)

OUT = "round_20_saopaulo_prediction.png"
plt.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="#0f0f0f")
plt.close(fig)
print("Picture saved -> " + os.path.abspath(OUT))
print("Open that PNG file to see the full picture.")