"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 17: SINGAPORE GP
 Marina Bay Street Circuit | Race Date: October 11, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.10
 Notes    : NO grid-position penalty feature (removed by request)
            NO rookie penalty
            Sprint weekend: Verstappen won the Sprint
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            TyreDegScore, TropicalHeatScore, SafetyCarSkill,
            ReliabilityRiskScore, CircuitScore,
            SprintWinnerBoost, HomeRaceBoost
 Author   : Melisa Sever
============================================================
"""

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

print("=" * 62)
print("  F1 PREDICTIONS 2026 — ROUND 17: SINGAPORE GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER  (qualifying Sat: 31°C / race Sun: 31°C)
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 31
RACE_TEMP        = 31
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # 0°C
RAIN_PROBABILITY = 0.30    # 30% race day (below the 60% PoleWetBonus threshold)
HUMIDITY         = 74      # % race day
WIND_SPEED       = 10      # km/h

print(f"\nQualifying: {QUALIFYING_TEMP}C  ->  Race: {RACE_TEMP}C  (delta {TEMP_DELTA}C)")
print(f"Rain: {int(RAIN_PROBABILITY*100)}%  |  Humidity: {HUMIDITY}%  |  Wind: {WIND_SPEED} km/h")

# ══════════════════════════════════════════════════════════
# 2. GP Q3 QUALIFYING DATA (from the official timing screen)
#    Gaps converted to absolute seconds from pole 1:31.373
# ══════════════════════════════════════════════════════════
POLE_TIME = 91.373

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Max Verstappen", "Charles Leclerc", "Lewis Hamilton",
        "Kimi Antonelli", "Lando Norris", "George Russell",
        "Oscar Piastri", "Pierre Gasly", "Liam Lawson",
        "Arvid Lindblad",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        91.373,   # Verstappen  POLE
        91.427,   # +0.054  Leclerc
        91.496,   # +0.123  Hamilton
        91.562,   # +0.189  Antonelli
        91.712,   # +0.339  Norris
        91.747,   # +0.374  Russell
        91.860,   # +0.487  Piastri
        92.225,   # +0.852  Gasly
        92.369,   # +0.996  Lawson
        92.781,   # +1.408  Lindblad
    ],
    # NOTE: Lawson is listed as Racing Bulls here (his timing-screen colour
    # matches Lindblad). Change to "Red Bull Racing" if that is wrong.
    "Team": [
        "Red Bull Racing", "Ferrari",  "Ferrari",
        "Mercedes",        "McLaren",  "Mercedes",
        "McLaren",         "Alpine",   "Racing Bulls",
        "Racing Bulls",
    ],
    "GridPenalty (s)":   [0] * 10,    # real grid penalties only (none this weekend)
    # Verstappen won the Sprint
    "SprintWinnerBoost": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Max Verstappen":  "VER", "Charles Leclerc": "LEC",
    "Lewis Hamilton":  "HAM", "Kimi Antonelli":  "ANT",
    "Lando Norris":    "NOR", "George Russell":  "RUS",
    "Oscar Piastri":   "PIA", "Pierre Gasly":    "GAS",
    "Liam Lawson":     "LAW", "Arvid Lindblad":  "LIN",
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
# 4. ADJUSTED TEAM SCORE — updated after 16 rounds
#    (hand-set estimates, higher = stronger)
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.3,
    "McLaren":         9.0,
    "Ferrari":         9.0,
    "Red Bull Racing": 8.8,   # Verstappen: Malaysia win, Sprint win + pole here
    "Alpine":          7.0,
    "Racing Bulls":    5.8,
    "Audi":            6.0,
    "Williams":        5.5,
    "Haas":            5.0,
    "Aston Martin":    4.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(ADJUSTED_TEAM_SCORE)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR (lower = better in the wet)
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "VER": 0.966, "LEC": 0.974, "HAM": 0.964, "ANT": 0.972,
    "NOR": 0.976, "RUS": 0.966, "PIA": 0.975, "GAS": 0.977,
    "LAW": 0.979, "LIN": 0.983,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(WET_PERFORMANCE)

# PoleWetBonus only activates above 60% rain (30% here -> stays 0)
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY
    if (p == 1 and RAIN_PROBABILITY >= 0.60) else 0.0
)

# ══════════════════════════════════════════════════════════
# 6. TROPICAL HEAT SCORE (lower = better in heat + humidity)
#    Singapore is as humid as Sepang (74%)
# ══════════════════════════════════════════════════════════
TROPICAL_HEAT = {
    "VER": 2.0, "LEC": 2.0, "HAM": 1.5, "ANT": 2.0, "NOR": 2.0,
    "RUS": 2.0, "PIA": 2.0, "GAS": 2.5, "LAW": 2.5, "LIN": 2.5,
}
qualifying_2026["TropicalHeatScore"] = qualifying_2026["DriverCode"].map(TROPICAL_HEAT)

# ══════════════════════════════════════════════════════════
# 7. SAFETY CAR SKILL (estimates; Marina Bay almost always has one)
# ══════════════════════════════════════════════════════════
SAFETY_CAR_SKILL = {
    "VER": 0.90, "LEC": 0.90, "HAM": 0.92, "ANT": 0.82, "NOR": 0.85,
    "RUS": 0.85, "PIA": 0.80, "GAS": 0.82, "LAW": 0.78, "LIN": 0.75,
}
qualifying_2026["SafetyCarSkill"] = qualifying_2026["DriverCode"].map(SAFETY_CAR_SKILL)

# ══════════════════════════════════════════════════════════
# 8. TYRE DEG / ERS / RELIABILITY (team-level estimates)
# ══════════════════════════════════════════════════════════
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
qualifying_2026["TyreDegScore"]         = qualifying_2026["Team"].map(TYRE_DEG)
qualifying_2026["ERSDependencyScore"]   = qualifying_2026["Team"].map(ERS_DEPENDENCY)
qualifying_2026["ReliabilityRiskScore"] = qualifying_2026["Team"].map(RELIABILITY_RISK)
qualifying_2026["HomeRaceBoost"]        = 0.0   # no Singaporean drivers

# ══════════════════════════════════════════════════════════
# 9. CIRCUIT SCORE — 16 rounds of 2026 finishing positions
#    (20 = DNF / outside the top 10)
#    AUS CHN JPN MIA CAN MON ESP AUT GBR BEL HUN NED ITA MAD AZE MYS
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    "VER": [20, 20, 8, 5, 3, 20, 5, 2, 20, 3, 2, 20, 3, 2, 2, 1],
    "LEC": [3, 4, 3, 6, 4, 20, 20, 20, 1, 2, 4, 5, 20, 4, 4, 4],
    "HAM": [7, 3, 6, 7, 2, 2, 1, 5, 3, 4, 5, 4, 6, 20, 6, 3],
    "ANT": [2, 1, 1, 1, 1, 1, 4, 3, 20, 1, 3, 2, 1, 1, 5, 2],
    "NOR": [5, 20, 5, 2, 20, 20, 3, 20, 4, 7, 1, 1, 4, 3, 5, 9],
    "RUS": [1, 2, 4, 4, 20, 20, 2, 1, 2, 20, 7, 3, 2, 5, 1, 20],
    "PIA": [22, 2, 2, 3, 20, 4, 20, 4, 20, 5, 20, 6, 5, 8, 20, 6],
    "GAS": [10, 6, 10, 20, 8, 7, 20, 20, 20, 20, 12, 10, 7, 20, 20, 20],
    "LAW": [20, 20, 20, 20, 7, 5, 20, 20, 8, 20, 8, 7, 20, 6, 20, 7],
    "LIN": [8, 20, 14, 20, 6, 6, 20, 20, 7, 9, 10, 10, 8, 9, 7, 10],
}
circuit_scores = {
    code: round(1 + (np.mean(res) - 1) * (4 / 19), 3)
    for code, res in RESULTS_2026.items()
}
qualifying_2026["CircuitScore"] = qualifying_2026["DriverCode"].map(circuit_scores).fillna(3.5)

# ══════════════════════════════════════════════════════════
# 10. SYNTHETIC SECTOR TIMES + RACE PACE TARGET
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.31
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.39
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.30
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.10

qualifying_2026["RainProbability"] = RAIN_PROBABILITY
qualifying_2026["Temperature"]     = RACE_TEMP
qualifying_2026["TempDelta"]       = TEMP_DELTA
qualifying_2026["Humidity"]        = HUMIDITY
qualifying_2026["WindSpeed"]       = WIND_SPEED

print("\nFeature set:")
print(qualifying_2026[[
    "Driver", "QualifyingTime (s)", "GapFromPole (s)", "AdjustedTeamScore",
    "TropicalHeatScore", "SafetyCarSkill", "CircuitScore",
]].to_string(index=False))

# ══════════════════════════════════════════════════════════
# 11. FEATURES (no grid-position penalty feature)
# ══════════════════════════════════════════════════════════
FEATURE_COLS = [
    "QualifyingTime (s)", "GapFromPole (s)", "AdjustedTeamScore",
    "GridPenalty (s)", "WetPerformanceFactor", "PoleWetBonus",
    "RainProbability", "Temperature", "TempDelta", "Humidity", "WindSpeed",
    "ERSDependencyScore", "TyreDegScore", "TropicalHeatScore",
    "SafetyCarSkill", "HomeRaceBoost", "ReliabilityRiskScore",
    "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)",
    "CircuitScore", "SprintWinnerBoost",
]
TARGET = "RacePace (s)"

X = qualifying_2026[FEATURE_COLS].fillna(0)
y = qualifying_2026[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
)
model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"\nModel MAE on test set: {mae:.2f} seconds")

# ══════════════════════════════════════════════════════════
# 12. PREDICT — post-model adjustments (NO grid penalty, NO rookie penalty)
# ══════════════════════════════════════════════════════════
data = qualifying_2026.copy()
data["PredictedLapTime (s)"] = model.predict(X)

data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.02
data["PredictedLapTime (s)"] += data["TropicalHeatScore"] * 0.03
data["PredictedLapTime (s)"] -= data["SafetyCarSkill"] * 0.02

# Wet bonus: same formula as previous rounds
data["WetBonus"] = (1 - data["WetPerformanceFactor"]) * RAIN_PROBABILITY * 100
data["PredictedLapTime (s)"] -= data["WetBonus"]
data["PredictedLapTime (s)"] -= data["PoleWetBonus"]

data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 13. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "1st", 2: "2nd", 3: "3rd"}
print("\n" + "=" * 62)
print("  2026 SINGAPORE GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos = int(row["PredictedPosition"])
    print(f"  {medals.get(pos, f'P{pos}'):<5} {row['Driver']:<22} "
          f"{row['Team']:<18} {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  Race: {RACE_TEMP}C | Rain: {int(RAIN_PROBABILITY*100)}% | "
      f"Humidity: {HUMIDITY}% | Wind: {WIND_SPEED} km/h")
print("  Sprint winner: Max Verstappen | Pole: Max Verstappen")
print("  No grid penalty applied.\n")

# ══════════════════════════════════════════════════════════
# 14. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"
driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 26), facecolor="#0f0f0f")
fig.suptitle(
    "F1 2026 — ROUND 17: SINGAPORE GP\n"
    "MARINA BAY STREET CIRCUIT  |  OCT 11, 2026  |  31°C  |  74% HUMIDITY",
    fontsize=16, fontweight="bold", color="white", fontfamily=FONT, y=0.98
)
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# Chart 1: predicted order
ax1 = fig.add_subplot(gs[0, :])
ax1.barh(data["Driver"][::-1], data["PredictedLapTime (s)"][::-1],
         color=driver_colors[::-1], edgecolor="white", linewidth=0.4, height=0.7)
ax1.set_title("Predicted Race Finishing Order", fontsize=13, fontweight="bold",
              color="white", fontfamily=FONT, pad=12)
ax1.set_xlabel("Predicted Avg Lap Time (s) — lower = faster",
               color="#AAAAAA", fontsize=9, fontfamily=FONT)
ax1.tick_params(colors="white", labelsize=9)
ax1.set_facecolor("#1a1a1a")
for spine in ax1.spines.values():
    spine.set_edgecolor("#333333")
seen, patches = set(), []
for _, row in data.iterrows():
    if row["Team"] not in seen:
        seen.add(row["Team"])
        patches.append(mpatches.Patch(color=TEAM_COLORS.get(row["Team"], "#FFF"),
                                      label=row["Team"]))
ax1.legend(handles=patches, loc="lower right", fontsize=8,
           facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")

# Chart 2: tropical heat
ax2 = fig.add_subplot(gs[1, 0])
hs = data.sort_values("TropicalHeatScore")
ax2.barh(hs["Driver"][::-1], hs["TropicalHeatScore"][::-1],
         color=[TEAM_COLORS.get(t, "#FFF") for t in hs["Team"]][::-1],
         edgecolor="white", linewidth=0.4, height=0.65)
ax2.set_title("Tropical Heat Score\n(lower = better in heat + humidity)",
              fontsize=10, fontweight="bold", color="white", fontfamily=FONT, pad=10)
ax2.tick_params(colors="white", labelsize=8)
ax2.set_facecolor("#1a1a1a")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333333")

# Chart 3: qualifying gap
ax3 = fig.add_subplot(gs[1, 1])
qs = qualifying_2026.sort_values("GapFromPole (s)")
ax3.barh(qs["Driver"][::-1], qs["GapFromPole (s)"][::-1],
         color=[TEAM_COLORS.get(t, "#FFF") for t in qs["Team"]][::-1],
         edgecolor="white", linewidth=0.4, height=0.65)
ax3.set_title("Qualifying Gap to Pole", fontsize=11, fontweight="bold",
              color="white", fontfamily=FONT, pad=10)
ax3.set_xlabel("Gap (s)", color="#AAAAAA", fontsize=9, fontfamily=FONT)
ax3.tick_params(colors="white", labelsize=8)
ax3.set_facecolor("#1a1a1a")
for spine in ax3.spines.values():
    spine.set_edgecolor("#333333")
for i, (_, row) in enumerate(qs[::-1].iterrows()):
    ax3.text(row["GapFromPole (s)"] + 0.01, i, f"+{row['GapFromPole (s)']:.3f}s",
             va="center", fontsize=7.5, color="white", fontfamily=FONT)

# Chart 4: feature importance
ax4 = fig.add_subplot(gs[2, 0])
labels = [c.replace(" (s)", "") for c in FEATURE_COLS]
imp = model.feature_importances_
order = np.argsort(imp)
ax4.barh([labels[i] for i in order], imp[order],
         color=plt.cm.Blues(np.linspace(0.3, 0.95, len(imp))),
         edgecolor="white", linewidth=0.3, height=0.6)
ax4.set_title("Model Feature Importance", fontsize=11, fontweight="bold",
              color="white", fontfamily=FONT, pad=10)
ax4.tick_params(colors="white", labelsize=7)
ax4.set_facecolor("#1a1a1a")
for spine in ax4.spines.values():
    spine.set_edgecolor("#333333")

# Chart 5: podium
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor("#1a1a1a")
ax5.axis("off")
ax5.set_title("Predicted Podium", fontsize=13, fontweight="bold",
              color="white", fontfamily=FONT, pad=12)
podium = data[data["PredictedPosition"] <= 3].sort_values("PredictedPosition")
for i, (_, row) in enumerate(podium.iterrows()):
    y0 = [0.75, 0.47, 0.19][i]
    ax5.text(0.5, y0 + 0.08, ["1st", "2nd", "3rd"][i], ha="center", va="center",
             fontsize=16, color="white", fontfamily=FONT, transform=ax5.transAxes)
    ax5.text(0.5, y0, row["Driver"], ha="center", va="center", fontsize=12,
             fontweight="bold", color=TEAM_COLORS.get(row["Team"], "#FFF"),
             fontfamily=FONT, transform=ax5.transAxes)
    ax5.text(0.5, y0 - 0.08, row["Team"], ha="center", va="center", fontsize=9,
             color="#AAAAAA", fontfamily=FONT, transform=ax5.transAxes)

fig.text(0.5, 0.01,
         f"MAE: {mae:.2f}s  |  Race: {RACE_TEMP}°C  |  Humidity: {HUMIDITY}%  |  "
         f"Rain: {int(RAIN_PROBABILITY*100)}%  |  No grid penalty",
         ha="center", fontsize=8, color="#888888", fontfamily=FONT)

plt.savefig("round_17_singapore_prediction.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
print("Chart saved -> round_17_singapore_prediction.png")
plt.show()