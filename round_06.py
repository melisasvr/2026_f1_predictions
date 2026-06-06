"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 6: MONACO GP
 Circuit de Monaco | Race Date: June 8, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.05 (Monaco slower)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            MonacoGridPenalty, ReliabilityRiskScore,
            CircuitScore, SprintWinnerBoost,
            MonacoHistoryScore
 Upgrades vs R05:
            + MonacoGridPenalty — strongest grid penalty
              of any circuit (overtaking nearly impossible)
            + MonacoHistoryScore — driver historical
              performance at Monaco specifically
            + No sprint this weekend
            + Dry race — PoleWetBonus minimal (low rain)
            + 5 rounds of 2026 CircuitScore data
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 6: MONACO GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER — Dry sunny weekend
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 22    # °C — sunny
RACE_TEMP        = 23    # °C — sunny/cloudy
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # +1°C minimal
RAIN_PROBABILITY = 0.05  # ~5% — essentially dry
HUMIDITY         = 55    # % estimated
WIND_SPEED       = 8     # km/h — light winds Monaco harbour

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  Race: {RACE_TEMP}°C ⛅  (Δ+{TEMP_DELTA}°C)")
print(f"🌤️  Rain: {int(RAIN_PROBABILITY*100)}% (dry race expected)")
print(f"🏰  MONACO — Qualifying order = Race order. Pole is EVERYTHING.")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Antonelli — 5th pole in 6 races 🌟 (0.043s over VER!)
# ══════════════════════════════════════════════════════════
POLE_TIME = 72.051  # Antonelli 1:12.051

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Kimi Antonelli",
        "Max Verstappen",
        "Lewis Hamilton",
        "Charles Leclerc",
        "Isack Hadjar",
        "George Russell",
        "Oscar Piastri",
        "Lando Norris",
        "Pierre Gasly",
        "Liam Lawson",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        72.051,   # 1:12.051 — Antonelli POLE 🌟
        72.094,   # 1:12.094 — Verstappen  +0.043s
        72.279,   # 1:12.279 — Hamilton    +0.228s
        72.351,   # 1:12.351 — Leclerc     +0.300s
        72.434,   # 1:12.434 — Hadjar      +0.383s
        72.445,   # 1:12.445 — Russell     +0.394s
        72.624,   # 1:12.624 — Piastri     +0.573s
        72.765,   # 1:12.765 — Norris      +0.714s
        73.226,   # 1:13.226 — Gasly       +1.175s
        73.412,   # 1:13.412 — Lawson      +1.361s
    ],
    "Team": [
        "Mercedes", "Red Bull Racing", "Ferrari",
        "Ferrari",  "Red Bull Racing", "Mercedes",
        "McLaren",  "McLaren",         "Alpine",
        "Racing Bulls",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "SprintWinnerBoost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Kimi Antonelli":  "ANT",
    "Max Verstappen":  "VER",
    "Lewis Hamilton":  "HAM",
    "Charles Leclerc": "LEC",
    "Isack Hadjar":    "HAD",
    "George Russell":  "RUS",
    "Oscar Piastri":   "PIA",
    "Lando Norris":    "NOR",
    "Pierre Gasly":    "GAS",
    "Liam Lawson":     "LAW",
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
    "Red Bull Racing": "#3671C6",
    "Racing Bulls":    "#6692FF",
    "Alpine":          "#FF87BC",
    "Aston Martin":    "#358C75",
    "Williams":        "#64C4FF",
    "Haas":            "#B6BABD",
    "Audi":            "#B8B8B8",
    "Cadillac":        "#C8102E",
}

# ══════════════════════════════════════════════════════════
# 4. ADJUSTED TEAM SCORE
#    Updated after 5 rounds — Antonelli dominance confirmed
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.8,  # 4 wins from 5 races — dominant
    "Ferrari":         8.0,  # Consistent podiums — Hamilton P2 Canada
    "Red Bull Racing": 7.5,  # VER improving every race — P3 Canada
    "McLaren":         7.5,  # Strong race pace when reliable
    "Alpine":          5.5,  # Gasly consistently in points
    "Racing Bulls":    5.0,  # Hadjar P5 Canada — improving
    "Haas":            4.5,
    "Aston Martin":    4.0,
    "Williams":        4.5,  # Colapinto scoring points
    "Audi":            3.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR
#    Dry race — minimal impact but kept for consistency
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "ANT": 0.972,
    "VER": 0.968,
    "HAM": 0.964,
    "LEC": 0.974,
    "HAD": 0.980,
    "RUS": 0.966,
    "PIA": 0.975,
    "NOR": 0.976,
    "GAS": 0.977,
    "LAW": 0.979,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — minimal at 5% rain
#    0.20 * 0.05 = 0.01s — essentially zero effect
# ══════════════════════════════════════════════════════════
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY if (
        p == 1 and RAIN_PROBABILITY >= 0.60
    ) else 0.0
)

# ══════════════════════════════════════════════════════════
# 7. MONACO GRID PENALTY — NEW FEATURE 🆕
#    Monaco is THE hardest circuit to overtake on
#    Grid position matters more here than anywhere else
#    P1 = 0 penalty (controls the race)
#    P2+ = increasing penalty per position
#    Scale: each position back = +0.15s race pace penalty
#    This is stronger than Suzuka (0.05s) and Miami (0.08s)
# ══════════════════════════════════════════════════════════
qualifying_2026["MonacoGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.15
)

# ══════════════════════════════════════════════════════════
# 8. MONACO HISTORY SCORE — NEW FEATURE 🆕
#    Driver-specific Monaco performance history
#    Lower = better Monaco performer
#    Based on historical Monaco results and DNFs
# ══════════════════════════════════════════════════════════
MONACO_HISTORY = {
    "ANT": 2.0,   # Limited Monaco F1 history — strong junior record
    "VER": 2.5,   # Won Monaco 2023 — but has Monaco incident history
    "HAM": 2.0,   # Won Monaco multiple times — loves this circuit
    "LEC": 1.5,   # LOVES Monaco — home race, multiple poles, heartbreaks
    "HAD": 4.0,   # Limited Monaco F1 data
    "RUS": 3.0,   # Decent Monaco record
    "PIA": 3.5,   # Limited Monaco F1 data
    "NOR": 3.0,   # Strong Monaco pace historically
    "GAS": 2.5,   # Good Monaco performer — home race feeling
    "LAW": 4.5,   # Limited Monaco experience
}
qualifying_2026["MonacoHistoryScore"] = qualifying_2026["DriverCode"].map(
    MONACO_HISTORY
)

# ══════════════════════════════════════════════════════════
# 9. ERS DEPENDENCY (7MJ limit continues)
# ══════════════════════════════════════════════════════════
ERS_DEPENDENCY = {
    "Mercedes":        9.0,
    "McLaren":         9.0,
    "Ferrari":         6.5,
    "Red Bull Racing": 5.5,
    "Alpine":          6.5,
    "Racing Bulls":    5.5,
    "Haas":            6.5,
    "Aston Martin":    9.0,
    "Williams":        9.0,
    "Audi":            7.0,
    "Cadillac":        6.5,
}
qualifying_2026["ERSDependencyScore"] = qualifying_2026["Team"].map(
    ERS_DEPENDENCY
)

# ══════════════════════════════════════════════════════════
# 10. RELIABILITY RISK (updated after 5 rounds)
#     Russell DNF Canada — Mercedes reliability dipped
# ══════════════════════════════════════════════════════════
RELIABILITY_RISK = {
    "Mercedes":        3.0,  # Russell DNF Canada — upgraded risk
    "Ferrari":         2.0,
    "Red Bull Racing": 3.0,
    "McLaren":         3.0,
    "Alpine":          4.0,
    "Racing Bulls":    3.5,
    "Haas":            3.0,
    "Aston Martin":    3.5,
    "Williams":        4.0,
    "Audi":            5.5,
    "Cadillac":        5.0,
}
qualifying_2026["ReliabilityRiskScore"] = qualifying_2026["Team"].map(
    RELIABILITY_RISK
)

# ══════════════════════════════════════════════════════════
# 11. CIRCUIT SCORE — 5 ROUNDS OF 2026 DATA
#     AUS + CHN + JPN + MIA + CAN
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS   CHN   JPN   MIA   CAN
    "ANT":              [2,    1,    1,    1,    1],
    "VER":              [20,   20,   8,    5,    3],
    "HAM":              [7,    3,    6,    7,    2],
    "LEC":              [3,    4,    3,    6,    4],
    "HAD":              [20,   8,    9,    20,   5],
    "RUS":              [1,    2,    4,    4,    20],  # Canada DNF
    "PIA":              [22,   2,    2,    3,    20],  # Canada DNF
    "NOR":              [5,    20,   5,    2,    20],  # Canada DNF
    "GAS":              [10,   6,    10,   20,   8],
    "LAW":              [20,   20,   20,   20,   7],
}

circuit_scores = {}
for code, results in RESULTS_2026.items():
    avg = np.mean(results)
    normalized = 1 + (avg - 1) * (4 / 19)
    circuit_scores[code] = round(normalized, 3)

qualifying_2026["CircuitScore"] = qualifying_2026["DriverCode"].map(
    circuit_scores
).fillna(3.5)

# ══════════════════════════════════════════════════════════
# 12. SYNTHETIC SECTOR TIMES
#     Monaco split ratios (approximate)
#     S1: 35% — Sainte Devote to Casino
#     S2: 38% — Casino to Tunnel exit
#     S3: 27% — Swimming pool to Rascasse to finish
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.35
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.38
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.27
# Monaco: race pace only ~5% slower than qualifying (very slow circuit)
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.05

# ══════════════════════════════════════════════════════════
# 13. WEATHER FEATURES
# ══════════════════════════════════════════════════════════
qualifying_2026["RainProbability"] = RAIN_PROBABILITY
qualifying_2026["Temperature"]     = RACE_TEMP
qualifying_2026["TempDelta"]       = TEMP_DELTA
qualifying_2026["Humidity"]        = HUMIDITY
qualifying_2026["WindSpeed"]       = WIND_SPEED

print("\n📊 Full Feature Set:")
print(qualifying_2026[[
    "Driver", "QualifyingTime (s)", "GapFromPole (s)",
    "AdjustedTeamScore", "MonacoGridPenalty",
    "MonacoHistoryScore", "CircuitScore"
]].to_string(index=False))

# ══════════════════════════════════════════════════════════
# 14. FEATURE COLUMNS
# ══════════════════════════════════════════════════════════
FEATURE_COLS = [
    "QualifyingTime (s)",
    "GapFromPole (s)",
    "AdjustedTeamScore",
    "GridPenalty (s)",
    "WetPerformanceFactor",
    "PoleWetBonus",           # minimal — dry race
    "RainProbability",        # 5% — dry
    "Temperature",
    "TempDelta",
    "Humidity",
    "WindSpeed",
    "ERSDependencyScore",
    "MonacoGridPenalty",      # 🆕 strongest grid penalty of the season
    "MonacoHistoryScore",     # 🆕 driver Monaco history
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",           # 5 rounds of 2026 data
    "SprintWinnerBoost",      # no sprint this weekend
]
TARGET = "RacePace (s)"

# ══════════════════════════════════════════════════════════
# 15. TRAIN MODEL
# ══════════════════════════════════════════════════════════
X = qualifying_2026[FEATURE_COLS].fillna(0)
y = qualifying_2026[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"\n🔍 Model MAE on test set: {mae:.2f} seconds")

# ══════════════════════════════════════════════════════════
# 16. PREDICT RACE
# ══════════════════════════════════════════════════════════
data = qualifying_2026.copy()
data["PredictedLapTime (s)"] = model.predict(X)

# Monaco grid penalty — applied directly to predicted lap time
# P1 stays P1, everyone else penalised by grid position
data["PredictedLapTime (s)"] += data["MonacoGridPenalty"] * 0.5

# Monaco history bonus — great Monaco drivers get advantage
data["PredictedLapTime (s)"] += (data["MonacoHistoryScore"] - 1) * 0.05

# Wet bonus — minimal at 5% rain
data["WetBonus"] = (
    (1 - data["WetPerformanceFactor"]) * RAIN_PROBABILITY * 100
)
data["PredictedLapTime (s)"] -= data["WetBonus"]

# Pole wet bonus — essentially zero at 5% rain (won't activate <60%)
data["PredictedLapTime (s)"] -= data["PoleWetBonus"]

# Sort by fastest predicted lap time
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 17. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 MONACO GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  Race: {RACE_TEMP}°C ⛅")
print(f"  🌤️  Rain: {int(RAIN_PROBABILITY*100)}% — dry race expected")
print(f"  🔋  ERS limit: 7MJ")
print(f"  ⚡  Boost cap: +150kW")
print(f"  🏰  Monaco — qualifying order likely = race order")
print(f"  🌟  Pole: Antonelli — 5th pole in 6 races!\n")

# ══════════════════════════════════════════════════════════
# 18. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 6: MONACO GP\n"
    "CIRCUIT DE MONACO  |  JUNE 8, 2026  |  ☀️ DRY",
    fontsize=18, fontweight="bold", color="white",
    fontfamily=FONT, y=0.98
)
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Chart 1: Predicted Race Finishing Order ───────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.barh(
    data["Driver"][::-1],
    data["PredictedLapTime (s)"][::-1],
    color=driver_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.7
)
ax1.set_title(
    "📊 Predicted Race Finishing Order  (🏰 Monaco — Grid = Destiny)",
    fontsize=13, fontweight="bold", color="white",
    fontfamily=FONT, pad=12
)
ax1.set_xlabel("Predicted Avg Lap Time (s) — lower = faster",
               color="#AAAAAA", fontsize=9, fontfamily=FONT)
ax1.tick_params(colors="white", labelsize=9)
ax1.set_facecolor("#1a1a1a")
for spine in ax1.spines.values():
    spine.set_edgecolor("#333333")
for i, (_, row) in enumerate(data[::-1].iterrows()):
    pos   = int(row["PredictedPosition"])
    label = medals.get(pos, f"P{pos}")
    ax1.text(
        data["PredictedLapTime (s)"].min() * 0.9997, i, label,
        va="center", ha="right", fontsize=9,
        color="white", fontfamily=FONT, fontweight="bold"
    )
seen = set()
legend_patches = []
for _, row in data.iterrows():
    t = row["Team"]
    if t not in seen:
        seen.add(t)
        legend_patches.append(
            mpatches.Patch(color=TEAM_COLORS.get(t, "#FFF"), label=t)
        )
ax1.legend(handles=legend_patches, loc="lower right",
           fontsize=8, facecolor="#1a1a1a",
           edgecolor="#444", labelcolor="white")

# ── Chart 2: Monaco Grid Penalty ─────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
grid_colors = [TEAM_COLORS.get(t, "#FFF") for t in qualifying_2026["Team"]]
ax2.barh(
    qualifying_2026["Driver"][::-1],
    qualifying_2026["MonacoGridPenalty"][::-1],
    color=grid_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title(
    "🏰 Monaco Grid Penalty\n(higher = harder to recover from grid position)",
    fontsize=10, fontweight="bold", color="white",
    fontfamily=FONT, pad=10
)
ax2.set_xlabel("Grid Penalty (s)", color="#AAAAAA",
               fontsize=8, fontfamily=FONT)
ax2.tick_params(colors="white", labelsize=8)
ax2.set_facecolor("#1a1a1a")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333333")

# ── Chart 3: Qualifying Gap to Pole ──────────────────────
ax3 = fig.add_subplot(gs[1, 1])
qual_sorted = qualifying_2026.sort_values("GapFromPole (s)")
qual_colors = [TEAM_COLORS.get(t, "#FFF") for t in qual_sorted["Team"]]
ax3.barh(
    qual_sorted["Driver"][::-1],
    qual_sorted["GapFromPole (s)"][::-1],
    color=qual_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax3.set_title("⏱️  Qualifying Gap to Pole (Real Q3 Times)",
              fontsize=11, fontweight="bold", color="white",
              fontfamily=FONT, pad=10)
ax3.set_xlabel("Gap to Pole (seconds)", color="#AAAAAA",
               fontsize=9, fontfamily=FONT)
ax3.tick_params(colors="white", labelsize=8)
ax3.set_facecolor("#1a1a1a")
for spine in ax3.spines.values():
    spine.set_edgecolor("#333333")
for i, (_, row) in enumerate(qual_sorted[::-1].iterrows()):
    ax3.text(
        row["GapFromPole (s)"] + 0.005, i,
        f"+{row['GapFromPole (s)']:.3f}s",
        va="center", fontsize=7.5,
        color="white", fontfamily=FONT
    )

# ── Chart 4: Feature Importance ──────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
feat_labels = [
    "Qualifying Time", "Gap From Pole", "Team Score",
    "Grid Penalty", "Wet Factor", "Pole Wet Bonus",
    "Rain Prob", "Temperature", "Temp Delta",
    "Humidity", "Wind Speed", "ERS Dependency",
    "Monaco Grid 🆕", "Monaco History 🆕",
    "Reliability", "Sector 1", "Sector 2", "Sector 3",
    "Circuit Score", "Sprint Boost"
]
feat_import   = model.feature_importances_
sorted_idx    = np.argsort(feat_import)
sorted_labels = [feat_labels[i] for i in sorted_idx]
sorted_values = feat_import[sorted_idx]
colors_bar    = plt.cm.Reds(np.linspace(0.3, 0.95, len(sorted_values)))
ax4.barh(sorted_labels, sorted_values,
         color=colors_bar,
         edgecolor="white", linewidth=0.3, height=0.6)
ax4.set_title("🤖 Model Feature Importance",
              fontsize=11, fontweight="bold", color="white",
              fontfamily=FONT, pad=10)
ax4.set_xlabel("Importance Score", color="#AAAAAA",
               fontsize=9, fontfamily=FONT)
ax4.tick_params(colors="white", labelsize=7)
ax4.set_facecolor("#1a1a1a")
for spine in ax4.spines.values():
    spine.set_edgecolor("#333333")
for i, v in enumerate(sorted_values):
    ax4.text(v + 0.001, i, f"{v:.3f}",
             va="center", fontsize=7,
             color="white", fontfamily=FONT)

# ── Chart 5: Predicted Podium ─────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor("#1a1a1a")
ax5.axis("off")
for spine in ax5.spines.values():
    spine.set_edgecolor("#333333")

podium      = data[data["PredictedPosition"] <= 3].sort_values(
    "PredictedPosition"
)
podium_y    = [0.75, 0.47, 0.19]
podium_icon = ["🥇", "🥈", "🥉"]
podium_size = [22, 18, 16]
ax5.set_title("🏆 Predicted Podium  🏰",
              fontsize=13, fontweight="bold", color="white",
              fontfamily=FONT, pad=12)
for i, (_, row) in enumerate(podium.iterrows()):
    color = TEAM_COLORS.get(row["Team"], "#FFFFFF")
    ax5.text(0.5, podium_y[i] + 0.08, podium_icon[i],
             ha="center", va="center",
             fontsize=podium_size[i],
             transform=ax5.transAxes)
    ax5.text(0.5, podium_y[i], row["Driver"],
             ha="center", va="center",
             fontsize=12, fontweight="bold",
             color=color, fontfamily=FONT,
             transform=ax5.transAxes)
    ax5.text(0.5, podium_y[i] - 0.08, row["Team"],
             ha="center", va="center",
             fontsize=9, color="#AAAAAA",
             fontfamily=FONT,
             transform=ax5.transAxes)

# Footer
fig.text(
    0.5, 0.01,
    f"🔍 MAE: {mae:.2f}s  |  "
    f"☀️ Dry race {RACE_TEMP}°C  |  "
    f"🔋 ERS: 7MJ  |  "
    f"🏰 MonacoGridPenalty: +0.15s/pos  |  "
    f"🌟 Pole: Antonelli (5th in 6 races!)",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_06_monaco_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_06_monaco_prediction.png")
plt.show()