"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 16: BAHRAIN GP (MALAYSIA)
 Sepang International Circuit | Race Date: October 4, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.08 (tropical circuit)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            SepangGridPenalty, TyreDegScore,
            TropicalHeatScore, ReliabilityRiskScore,
            CircuitScore, SprintWinnerBoost,
            HomeRaceBoost
 Upgrades vs R15:
            + Verstappen POLE — Red Bull 1-3!
            + TropicalHeatScore — 32°C + high humidity
            + 30% rain race day — PoleWetBonus may activate
            + Hadjar P3 — Red Bull surging
            + Bortoleto P10 — Audi in Q3
            + 15 rounds of 2026 CircuitScore data
            + Sepang tropical conditions unique this season
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 16: BAHRAIN GP (MALAYSIA)")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER — Tropical Sepang conditions
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 32    # °C — hot tropical qualifying
RACE_TEMP        = 32    # °C — hot tropical race
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # 0°C
RAIN_PROBABILITY = 0.30  # 30% race day — could trigger PoleWetBonus
HUMIDITY         = 75    # % estimated — tropical Sepang
WIND_SPEED       = 10    # km/h

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ☁️  →  Race: {RACE_TEMP}°C ⛅  (Δ{TEMP_DELTA}°C)")
print(f"🌧️  Rain: {int(RAIN_PROBABILITY*100)}% race day — tropical showers possible")
print(f"💧  Humidity: {HUMIDITY}% — tropical Sepang conditions!")
print(f"🔵  Verstappen POLE — Red Bull Ford 1-3 in quali!")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Verstappen POLE — Red Bull dominant! 🔵
#    Hamilton P2 — Ferrari strong in heat
#    Hadjar P3 — Red Bull Ford 1-3!
#    Russell P8 — Mercedes off pace this weekend
# ══════════════════════════════════════════════════════════
POLE_TIME = 95.130  # Verstappen 1:35.130

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Max Verstappen",
        "Lewis Hamilton",
        "Isack Hadjar",
        "Kimi Antonelli",
        "Charles Leclerc",
        "Lando Norris",
        "Oscar Piastri",
        "George Russell",
        "Pierre Gasly",
        "Gabriel Bortoleto",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        95.130,   # 1:35.130 — Verstappen POLE 🌟
        95.428,   # +0.298s  — Hamilton
        95.558,   # +0.428s  — Hadjar
        95.631,   # +0.501s  — Antonelli
        95.666,   # +0.536s  — Leclerc
        95.757,   # +0.627s  — Norris
        95.762,   # +0.632s  — Piastri
        95.871,   # +0.741s  — Russell
        97.210,   # +2.080s  — Gasly
        97.673,   # +2.543s  — Bortoleto
    ],
    "Team": [
        "Red Bull Racing", "Ferrari",   "Red Bull Racing",
        "Mercedes",        "Ferrari",   "McLaren",
        "McLaren",         "Mercedes",  "Alpine",
        "Audi",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "SprintWinnerBoost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Max Verstappen":   "VER",
    "Lewis Hamilton":   "HAM",
    "Isack Hadjar":     "HAD",
    "Kimi Antonelli":   "ANT",
    "Charles Leclerc":  "LEC",
    "Lando Norris":     "NOR",
    "Oscar Piastri":    "PIA",
    "George Russell":   "RUS",
    "Pierre Gasly":     "GAS",
    "Gabriel Bortoleto":"BOR",
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
# 4. ADJUSTED TEAM SCORE — updated after 15 rounds
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.3,  # Slight dip — Russell P8 quali
    "McLaren":         9.3,  # Norris 3 wins — consistent
    "Ferrari":         9.0,  # Hamilton P2 — strong in heat
    "Red Bull Racing": 8.5,  # VER pole + Hadjar P3 — SURGING!
    "Alpine":          7.0,  # Gasly consistent
    "Audi":            6.0,  # Bortoleto P10 Q3 — improving!
    "Racing Bulls":    5.5,
    "Williams":        5.5,
    "Haas":            5.0,
    "Aston Martin":    4.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR
#    30% rain — meaningful but below 60% PoleWetBonus threshold
#    Still worth factoring in
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "VER": 0.966,   # Legendary wet driver
    "HAM": 0.964,   # All time wet driver
    "HAD": 0.980,
    "ANT": 0.972,
    "LEC": 0.974,
    "NOR": 0.976,
    "PIA": 0.975,
    "RUS": 0.966,   # Elite wet driver
    "GAS": 0.977,
    "BOR": 0.980,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS
#    30% rain < 60% threshold — won't activate
#    But keeping feature for consistency
# ══════════════════════════════════════════════════════════
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY if (
        p == 1 and RAIN_PROBABILITY >= 0.60
    ) else 0.0
)

# ══════════════════════════════════════════════════════════
# 7. SEPANG GRID PENALTY
#    Good overtaking at Sepang — long back straight
#    Medium-easy overtaking: 0.08s per position
# ══════════════════════════════════════════════════════════
qualifying_2026["SepangGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.08
)

# ══════════════════════════════════════════════════════════
# 8. TROPICAL HEAT SCORE — NEW FEATURE 🆕
#    32°C + 75% humidity = most extreme conditions of season
#    Affects driver performance and tyre behaviour
#    Tropical conditions favour teams with better cooling
#    Lower = better adapted to extreme tropical heat
# ══════════════════════════════════════════════════════════
TROPICAL_HEAT = {
    "VER": 2.0,   # Red Bull good in heat
    "HAM": 1.5,   # Has won in Malaysia before — heat experience
    "HAD": 2.5,   # Limited tropical F1 experience
    "ANT": 2.0,   # Mercedes good cooling
    "LEC": 2.0,   # Ferrari good in hot conditions
    "NOR": 2.0,   # McLaren good cooling
    "PIA": 2.0,   # McLaren
    "RUS": 2.0,   # Mercedes
    "GAS": 2.5,   # Renault PU
    "BOR": 3.0,   # Audi — new team, less tropical experience
}
qualifying_2026["TropicalHeatScore"] = qualifying_2026["DriverCode"].map(
    TROPICAL_HEAT
)

# ══════════════════════════════════════════════════════════
# 9. TYRE DEG — 32°C + 75% HUMIDITY = EXTREME
#    Most extreme tyre conditions of 2026 season
# ══════════════════════════════════════════════════════════
TYRE_DEG = {
    "Mercedes":        2.0,
    "Ferrari":         2.0,
    "McLaren":         1.5,  # Best tyre management
    "Red Bull Racing": 2.0,
    "Alpine":          3.0,
    "Audi":            3.5,
    "Racing Bulls":    3.0,
    "Williams":        3.5,
    "Haas":            3.5,
    "Aston Martin":    3.0,
    "Cadillac":        4.5,
}
qualifying_2026["TyreDegScore"] = qualifying_2026["Team"].map(TYRE_DEG)

# ══════════════════════════════════════════════════════════
# 10. ERS DEPENDENCY (7MJ — long back straight at Sepang)
# ══════════════════════════════════════════════════════════
ERS_DEPENDENCY = {
    "Mercedes":        9.0,
    "McLaren":         9.0,
    "Ferrari":         6.5,
    "Red Bull Racing": 5.5,   # Biggest benefit at Sepang
    "Alpine":          6.5,
    "Racing Bulls":    5.5,
    "Williams":        9.0,
    "Audi":            7.0,
    "Haas":            6.5,
    "Aston Martin":    8.0,
    "Cadillac":        6.5,
}
qualifying_2026["ERSDependencyScore"] = qualifying_2026["Team"].map(
    ERS_DEPENDENCY
)

# ══════════════════════════════════════════════════════════
# 11. RELIABILITY RISK — heat stress on cars
# ══════════════════════════════════════════════════════════
RELIABILITY_RISK = {
    "Mercedes":        4.5,
    "McLaren":         2.5,
    "Ferrari":         2.5,
    "Red Bull Racing": 3.0,
    "Alpine":          4.0,
    "Audi":            5.5,
    "Racing Bulls":    3.5,
    "Williams":        4.0,
    "Haas":            4.0,
    "Aston Martin":    4.0,
    "Cadillac":        5.0,
}
qualifying_2026["ReliabilityRiskScore"] = qualifying_2026["Team"].map(
    RELIABILITY_RISK
)

# ══════════════════════════════════════════════════════════
# 12. HOME RACE BOOST — no Malaysian drivers in Q3
# ══════════════════════════════════════════════════════════
qualifying_2026["HomeRaceBoost"] = 0.0

# ══════════════════════════════════════════════════════════
# 13. CIRCUIT SCORE — 15 ROUNDS OF 2026 DATA
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS  CHN  JPN  MIA  CAN  MON  ESP  AUT  GBR  BEL  HUN  NED  ITA  MAD  AZE
    "VER":              [20,  20,  8,   5,   3,   20,  5,   2,   20,  3,   2,   20,  3,   2,   2],
    "HAM":              [7,   3,   6,   7,   2,   2,   1,   5,   3,   4,   5,   4,   6,   20,  6],
    "HAD":              [20,  8,   9,   20,  5,   3,   6,   20,  5,   6,   20,  20,  20,  20,  3],
    "ANT":              [2,   1,   1,   1,   1,   1,   4,   3,   20,  1,   3,   2,   1,   1,   5],
    "LEC":              [3,   4,   3,   6,   4,   20,  20,  20,  1,   2,   4,   5,   20,  4,   4],
    "NOR":              [5,   20,  5,   2,   20,  20,  3,   20,  4,   7,   1,   1,   4,   3,   5],
    "PIA":              [22,  2,   2,   3,   20,  4,   20,  4,   20,  5,   20,  6,   5,   8,   20],
    "RUS":              [1,   2,   4,   4,   20,  20,  2,   1,   2,   20,  7,   3,   2,   5,   1],
    "GAS":              [10,  6,   10,  20,  8,   7,   20,  20,  20,  20,  12,  10,  7,   20,  20],
    "BOR":              [20,  20,  20,  20,  20,  20,  20,  8,   8,   20,  20,  20,  20,  20,  20],
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
# 14. SYNTHETIC SECTOR TIMES
#     Sepang split ratios (approximate)
#     S1: 30% — start to Turn 9
#     S2: 38% — back straight section
#     S3: 32% — final sector to finish
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.30
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.38
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.32
# Tropical race pace ~8% slower (heat, humidity, potential rain)
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.08

# ══════════════════════════════════════════════════════════
# 15. WEATHER FEATURES
# ══════════════════════════════════════════════════════════
qualifying_2026["RainProbability"] = RAIN_PROBABILITY
qualifying_2026["Temperature"]     = RACE_TEMP
qualifying_2026["TempDelta"]       = TEMP_DELTA
qualifying_2026["Humidity"]        = HUMIDITY
qualifying_2026["WindSpeed"]       = WIND_SPEED

print("\n📊 Full Feature Set:")
print(qualifying_2026[[
    "Driver", "QualifyingTime (s)", "GapFromPole (s)",
    "AdjustedTeamScore", "TropicalHeatScore",
    "TyreDegScore", "CircuitScore"
]].to_string(index=False))

# ══════════════════════════════════════════════════════════
# 16. FEATURE COLUMNS
# ══════════════════════════════════════════════════════════
FEATURE_COLS = [
    "QualifyingTime (s)",
    "GapFromPole (s)",
    "AdjustedTeamScore",
    "GridPenalty (s)",
    "WetPerformanceFactor",
    "PoleWetBonus",
    "RainProbability",
    "Temperature",
    "TempDelta",
    "Humidity",               # 75% — most humid race of season
    "WindSpeed",
    "ERSDependencyScore",
    "SepangGridPenalty",
    "TyreDegScore",           # extreme tropical heat
    "TropicalHeatScore",      # 🆕 32°C + 75% humidity
    "HomeRaceBoost",
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",           # 15 rounds of 2026 data
    "SprintWinnerBoost",
]
TARGET = "RacePace (s)"

# ══════════════════════════════════════════════════════════
# 17. TRAIN MODEL
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
# 18. PREDICT RACE
# ══════════════════════════════════════════════════════════
data = qualifying_2026.copy()
data["PredictedLapTime (s)"] = model.predict(X)

# Sepang grid penalty
data["PredictedLapTime (s)"] += data["SepangGridPenalty"] * 0.35

# Tropical heat penalty
data["PredictedLapTime (s)"] += data["TropicalHeatScore"] * 0.03

# Tyre deg — extreme heat
data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.02

# Wet bonus — 30% rain meaningful but below threshold
data["WetBonus"] = (
    (1 - data["WetPerformanceFactor"]) * RAIN_PROBABILITY * 100
)
data["PredictedLapTime (s)"] -= data["WetBonus"]

# Pole wet bonus — zero (30% < 60%)
data["PredictedLapTime (s)"] -= data["PoleWetBonus"]

# Sort
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 19. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 BAHRAIN GP (MALAYSIA) — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Race: {RACE_TEMP}°C ⛅  |  💧 Humidity: {HUMIDITY}%")
print(f"  🌧️  Rain: {int(RAIN_PROBABILITY*100)}% — tropical showers possible")
print(f"  🔥  Most extreme tropical conditions of 2026 season!")
print(f"  🔵  Verstappen pole — Red Bull Ford 1-3 in qualifying!")
print(f"  🌟  Historic: First Bahrain GP held at Sepang, Malaysia!\n")

# ══════════════════════════════════════════════════════════
# 20. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 16: BAHRAIN GP (MALAYSIA)\n"
    "SEPANG INTERNATIONAL CIRCUIT  |  OCT 4, 2026  |  🔥 32°C  |  💧 75% HUMIDITY",
    fontsize=15, fontweight="bold", color="white",
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
    "📊 Predicted Race Finishing Order  (🔥 Tropical Sepang — Most Extreme Conditions!)",
    fontsize=12, fontweight="bold", color="white",
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

# ── Chart 2: Tropical Heat Score ─────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
heat_sorted = data.sort_values("TropicalHeatScore")
heat_colors = [TEAM_COLORS.get(t, "#FFF") for t in heat_sorted["Team"]]
ax2.barh(
    heat_sorted["Driver"][::-1],
    heat_sorted["TropicalHeatScore"][::-1],
    color=heat_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title(
    "🌴 Tropical Heat Score\n(lower = better adapted to 32°C + 75% humidity)",
    fontsize=10, fontweight="bold", color="white",
    fontfamily=FONT, pad=10
)
ax2.set_xlabel("Tropical Heat Score (lower = better)",
               color="#AAAAAA", fontsize=8, fontfamily=FONT)
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
ax3.set_title("⏱️  Qualifying Gap to Pole (Sepang Q3)",
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
        row["GapFromPole (s)"] + 0.01, i,
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
    "Humidity 💧", "Wind Speed", "ERS Dependency",
    "Sepang Grid", "Tyre Deg 🔥",
    "Tropical Heat 🆕", "Home Boost",
    "Reliability", "Sector 1", "Sector 2", "Sector 3",
    "Circuit Score", "Sprint Boost"
]
feat_import   = model.feature_importances_
sorted_idx    = np.argsort(feat_import)
sorted_labels = [feat_labels[i] for i in sorted_idx]
sorted_values = feat_import[sorted_idx]
colors_bar    = plt.cm.YlOrRd(np.linspace(0.3, 0.95, len(sorted_values)))
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

podium = data[data["PredictedPosition"] <= 3].sort_values("PredictedPosition")
podium_y    = [0.75, 0.47, 0.19]
podium_icon = ["🥇", "🥈", "🥉"]
podium_size = [22, 18, 16]
ax5.set_title("🏆 Predicted Podium  🇲🇾",
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

fig.text(
    0.5, 0.01,
    f"🔍 MAE: {mae:.2f}s  |  "
    f"🔥 Race: {RACE_TEMP}°C  |  "
    f"💧 Humidity: {HUMIDITY}%  |  "
    f"🌧️ Rain: {int(RAIN_PROBABILITY*100)}%  |  "
    f"🔵 VER pole — Red Bull Ford 1-3!  |  "
    f"🌟 Historic: Bahrain GP at Sepang!",
    ha="center", fontsize=7, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_16_malaysia_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_16_malaysia_prediction.png")
plt.show()