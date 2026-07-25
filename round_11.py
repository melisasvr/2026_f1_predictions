"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 11: HUNGARIAN GP
 Hungaroring, Budapest | Race Date: July 26, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.07 (7% slower)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            HungaryGridPenalty, TyreDegScore,
            ReliabilityRiskScore, CircuitScore,
            SprintWinnerBoost, HomeRaceBoost,
            Hungary2025Score
 Upgrades vs R10:
            + Norris pole — Antonelli streak broken at 7!
            + Hungary2025Score — 2025 race result at Budapest
            + McLaren 1-2 in 2025 Hungary — strong signal
            + Ferrari P2+P3 — Hamilton razor close to Norris
            + Russell no time set in Q3 — starts P7
            + HungaryGridPenalty — very hard to overtake
            + 10 rounds of 2026 CircuitScore data
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 11: HUNGARIAN GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 29    # °C — sunshine Saturday
RACE_TEMP        = 30    # °C — sunshine/cloudy Sunday
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # +1°C
RAIN_PROBABILITY = 0.15  # 15% — essentially dry
HUMIDITY         = 50    # % estimated
WIND_SPEED       = 10    # km/h

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  Race: {RACE_TEMP}°C ⛅  (Δ+{TEMP_DELTA}°C)")
print(f"🔥  30°C hot race — tyre deg meaningful")
print(f"🌧️  Rain: {int(RAIN_PROBABILITY*100)}% — essentially dry")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Norris POLE — Antonelli streak ends at 7! 🟠
#    Ferrari P2+P3 — Hamilton only 0.012s behind!
#    Russell P7 — no time set (deleted lap)
# ══════════════════════════════════════════════════════════
POLE_TIME = 77.207  # Norris 1:17.207

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Lando Norris",
        "Lewis Hamilton",
        "Charles Leclerc",
        "Kimi Antonelli",
        "Oscar Piastri",
        "Max Verstappen",
        "George Russell",
        "Isack Hadjar",
        "Arvid Lindblad",
        "Nico Hulkenberg",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        77.207,   # 1:17.207 — Norris POLE 🌟
        77.219,   # +0.012s  — Hamilton
        77.445,   # +0.238s  — Leclerc
        77.479,   # +0.272s  — Antonelli
        77.684,   # +0.477s  — Piastri
        77.725,   # +0.518s  — Verstappen
        77.900,   # +0.693s  — Russell (estimated — no time)
        77.856,   # +0.649s  — Hadjar
        78.281,   # +1.074s  — Lindblad
        78.686,   # +1.479s  — Hulkenberg
    ],
    "Team": [
        "McLaren",  "Ferrari",         "Ferrari",
        "Mercedes", "McLaren",          "Red Bull Racing",
        "Mercedes", "Red Bull Racing",  "Racing Bulls",
        "Audi",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "SprintWinnerBoost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Lando Norris":    "NOR",
    "Lewis Hamilton":  "HAM",
    "Charles Leclerc": "LEC",
    "Kimi Antonelli":  "ANT",
    "Oscar Piastri":   "PIA",
    "Max Verstappen":  "VER",
    "George Russell":  "RUS",
    "Isack Hadjar":    "HAD",
    "Arvid Lindblad":  "LIN",
    "Nico Hulkenberg": "HUL",
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
# 4. ADJUSTED TEAM SCORE — updated after 10 rounds
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.5,  # Dominant but Antonelli 2x DNF concern
    "Ferrari":         9.2,  # Leclerc Belgium P2, Hamilton strong
    "McLaren":         9.0,  # Norris pole Hungary — bouncing back!
    "Red Bull Racing": 7.0,  # VER P3 Belgium — improving
    "Racing Bulls":    6.0,  # Lindblad P9 consistently
    "Audi":            5.5,  # Hulkenberg P10 — Q3 two races running
    "Alpine":          5.0,
    "Haas":            4.5,
    "Aston Martin":    4.5,  # Alonso P5 in 2025 Hungary — circuit suits them
    "Williams":        4.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR — minimal (15% rain)
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "NOR": 0.976,
    "HAM": 0.964,
    "LEC": 0.974,
    "ANT": 0.972,
    "PIA": 0.975,
    "VER": 0.968,
    "RUS": 0.966,
    "HAD": 0.980,
    "LIN": 0.983,
    "HUL": 0.978,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — 15% rain < 60% threshold = zero
# ══════════════════════════════════════════════════════════
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY if (
        p == 1 and RAIN_PROBABILITY >= 0.60
    ) else 0.0
)

# ══════════════════════════════════════════════════════════
# 7. HUNGARY GRID PENALTY
#    Hungaroring is 2nd hardest to overtake after Monaco
#    Tight twisty layout — very few overtaking spots
#    0.13s per position penalty (between Monaco 0.15 and Spa 0.09)
# ══════════════════════════════════════════════════════════
qualifying_2026["HungaryGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.13
)

# ══════════════════════════════════════════════════════════
# 8. TYRE DEG SCORE — 30°C hot race
# ══════════════════════════════════════════════════════════
TYRE_DEG = {
    "Mercedes":        2.0,
    "Ferrari":         2.5,
    "McLaren":         1.5,  # Best tyre management — key at Hungary
    "Red Bull Racing": 2.5,
    "Racing Bulls":    3.0,
    "Audi":            3.5,
    "Alpine":          3.0,
    "Haas":            3.5,
    "Aston Martin":    2.5,  # Good tyre management
    "Williams":        3.5,
    "Cadillac":        4.5,
}
qualifying_2026["TyreDegScore"] = qualifying_2026["Team"].map(TYRE_DEG)

# ══════════════════════════════════════════════════════════
# 9. HUNGARY 2025 SCORE — NEW FEATURE 🆕
#    2025 Hungarian GP race results as circuit-specific signal
#    NOR won, PIA P2, RUS P3, LEC P4 — McLaren dominated
#    Lower = better 2025 Hungary performance
# ══════════════════════════════════════════════════════════
HUNGARY_2025 = {
    "NOR": 1.0,   # Won Hungary 2025
    "HAM": 10.0,  # Not in top results (was at Mercedes)
    "LEC": 4.0,   # P4 in 2025
    "ANT": 10.0,  # Not racing in 2025 Hungary
    "PIA": 2.0,   # P2 in 2025
    "VER": 9.0,   # Struggled at Hungary 2025
    "RUS": 3.0,   # P3 in 2025
    "HAD": 10.0,  # Limited 2025 data
    "LIN": 10.0,  # Not in F1 2025
    "HUL": 10.0,  # Not in top results
}
qualifying_2026["Hungary2025Score"] = qualifying_2026["DriverCode"].map(
    HUNGARY_2025
)
# Normalize to 1-5 scale
qualifying_2026["Hungary2025Score"] = 1 + (
    qualifying_2026["Hungary2025Score"] - 1
) * (4 / 9)

# ══════════════════════════════════════════════════════════
# 10. ERS DEPENDENCY (7MJ limit)
# ══════════════════════════════════════════════════════════
ERS_DEPENDENCY = {
    "Mercedes":        9.0,
    "McLaren":         9.0,
    "Ferrari":         6.5,
    "Red Bull Racing": 5.5,
    "Alpine":          6.5,
    "Racing Bulls":    5.5,
    "Haas":            6.5,
    "Aston Martin":    8.0,
    "Williams":        9.0,
    "Audi":            7.0,
    "Cadillac":        6.5,
}
qualifying_2026["ERSDependencyScore"] = qualifying_2026["Team"].map(
    ERS_DEPENDENCY
)

# ══════════════════════════════════════════════════════════
# 11. RELIABILITY RISK — Antonelli 2x DNF flagged
# ══════════════════════════════════════════════════════════
RELIABILITY_RISK = {
    "Mercedes":        4.5,  # Antonelli 2x DNF — high concern
    "Ferrari":         2.0,
    "McLaren":         2.5,  # Improving reliability
    "Red Bull Racing": 3.0,
    "Racing Bulls":    3.5,
    "Audi":            5.0,
    "Alpine":          4.0,
    "Haas":            3.5,
    "Aston Martin":    3.5,
    "Williams":        4.0,
    "Cadillac":        5.0,
}
qualifying_2026["ReliabilityRiskScore"] = qualifying_2026["Team"].map(
    RELIABILITY_RISK
)

# ══════════════════════════════════════════════════════════
# 12. CIRCUIT SCORE — 10 ROUNDS OF 2026 DATA
#     AUS+CHN+JPN+MIA+CAN+MON+ESP+AUT+GBR+BEL
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS  CHN  JPN  MIA  CAN  MON  ESP  AUT  GBR  BEL
    "NOR":              [5,   20,  5,   2,   20,  20,  3,   20,  4,   7],
    "HAM":              [7,   3,   6,   7,   2,   2,   1,   5,   3,   4],
    "LEC":              [3,   4,   3,   6,   4,   20,  20,  20,  1,   2],
    "ANT":              [2,   1,   1,   1,   1,   1,   4,   3,   20,  1],
    "PIA":              [22,  2,   2,   3,   20,  4,   20,  4,   20,  5],
    "VER":              [20,  20,  8,   5,   3,   20,  5,   2,   20,  3],
    "RUS":              [1,   2,   4,   4,   20,  20,  2,   1,   2,   20],
    "HAD":              [20,  8,   9,   20,  5,   3,   6,   20,  5,   6],
    "LIN":              [8,   20,  14,  20,  6,   6,   20,  20,  7,   9],
    "HUL":              [20,  20,  20,  20,  20,  20,  9,   20,  20,  13],
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
# 13. HOME RACE BOOST — No specific home drivers in top 10
# ══════════════════════════════════════════════════════════
qualifying_2026["HomeRaceBoost"] = 0.0

# ══════════════════════════════════════════════════════════
# 14. SYNTHETIC SECTOR TIMES
#     Hungaroring split ratios (approximate)
#     S1: 33% — start to T4 chicane
#     S2: 40% — T5 to T11 twisty section
#     S3: 27% — T12 to finish
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.33
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.40
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.27
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.07

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
    "AdjustedTeamScore", "Hungary2025Score",
    "HungaryGridPenalty", "CircuitScore"
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
    "Humidity",
    "WindSpeed",
    "ERSDependencyScore",
    "HungaryGridPenalty",    # 2nd hardest to overtake after Monaco
    "TyreDegScore",          # 30°C hot race
    "Hungary2025Score",      # 🆕 2025 Hungary race history
    "HomeRaceBoost",
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",          # 10 rounds of 2026 data
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

# Hungary grid penalty — 2nd hardest circuit to overtake
data["PredictedLapTime (s)"] += data["HungaryGridPenalty"] * 0.45

# Tyre deg — 30°C meaningful
data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.02

# Hungary 2025 history bonus
data["PredictedLapTime (s)"] += (data["Hungary2025Score"] - 1) * 0.03

# Wet bonus — minimal at 15%
data["WetBonus"] = (
    (1 - data["WetPerformanceFactor"]) * RAIN_PROBABILITY * 100
)
data["PredictedLapTime (s)"] -= data["WetBonus"]

# Pole wet bonus — zero (<60%)
data["PredictedLapTime (s)"] -= data["PoleWetBonus"]

# Sort
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 19. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 HUNGARIAN GP — PREDICTED RACE RESULT")
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
print(f"  🔥  30°C — hot race, tyre management key")
print(f"  🌧️  Rain: {int(RAIN_PROBABILITY*100)}% — essentially dry")
print(f"  🔋  ERS: 7MJ  |  ⚡ Boost cap: +150kW")
print(f"  🟠  Pole: Norris — Antonelli streak ends at 7!")
print(f"  🔴  Ferrari P2+P3 — Hamilton only 0.012s off pole!\n")

# ══════════════════════════════════════════════════════════
# 20. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 11: HUNGARIAN GP\n"
    "HUNGARORING, BUDAPEST  |  JULY 27, 2026  |  ☀️ 30°C",
    fontsize=17, fontweight="bold", color="white",
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
    "📊 Predicted Race Finishing Order  (🟠 Norris Pole — Antonelli Streak Broken!)",
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

# ── Chart 2: Hungary 2025 History Score ──────────────────
ax2 = fig.add_subplot(gs[1, 0])
h25_sorted = data.sort_values("Hungary2025Score")
h25_colors = [TEAM_COLORS.get(t, "#FFF") for t in h25_sorted["Team"]]
ax2.barh(
    h25_sorted["Driver"][::-1],
    h25_sorted["Hungary2025Score"][::-1],
    color=h25_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title(
    "🏆 Hungary 2025 History Score\n(lower = better 2025 Budapest result)",
    fontsize=10, fontweight="bold", color="white",
    fontfamily=FONT, pad=10
)
ax2.set_xlabel("History Score (lower = better)",
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
    "Hungary Grid", "Tyre Deg",
    "Hungary 2025 🆕", "Home Boost",
    "Reliability", "Sector 1", "Sector 2", "Sector 3",
    "Circuit Score", "Sprint Boost"
]
feat_import   = model.feature_importances_
sorted_idx    = np.argsort(feat_import)
sorted_labels = [feat_labels[i] for i in sorted_idx]
sorted_values = feat_import[sorted_idx]
colors_bar    = plt.cm.Oranges(np.linspace(0.3, 0.95, len(sorted_values)))
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
ax5.set_title("🏆 Predicted Podium  🇭🇺",
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
    f"☀️ Race: {RACE_TEMP}°C  |  "
    f"🌧️ Rain: {int(RAIN_PROBABILITY*100)}%  |  "
    f"🔋 ERS: 7MJ  |  "
    f"🟠 Pole: Norris (streak broken!)  |  "
    f"🔴 HAM only 0.012s off pole",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_11_hungary_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_11_hungary_prediction.png")
plt.show()