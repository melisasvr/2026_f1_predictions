"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 8: AUSTRIAN GP
 Red Bull Ring, Spielberg | Race Date: June 29, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.07 (7% slower)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            AustriaGridPenalty, TyreDegScore,
            ReliabilityRiskScore, CircuitScore,
            SprintWinnerBoost, HomeRaceBoost
 Upgrades vs R07:
            + 33°C hottest race of season — TyreDeg #1
            + HomeRaceBoost for Verstappen (Red Bull Ring)
            + Verstappen crash in Q3 — yellow flag drama
            + Ferrari front row lock P2+P3
            + Both Racing Bulls in Q3 (Lawson + Lindblad)
            + AustriaGridPenalty — good overtaking circuit
            + 7 rounds of 2026 CircuitScore data
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 8: AUSTRIAN GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 33    # °C — hottest qualifying of season
RACE_TEMP        = 33    # °C — equally hot race day
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # 0°C
RAIN_PROBABILITY = 0.20  # 20% — low but worth noting
HUMIDITY         = 50    # % race day
WIND_SPEED       = 8     # km/h race day

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  Race: {RACE_TEMP}°C ☀️  (Δ{TEMP_DELTA}°C)")
print(f"🔥  HOTTEST RACE OF 2026 — 33°C tyre degradation critical!")
print(f"🌧️  Rain: {int(RAIN_PROBABILITY*100)}% — low but logged")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Russell pole — 2nd consecutive!
#    Verstappen CRASHED end of Q3 — yellow flag 😱
#    Ferrari P2+P3 — front row threat
# ══════════════════════════════════════════════════════════
POLE_TIME = 66.113  # Russell 1:06.113

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "George Russell",
        "Charles Leclerc",
        "Lewis Hamilton",
        "Kimi Antonelli",
        "Max Verstappen",
        "Lando Norris",
        "Oscar Piastri",
        "Isack Hadjar",
        "Liam Lawson",
        "Arvid Lindblad",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        66.113,   # 1:06.113 — Russell POLE 🌟
        66.349,   # +0.236s  — Leclerc
        66.408,   # +0.295s  — Hamilton
        66.414,   # +0.301s  — Antonelli
        66.475,   # +0.362s  — Verstappen (CRASHED after lap) 😱
        66.502,   # +0.389s  — Norris
        66.511,   # +0.398s  — Piastri
        66.632,   # +0.519s  — Hadjar
        66.955,   # +0.842s  — Lawson
        67.007,   # +0.894s  — Lindblad
    ],
    "Team": [
        "Mercedes", "Ferrari",         "Ferrari",
        "Mercedes", "Red Bull Racing",  "McLaren",
        "McLaren",  "Red Bull Racing",  "Racing Bulls",
        "Racing Bulls",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "SprintWinnerBoost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "George Russell":  "RUS",
    "Charles Leclerc": "LEC",
    "Lewis Hamilton":  "HAM",
    "Kimi Antonelli":  "ANT",
    "Max Verstappen":  "VER",
    "Lando Norris":    "NOR",
    "Oscar Piastri":   "PIA",
    "Isack Hadjar":    "HAD",
    "Liam Lawson":     "LAW",
    "Arvid Lindblad":  "LIN",
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
# 4. ADJUSTED TEAM SCORE — updated after 7 rounds
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.8,  # Dominant — Russell 2 poles in a row
    "Ferrari":         8.8,  # Hamilton wins Spain — Ferrari surging!
    "McLaren":         8.0,  # Consistent pace, good tyre management
    "Red Bull Racing": 7.0,  # VER crash in Q3 — concerning
    "Racing Bulls":    5.5,  # Both cars in Q3! Best result yet
    "Alpine":          5.0,
    "Haas":            4.5,
    "Aston Martin":    4.0,
    "Williams":        4.5,
    "Audi":            5.0,  # Hulkenberg P9 Spain
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR (20% rain — low impact)
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "RUS": 0.966,
    "LEC": 0.974,
    "HAM": 0.964,
    "ANT": 0.972,
    "VER": 0.968,
    "NOR": 0.976,
    "PIA": 0.975,
    "HAD": 0.980,
    "LAW": 0.979,
    "LIN": 0.983,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — low rain (20% < 60% threshold)
# ══════════════════════════════════════════════════════════
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY if (
        p == 1 and RAIN_PROBABILITY >= 0.60
    ) else 0.0
)

# ══════════════════════════════════════════════════════════
# 7. AUSTRIA GRID PENALTY
#    Red Bull Ring has good overtaking at T3 and T4
#    Medium-high overtaking vs Monaco/Suzuka
#    But still significant — 0.10s per position
# ══════════════════════════════════════════════════════════
qualifying_2026["AustriaGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.10
)

# ══════════════════════════════════════════════════════════
# 8. TYRE DEG SCORE — 33°C HOTTEST RACE OF SEASON 🔥
#    Most critical feature this round
#    McLaren best tyre managers — big advantage here
# ══════════════════════════════════════════════════════════
TYRE_DEG = {
    "Mercedes":        2.0,
    "Ferrari":         2.5,
    "McLaren":         1.5,  # Best tyre management
    "Red Bull Racing": 2.5,
    "Racing Bulls":    3.5,
    "Audi":            4.0,
    "Alpine":          3.5,
    "Haas":            3.5,
    "Aston Martin":    3.0,
    "Williams":        3.5,
    "Cadillac":        4.5,
}
qualifying_2026["TyreDegScore"] = qualifying_2026["Team"].map(TYRE_DEG)

# ══════════════════════════════════════════════════════════
# 9. HOME RACE BOOST — Verstappen at Red Bull Ring 🆕
#    Verstappen has incredible record at his home race
#    Austrian crowd = massive motivation
#    Small but meaningful boost to race pace
# ══════════════════════════════════════════════════════════
qualifying_2026["HomeRaceBoost"] = qualifying_2026["DriverCode"].apply(
    lambda d: 1 if d == "VER" else 0
)

# ══════════════════════════════════════════════════════════
# 10. ERS DEPENDENCY (7MJ limit — short lap = more biting)
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
# 11. RELIABILITY RISK
# ══════════════════════════════════════════════════════════
RELIABILITY_RISK = {
    "Mercedes":        3.0,
    "Ferrari":         2.0,
    "Red Bull Racing": 4.0,  # VER crash in Q3 — car damage concern
    "McLaren":         3.0,
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
# 12. CIRCUIT SCORE — 7 ROUNDS OF 2026 DATA
#     AUS+CHN+JPN+MIA+CAN+MON+ESP
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS  CHN  JPN  MIA  CAN  MON  ESP
    "RUS":              [1,   2,   4,   4,   20,  20,  2],
    "LEC":              [3,   4,   3,   6,   4,   20,  20],
    "HAM":              [7,   3,   6,   7,   2,   2,   1],
    "ANT":              [2,   1,   1,   1,   1,   1,   4],
    "VER":              [20,  20,  8,   5,   3,   20,  5],
    "NOR":              [5,   20,  5,   2,   20,  20,  3],
    "PIA":              [22,  2,   2,   3,   20,  4,   20],
    "HAD":              [20,  8,   9,   20,  5,   3,   6],
    "LAW":              [20,  20,  20,  20,  7,   5,   20],
    "LIN":              [8,   20,  14,  20,  6,   6,   20],
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
# 13. SYNTHETIC SECTOR TIMES
#     Austria split ratios (approximate)
#     S1: 33% — start to T3 hairpin
#     S2: 37% — T3 to T9 (Rindt/Jochen)
#     S3: 30% — T9 to finish line
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.33
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.37
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.30
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.07

# ══════════════════════════════════════════════════════════
# 14. WEATHER FEATURES
# ══════════════════════════════════════════════════════════
qualifying_2026["RainProbability"] = RAIN_PROBABILITY
qualifying_2026["Temperature"]     = RACE_TEMP
qualifying_2026["TempDelta"]       = TEMP_DELTA
qualifying_2026["Humidity"]        = HUMIDITY
qualifying_2026["WindSpeed"]       = WIND_SPEED

print("\n📊 Full Feature Set:")
print(qualifying_2026[[
    "Driver", "QualifyingTime (s)", "GapFromPole (s)",
    "AdjustedTeamScore", "TyreDegScore",
    "AustriaGridPenalty", "HomeRaceBoost", "CircuitScore"
]].to_string(index=False))

# ══════════════════════════════════════════════════════════
# 15. FEATURE COLUMNS
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
    "AustriaGridPenalty",   # medium overtaking difficulty
    "TyreDegScore",         # 🔥 33°C hottest race — #1 feature
    "HomeRaceBoost",        # 🆕 Verstappen home race
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",         # 7 rounds of 2026 data
    "SprintWinnerBoost",
]
TARGET = "RacePace (s)"

# ══════════════════════════════════════════════════════════
# 16. TRAIN MODEL
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
# 17. PREDICT RACE
# ══════════════════════════════════════════════════════════
data = qualifying_2026.copy()
data["PredictedLapTime (s)"] = model.predict(X)

# Austria grid penalty
data["PredictedLapTime (s)"] += data["AustriaGridPenalty"] * 0.4

# Tyre deg — 33°C punishes high deg teams hard
data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.05

# Home race boost — Verstappen
data["PredictedLapTime (s)"] -= data["HomeRaceBoost"] * 0.08

# Wet bonus — minimal at 20%
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
# 18. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 AUSTRIAN GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Race: {RACE_TEMP}°C ☀️ — hottest of 2026!")
print(f"  🔥  Tyre deg: critical feature at 33°C")
print(f"  🌧️  Rain: {int(RAIN_PROBABILITY*100)}% — low")
print(f"  🔋  ERS: 7MJ — short lap bites hard")
print(f"  😱  Verstappen CRASHED in Q3 — starts P5")
print(f"  🌟  Pole: Russell — 2nd consecutive!\n")

# ══════════════════════════════════════════════════════════
# 19. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 8: AUSTRIAN GP\n"
    "RED BULL RING, SPIELBERG  |  JUNE 29, 2026  |  🔥 33°C",
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
    "📊 Predicted Race Finishing Order  (🔥 33°C — Hottest Race of 2026)",
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

# ── Chart 2: Tyre Degradation Score ──────────────────────
ax2 = fig.add_subplot(gs[1, 0])
tyre_sorted  = data.sort_values("TyreDegScore")
tyre_colors  = [TEAM_COLORS.get(t, "#FFF") for t in tyre_sorted["Team"]]
ax2.barh(
    tyre_sorted["Driver"][::-1],
    tyre_sorted["TyreDegScore"][::-1],
    color=tyre_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title(
    "🔥 Tyre Degradation Score\n(lower = better tyre management — 33°C!)",
    fontsize=10, fontweight="bold", color="white",
    fontfamily=FONT, pad=10
)
ax2.set_xlabel("Tyre Deg Score (lower = better)",
               color="#AAAAAA", fontsize=8, fontfamily=FONT)
ax2.tick_params(colors="white", labelsize=8)
ax2.set_facecolor("#1a1a1a")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333333")
for i, (_, row) in enumerate(tyre_sorted[::-1].iterrows()):
    ax2.text(row["TyreDegScore"] + 0.05, i,
             f"{row['TyreDegScore']:.1f}",
             va="center", fontsize=8,
             color="white", fontfamily=FONT)

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
    ax3.text(row["GapFromPole (s)"] + 0.005, i,
             f"+{row['GapFromPole (s)']:.3f}s",
             va="center", fontsize=7.5,
             color="white", fontfamily=FONT)

# ── Chart 4: Feature Importance ──────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
feat_labels = [
    "Qualifying Time", "Gap From Pole", "Team Score",
    "Grid Penalty", "Wet Factor", "Pole Wet Bonus",
    "Rain Prob", "Temperature", "Temp Delta",
    "Humidity", "Wind Speed", "ERS Dependency",
    "Austria Grid", "Tyre Deg 🔥",
    "Home Boost 🆕", "Reliability",
    "Sector 1", "Sector 2", "Sector 3",
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

podium = data[data["PredictedPosition"] <= 3].sort_values("PredictedPosition")
podium_y    = [0.75, 0.47, 0.19]
podium_icon = ["🥇", "🥈", "🥉"]
podium_size = [22, 18, 16]
ax5.set_title("🏆 Predicted Podium  🔥",
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
    f"🌧️ Rain: {int(RAIN_PROBABILITY*100)}%  |  "
    f"🔋 ERS: 7MJ  |  "
    f"😱 VER crashed Q3  |  "
    f"🌟 Pole: Russell (2nd consecutive!)",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_08_austria_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_08_austria_prediction.png")
plt.show()
