"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 15: AZERBAIJAN GP
 Baku City Circuit | Race Date: September 26, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.06 (6% slower Baku)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            BakuGridPenalty, TyreDegScore,
            ReliabilityRiskScore, CircuitScore,
            SprintWinnerBoost, HomeRaceBoost,
            BakuStraightSpeed, SafetyCarProb
 Upgrades vs R14:
            + Russell pole strong wind qualifying
            + Antonelli OUT of Q3 massive shock!
            + Sainz P9 Williams in Q3!
            + BakuStraightSpeed 2.2km straight critical
            + SafetyCarProb Baku safety car almost certain
            + ERS most critical feature on longest straight
            + 14 rounds of 2026 CircuitScore data
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 15: AZERBAIJAN GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 23    # °C — sunny + STRONG winds
RACE_TEMP        = 24    # °C — sunny + breeze
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # +1°C
RAIN_PROBABILITY = 0.00  # 0% — completely dry
HUMIDITY         = 45    # % estimated
WIND_SPEED       = 20    # km/h qualifying (strong) — drops for race

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  Race: {RACE_TEMP}°C ☀️  (Δ+{TEMP_DELTA}°C)")
print(f"💨  Strong qualifying winds — calmer race day")
print(f"🏰  BAKU — longest straight in F1 (2.2km)!")
print(f"😱  Antonelli NOT in top 10 — shock of the weekend!")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Russell pole 🌟 — strong winds helped Mercedes
#    Antonelli MISSING from Q3 — eliminated in Q2!
#    Sainz P9 — Williams best result of 2026!
# ══════════════════════════════════════════════════════════
POLE_TIME = 102.526  # Russell 1:42.526

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "George Russell",
        "Charles Leclerc",
        "Oscar Piastri",
        "Isack Hadjar",
        "Lando Norris",
        "Lewis Hamilton",
        "Pierre Gasly",
        "Max Verstappen",
        "Carlos Sainz",
        "Franco Colapinto",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        102.526,  # 1:42.526 — Russell POLE 🌟
        103.363,  # +0.837s  — Leclerc
        103.364,  # +0.838s  — Piastri
        103.500,  # +0.974s  — Hadjar
        103.672,  # +1.146s  — Norris
        103.858,  # +1.332s  — Hamilton
        104.047,  # +1.521s  — Gasly
        104.081,  # +1.555s  — Verstappen
        104.566,  # +2.040s  — Sainz
        104.963,  # +2.437s  — Colapinto
    ],
    "Team": [
        "Mercedes", "Ferrari",         "McLaren",
        "Red Bull Racing", "McLaren",   "Ferrari",
        "Alpine",   "Red Bull Racing",  "Williams",
        "Alpine",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "SprintWinnerBoost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "George Russell":   "RUS",
    "Charles Leclerc":  "LEC",
    "Oscar Piastri":    "PIA",
    "Isack Hadjar":     "HAD",
    "Lando Norris":     "NOR",
    "Lewis Hamilton":   "HAM",
    "Pierre Gasly":     "GAS",
    "Max Verstappen":   "VER",
    "Carlos Sainz":     "SAI",
    "Franco Colapinto": "COL",
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
# 4. ADJUSTED TEAM SCORE — updated after 14 rounds
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.5,  # Russell strong — but Antonelli Q2 exit concern
    "McLaren":         9.3,  # Norris 3 wins — title battle
    "Ferrari":         9.0,  # Leclerc P2 — strong at Baku
    "Red Bull Racing": 7.5,  # Hadjar P4! — Red Bull improving
    "Alpine":          7.0,  # Gasly pole Monza — both cars Q3 again
    "Williams":        6.5,  # Sainz P9 — best result of 2026! 🌟
    "Racing Bulls":    5.5,
    "Audi":            5.5,
    "Haas":            5.0,
    "Aston Martin":    4.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE — 0% rain, minimal
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "RUS": 0.966,
    "LEC": 0.974,
    "PIA": 0.975,
    "HAD": 0.980,
    "NOR": 0.976,
    "HAM": 0.964,
    "GAS": 0.977,
    "VER": 0.966,
    "SAI": 0.978,
    "COL": 0.978,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — 0% rain = zero
# ══════════════════════════════════════════════════════════
qualifying_2026["PoleWetBonus"] = 0.0

# ══════════════════════════════════════════════════════════
# 7. BAKU GRID PENALTY
#    Baku has good overtaking on the main straight
#    But castle section makes position changes hard
#    Medium overtaking difficulty: 0.09s per position
# ══════════════════════════════════════════════════════════
qualifying_2026["BakuGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.09
)

# ══════════════════════════════════════════════════════════
# 8. BAKU STRAIGHT SPEED — NEW FEATURE 🆕
#    2.2km Caspian Sea straight — longest in F1
#    Pure straight line speed is critical
#    Low ERS dependency teams = more raw ICE power
#    Higher score = better straight line speed advantage
# ══════════════════════════════════════════════════════════
STRAIGHT_SPEED = {
    "RUS": 0.7,   # Mercedes PU — good straight line
    "LEC": 1.0,   # Ferrari PU — elite at Baku straight
    "PIA": 0.7,   # McLaren — good straight line
    "HAD": 1.2,   # Red Bull Ford — low ERS = ICE power
    "NOR": 0.7,   # McLaren
    "HAM": 1.0,   # Ferrari PU
    "GAS": 0.8,   # Renault PU
    "VER": 1.2,   # Red Bull Ford — best raw straight speed
    "SAI": 0.9,   # Mercedes PU customer
    "COL": 0.8,   # Renault PU
}
qualifying_2026["BakuStraightSpeed"] = qualifying_2026["DriverCode"].map(
    STRAIGHT_SPEED
)

# ══════════════════════════════════════════════════════════
# 9. SAFETY CAR PROBABILITY — NEW FEATURE 🆕
#    Baku almost always has safety car
#    Affects race outcome significantly
#    Drivers with better restart ability benefit
#    Experience at Baku matters
# ══════════════════════════════════════════════════════════
SAFETY_CAR_SKILL = {
    "RUS": 0.85,  # Good SC restart
    "LEC": 0.90,  # Excellent Baku experience
    "PIA": 0.80,
    "HAD": 0.75,  # Limited Baku experience
    "NOR": 0.85,
    "HAM": 0.92,  # Elite SC restarts
    "GAS": 0.82,
    "VER": 0.90,  # Won Baku before
    "SAI": 0.85,  # Good Baku history
    "COL": 0.78,  # Limited Baku F1 data
}
qualifying_2026["SafetyCarSkill"] = qualifying_2026["DriverCode"].map(
    SAFETY_CAR_SKILL
)

# ══════════════════════════════════════════════════════════
# 10. TYRE DEG — 24°C cool street circuit
# ══════════════════════════════════════════════════════════
TYRE_DEG = {
    "Mercedes":        2.0,
    "Ferrari":         2.0,  # Good at Baku
    "McLaren":         1.5,
    "Red Bull Racing": 2.0,
    "Alpine":          3.0,
    "Williams":        3.0,
    "Racing Bulls":    3.0,
    "Audi":            3.5,
    "Haas":            3.5,
    "Aston Martin":    3.0,
    "Cadillac":        4.5,
}
qualifying_2026["TyreDegScore"] = qualifying_2026["Team"].map(TYRE_DEG)

# ══════════════════════════════════════════════════════════
# 11. ERS DEPENDENCY (7MJ — critical on 2.2km straight)
# ══════════════════════════════════════════════════════════
ERS_DEPENDENCY = {
    "Mercedes":        9.0,
    "McLaren":         9.0,
    "Ferrari":         6.5,
    "Red Bull Racing": 5.5,   # Most benefit at Baku
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
# 12. HOME RACE BOOST — No specific home drivers in Q3
# ══════════════════════════════════════════════════════════
qualifying_2026["HomeRaceBoost"] = 0.0

# ══════════════════════════════════════════════════════════
# 13. RELIABILITY RISK
#     Baku is hard on cars — street circuit
# ══════════════════════════════════════════════════════════
RELIABILITY_RISK = {
    "Mercedes":        4.5,
    "McLaren":         2.5,
    "Ferrari":         2.5,
    "Red Bull Racing": 3.0,
    "Alpine":          4.0,
    "Williams":        4.0,
    "Racing Bulls":    3.5,
    "Audi":            5.0,
    "Haas":            4.0,
    "Aston Martin":    4.0,
    "Cadillac":        5.0,
}
qualifying_2026["ReliabilityRiskScore"] = qualifying_2026["Team"].map(
    RELIABILITY_RISK
)

# ══════════════════════════════════════════════════════════
# 14. CIRCUIT SCORE — 14 ROUNDS OF 2026 DATA
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS  CHN  JPN  MIA  CAN  MON  ESP  AUT  GBR  BEL  HUN  NED  ITA  MAD
    "RUS":              [1,   2,   4,   4,   20,  20,  2,   1,   2,   20,  7,   3,   2,   5],
    "LEC":              [3,   4,   3,   6,   4,   20,  20,  20,  1,   2,   4,   5,   20,  4],
    "PIA":              [22,  2,   2,   3,   20,  4,   20,  4,   20,  5,   20,  6,   5,   8],
    "HAD":              [20,  8,   9,   20,  5,   3,   6,   20,  5,   6,   20,  20,  20,  20],
    "NOR":              [5,   20,  5,   2,   20,  20,  3,   20,  4,   7,   1,   1,   4,   3],
    "HAM":              [7,   3,   6,   7,   2,   2,   1,   5,   3,   4,   5,   4,   6,   20],
    "GAS":              [10,  6,   10,  20,  8,   7,   20,  20,  20,  20,  12,  10,  7,   20],
    "VER":              [20,  20,  8,   5,   3,   20,  5,   2,   20,  3,   2,   20,  3,   2],
    "SAI":              [20,  20,  20,  9,   9,   20,  20,  20,  9,   20,  13,  20,  20,  20],
    "COL":              [20,  10,  20,  8,   20,  20,  20,  20,  20,  10,  15,  20,  9,   7],
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
# 15. SYNTHETIC SECTOR TIMES
#     Baku split ratios (approximate)
#     S1: 28% — castle section (technical)
#     S2: 38% — middle section
#     S3: 34% — long Caspian straight
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.28
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.38
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.34
# Baku race pace ~6% slower (safety cars, traffic)
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.06

# ══════════════════════════════════════════════════════════
# 16. WEATHER FEATURES
# ══════════════════════════════════════════════════════════
qualifying_2026["RainProbability"] = RAIN_PROBABILITY
qualifying_2026["Temperature"]     = RACE_TEMP
qualifying_2026["TempDelta"]       = TEMP_DELTA
qualifying_2026["Humidity"]        = HUMIDITY
qualifying_2026["WindSpeed"]       = WIND_SPEED

print("\n📊 Full Feature Set:")
print(qualifying_2026[[
    "Driver", "QualifyingTime (s)", "GapFromPole (s)",
    "AdjustedTeamScore", "BakuStraightSpeed",
    "SafetyCarSkill", "CircuitScore"
]].to_string(index=False))

# ══════════════════════════════════════════════════════════
# 17. FEATURE COLUMNS
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
    "ERSDependencyScore",     # critical on 2.2km straight
    "BakuGridPenalty",
    "TyreDegScore",
    "BakuStraightSpeed",      # 🆕 straight line speed advantage
    "SafetyCarSkill",         # 🆕 safety car restart ability
    "HomeRaceBoost",
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",           # 14 rounds of 2026 data
    "SprintWinnerBoost",
]
TARGET = "RacePace (s)"

# ══════════════════════════════════════════════════════════
# 18. TRAIN MODEL
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
# 19. PREDICT RACE
# ══════════════════════════════════════════════════════════
data = qualifying_2026.copy()
data["PredictedLapTime (s)"] = model.predict(X)

# Baku grid penalty
data["PredictedLapTime (s)"] += data["BakuGridPenalty"] * 0.35

# Straight line speed bonus
data["PredictedLapTime (s)"] -= data["BakuStraightSpeed"] * 0.03

# Safety car skill bonus
data["PredictedLapTime (s)"] -= data["SafetyCarSkill"] * 0.02

# Tyre deg
data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.01

# Sort
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 20. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 AZERBAIJAN GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  Race: {RACE_TEMP}°C ☀️")
print(f"  💨  Strong qualifying winds — calmer race day")
print(f"  🔋  ERS: 7MJ — most critical on 2.2km straight!")
print(f"  🏰  Baku safety car almost certain")
print(f"  😱  Antonelli starts OUTSIDE top 10!")
print(f"  🌟  Pole: Russell — 3rd pole of 2026!\n")

# ══════════════════════════════════════════════════════════
# 21. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 15: AZERBAIJAN GP\n"
    "BAKU CITY CIRCUIT  |  SEP 26, 2026  |  ☀️ 24°C  |  🏰 STREET CIRCUIT",
    fontsize=16, fontweight="bold", color="white",
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
    "📊 Predicted Race Finishing Order  (🏰 Baku — Safety Car Almost Certain!)",
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

# ── Chart 2: Baku Straight Speed ─────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ss_sorted = data.sort_values("BakuStraightSpeed", ascending=False)
ss_colors = [TEAM_COLORS.get(t, "#FFF") for t in ss_sorted["Team"]]
ax2.barh(
    ss_sorted["Driver"][::-1],
    ss_sorted["BakuStraightSpeed"][::-1],
    color=ss_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title(
    "🚀 Baku Straight Speed Score\n(higher = faster on 2.2km Caspian straight)",
    fontsize=10, fontweight="bold", color="white",
    fontfamily=FONT, pad=10
)
ax2.set_xlabel("Straight Speed Score",
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
ax3.set_title("⏱️  Qualifying Gap to Pole (Strong Wind Q3)",
              fontsize=10, fontweight="bold", color="white",
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
    "Humidity", "Wind Speed", "ERS Dependency ⚡",
    "Baku Grid", "Tyre Deg",
    "Straight Speed 🆕", "SC Skill 🆕",
    "Home Boost", "Reliability",
    "Sector 1", "Sector 2", "Sector 3",
    "Circuit Score", "Sprint Boost"
]
feat_import   = model.feature_importances_
sorted_idx    = np.argsort(feat_import)
sorted_labels = [feat_labels[i] for i in sorted_idx]
sorted_values = feat_import[sorted_idx]
colors_bar    = plt.cm.Blues(np.linspace(0.3, 0.95, len(sorted_values)))
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
ax5.set_title("🏆 Predicted Podium  🇦🇿",
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
    f"🔋 ERS: 7MJ — 2.2km straight  |  "
    f"🏰 Safety car likely  |  "
    f"😱 Antonelli outside top 10!",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_15_azerbaijan_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_15_azerbaijan_prediction.png")
plt.show()