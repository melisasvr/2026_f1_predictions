"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 10: BELGIAN GP
 Spa-Francorchamps | Race Date: July 19, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.07 (7% slower)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            SpaGridPenalty, TyreDegScore,
            ColdTyreScore, ReliabilityRiskScore,
            CircuitScore, SprintWinnerBoost,
            HomeRaceBoost, SpaWetHistory
 Upgrades vs R09:
            + Russell now included in HomeRaceBoost (lesson!)
            + ColdTyreScore — 20°C coldest race of season
            + SpaWetHistory — drivers wet record at Spa
            + Bortoleto P9 — Audi best quali of season
            + Lindblad P8 — rookie excelling
            + 9 rounds of 2026 CircuitScore data
            + Verstappen + Norris historically strong at Spa
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 10: BELGIAN GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 23    # °C — cloudy
RACE_TEMP        = 20    # °C — sunny/cloudy — COLDEST of season!
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # -3°C
RAIN_PROBABILITY = 0.20  # 20% race day — fairly dry
HUMIDITY         = 70    # % race day
WIND_SPEED       = 18    # km/h — strongest wind of season

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ☁️  →  Race: {RACE_TEMP}°C ⛅  (Δ{TEMP_DELTA}°C)")
print(f"❄️  COLDEST RACE OF 2026 — 20°C tyre warm-up challenge!")
print(f"💨  Wind: {WIND_SPEED}km/h — strongest of season at Spa")
print(f"🌧️  Rain: {int(RAIN_PROBABILITY*100)}% — low but Spa micro-climate unpredictable")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Antonelli — 7th pole of the season! 🌟
#    Verstappen P2 — Spa suits him perfectly
#    Ferrari only P5+P6 — unusually far back
# ══════════════════════════════════════════════════════════
POLE_TIME = 104.361  # Antonelli 1:44.361

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Kimi Antonelli",
        "Max Verstappen",
        "Lando Norris",
        "George Russell",
        "Charles Leclerc",
        "Lewis Hamilton",
        "Oscar Piastri",
        "Arvid Lindblad",
        "Gabriel Bortoleto",
        "Isack Hadjar",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        104.361,  # 1:44.361 — Antonelli POLE 🌟
        104.678,  # 1:44.678 — Verstappen  +0.317s
        104.801,  # 1:44.801 — Norris      +0.440s
        104.869,  # 1:44.869 — Russell     +0.508s
        104.893,  # 1:44.893 — Leclerc     +0.532s
        104.895,  # 1:44.895 — Hamilton    +0.534s
        105.016,  # 1:45.016 — Piastri     +0.655s
        105.143,  # 1:45.143 — Lindblad    +0.782s
        105.628,  # 1:45.628 — Bortoleto   +1.267s
        105.823,  # 1:45.823 — Hadjar      +1.462s
    ],
    "Team": [
        "Mercedes", "Red Bull Racing", "McLaren",
        "Mercedes", "Ferrari",         "Ferrari",
        "McLaren",  "Racing Bulls",    "Audi",
        "Red Bull Racing",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "SprintWinnerBoost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Kimi Antonelli":   "ANT",
    "Max Verstappen":   "VER",
    "Lando Norris":     "NOR",
    "George Russell":   "RUS",
    "Charles Leclerc":  "LEC",
    "Lewis Hamilton":   "HAM",
    "Oscar Piastri":    "PIA",
    "Arvid Lindblad":   "LIN",
    "Gabriel Bortoleto":"BOR",
    "Isack Hadjar":     "HAD",
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
# 4. ADJUSTED TEAM SCORE — updated after 9 rounds
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.7,  # Dominant — 6 wins but Antonelli 2x DNF concern
    "Ferrari":         9.0,  # Hamilton Spain + Leclerc Britain — surging!
    "McLaren":         8.5,  # Norris P3 Spa quali — strong at power circuits
    "Red Bull Racing": 7.5,  # VER P2 Spa — this is his circuit!
    "Racing Bulls":    6.0,  # Lindblad P8 best quali yet — both teams scoring
    "Audi":            5.5,  # Bortoleto P9 — best quali of 2026!
    "Alpine":          5.0,
    "Haas":            4.5,
    "Aston Martin":    4.0,
    "Williams":        4.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR
#    20% rain but Spa micro-climate — could change instantly
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "ANT": 0.972,
    "VER": 0.966,  # LEGENDARY at Spa in wet — 2021 etc
    "NOR": 0.976,
    "RUS": 0.966,
    "LEC": 0.974,
    "HAM": 0.964,
    "PIA": 0.975,
    "LIN": 0.983,
    "BOR": 0.980,
    "HAD": 0.980,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — 20% rain < 60% threshold
# ══════════════════════════════════════════════════════════
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY if (
        p == 1 and RAIN_PROBABILITY >= 0.60
    ) else 0.0
)

# ══════════════════════════════════════════════════════════
# 7. SPA GRID PENALTY
#    Spa has good overtaking at Les Combes, Bus Stop chicane
#    Medium overtaking difficulty — 0.09s per position
# ══════════════════════════════════════════════════════════
qualifying_2026["SpaGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.09
)

# ══════════════════════════════════════════════════════════
# 8. TYRE DEG SCORE — 20°C COLDEST RACE OF 2026
#    Cold tyres take longer to warm up
#    Teams that manage cold tyre performance better benefit
# ══════════════════════════════════════════════════════════
TYRE_DEG = {
    "Mercedes":        2.0,
    "Ferrari":         2.5,
    "McLaren":         1.5,  # Best tyre management
    "Red Bull Racing": 2.0,  # Good cold tyre management
    "Racing Bulls":    3.0,
    "Audi":            3.5,
    "Alpine":          3.0,
    "Haas":            3.5,
    "Aston Martin":    3.0,
    "Williams":        3.5,
    "Cadillac":        4.5,
}
qualifying_2026["TyreDegScore"] = qualifying_2026["Team"].map(TYRE_DEG)

# ══════════════════════════════════════════════════════════
# 9. COLD TYRE SCORE — NEW FEATURE 🆕
#    20°C is significantly colder than rest of season
#    Drivers/teams with better cold tyre warm-up benefit
#    Lower = better cold tyre performance
# ══════════════════════════════════════════════════════════
COLD_TYRE = {
    "ANT": 2.0,
    "VER": 1.5,  # Red Bull historically great in cold
    "NOR": 2.0,
    "RUS": 2.0,
    "LEC": 2.5,
    "HAM": 2.0,  # Experienced in cold conditions
    "PIA": 2.0,
    "LIN": 3.0,
    "BOR": 3.0,
    "HAD": 2.5,
}
qualifying_2026["ColdTyreScore"] = qualifying_2026["DriverCode"].map(
    COLD_TYRE
)

# ══════════════════════════════════════════════════════════
# 10. SPA WET HISTORY — NEW FEATURE 🆕
#     Driver historical performance at Spa in wet/mixed
#     Lower = better Spa wet record
# ══════════════════════════════════════════════════════════
SPA_WET_HISTORY = {
    "ANT": 2.5,   # Limited Spa F1 history
    "VER": 1.0,   # LEGENDARY at Spa — multiple wins inc. wet
    "NOR": 2.5,   # Good Spa record
    "RUS": 2.5,   # Decent Spa history
    "LEC": 2.0,   # Strong Spa performer
    "HAM": 1.5,   # Multiple Spa wins — elite
    "PIA": 3.0,   # Limited Spa F1 history
    "LIN": 4.0,   # Rookie — no Spa F1 data
    "BOR": 3.5,   # Limited Spa F1 history
    "HAD": 3.0,   # Limited Spa F1 history
}
qualifying_2026["SpaWetHistory"] = qualifying_2026["DriverCode"].map(
    SPA_WET_HISTORY
)

# ══════════════════════════════════════════════════════════
# 11. HOME RACE BOOST
#     No specific home race drivers at Spa
#     Gasly is French — not at Spa (out in Q2)
#     Keeping feature at 0 for all drivers
# ══════════════════════════════════════════════════════════
qualifying_2026["HomeRaceBoost"] = 0.0

# ══════════════════════════════════════════════════════════
# 12. ERS DEPENDENCY (7MJ — longest lap = most impactful)
#     Spa is 7km — the ERS depletes over a VERY long lap
#     This makes the 7MJ limit more significant here
# ══════════════════════════════════════════════════════════
ERS_DEPENDENCY = {
    "Mercedes":        9.0,
    "McLaren":         9.0,
    "Ferrari":         6.5,
    "Red Bull Racing": 5.5,  # Ford PU benefits most at Spa
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
# 13. RELIABILITY RISK — updated after Antonelli 2x DNF
# ══════════════════════════════════════════════════════════
RELIABILITY_RISK = {
    "Mercedes":        4.0,  # Antonelli 2x DNF — elevated concern
    "Ferrari":         2.0,
    "Red Bull Racing": 3.0,
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
# 14. CIRCUIT SCORE — 9 ROUNDS OF 2026 DATA
#     AUS+CHN+JPN+MIA+CAN+MON+ESP+AUT+GBR
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS  CHN  JPN  MIA  CAN  MON  ESP  AUT  GBR
    "ANT":              [2,   1,   1,   1,   1,   1,   4,   3,   20],
    "VER":              [20,  20,  8,   5,   3,   20,  5,   2,   20],
    "NOR":              [5,   20,  5,   2,   20,  20,  3,   20,  4],
    "RUS":              [1,   2,   4,   4,   20,  20,  2,   1,   2],
    "LEC":              [3,   4,   3,   6,   4,   20,  20,  20,  1],
    "HAM":              [7,   3,   6,   7,   2,   2,   1,   5,   3],
    "PIA":              [22,  2,   2,   3,   20,  4,   20,  4,   20],
    "LIN":              [8,   20,  14,  20,  6,   6,   20,  20,  7],
    "BOR":              [20,  20,  20,  20,  20,  20,  20,  8,   8],
    "HAD":              [20,  8,   9,   20,  5,   3,   6,   20,  5],
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
#     Spa-Francorchamps split ratios (approximate)
#     S1: 28% — Eau Rouge/Raidillon to Les Combes
#     S2: 42% — Pouhon to Paul Frere
#     S3: 30% — Blanchimont to Bus Stop chicane
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.28
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.42
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.30
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.07

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
    "AdjustedTeamScore", "ColdTyreScore",
    "SpaWetHistory", "CircuitScore"
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
    "ERSDependencyScore",
    "SpaGridPenalty",
    "TyreDegScore",
    "ColdTyreScore",         # 🆕 20°C coldest race of season
    "SpaWetHistory",         # 🆕 driver Spa wet history
    "HomeRaceBoost",
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",          # 9 rounds of 2026 data
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

# Spa grid penalty
data["PredictedLapTime (s)"] += data["SpaGridPenalty"] * 0.4

# Cold tyre — 20°C hurts teams with poor cold tyre management
data["PredictedLapTime (s)"] += data["ColdTyreScore"] * 0.03

# Spa wet history bonus
data["PredictedLapTime (s)"] += (data["SpaWetHistory"] - 1) * 0.02

# Tyre deg — milder at 20°C
data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.01

# Wet bonus — 20% rain
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
# 20. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 BELGIAN GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Qualifying: {QUALIFYING_TEMP}°C ☁️  →  Race: {RACE_TEMP}°C ⛅")
print(f"  ❄️  COLDEST RACE of 2026 — 20°C cold tyre challenge!")
print(f"  💨  Wind: {WIND_SPEED}km/h — strongest of season")
print(f"  🌧️  Rain: {int(RAIN_PROBABILITY*100)}% — Spa micro-climate unpredictable")
print(f"  🔋  ERS: 7MJ — longest lap, most impactful cut")
print(f"  🌟  Pole: Antonelli — 7th of 2026 season!")
print(f"  🔵  Verstappen P2 — this is his circuit!\n")

# ══════════════════════════════════════════════════════════
# 21. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 10: BELGIAN GP\n"
    "SPA-FRANCORCHAMPS  |  JULY 27, 2026  |  ❄️ 20°C  |  💨 18km/h WIND",
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
    "📊 Predicted Race Finishing Order  (❄️ Coldest Race of 2026 — 20°C)",
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

# ── Chart 2: Spa Wet History ──────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
spa_sorted  = data.sort_values("SpaWetHistory")
spa_colors  = [TEAM_COLORS.get(t, "#FFF") for t in spa_sorted["Team"]]
ax2.barh(
    spa_sorted["Driver"][::-1],
    spa_sorted["SpaWetHistory"][::-1],
    color=spa_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title(
    "🌧️  Spa Wet History Score\n(lower = better Spa wet record)",
    fontsize=10, fontweight="bold", color="white",
    fontfamily=FONT, pad=10
)
ax2.set_xlabel("Spa Wet History (lower = better)",
               color="#AAAAAA", fontsize=8, fontfamily=FONT)
ax2.tick_params(colors="white", labelsize=8)
ax2.set_facecolor("#1a1a1a")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333333")
for i, (_, row) in enumerate(spa_sorted[::-1].iterrows()):
    ax2.text(
        row["SpaWetHistory"] + 0.05, i,
        f"{row['SpaWetHistory']:.1f}",
        va="center", fontsize=8,
        color="white", fontfamily=FONT
    )

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
    "Humidity", "Wind Speed", "ERS Dependency",
    "Spa Grid", "Tyre Deg", "Cold Tyre 🆕",
    "Spa Wet 🆕", "Home Boost", "Reliability",
    "Sector 1", "Sector 2", "Sector 3",
    "Circuit Score", "Sprint Boost"
]
feat_import   = model.feature_importances_
sorted_idx    = np.argsort(feat_import)
sorted_labels = [feat_labels[i] for i in sorted_idx]
sorted_values = feat_import[sorted_idx]
colors_bar    = plt.cm.Greens(np.linspace(0.3, 0.95, len(sorted_values)))
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
ax5.set_title("🏆 Predicted Podium  🇧🇪",
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
    f"❄️ Race: {RACE_TEMP}°C  |  "
    f"💨 Wind: {WIND_SPEED}km/h  |  "
    f"🔋 ERS: 7MJ  |  "
    f"🌟 Pole: Antonelli (7th!)  |  "
    f"🔵 VER P2 — Spa is his circuit",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_10_belgium_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_10_belgium_prediction.png")
plt.show()