"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 12: DUTCH GP
 Zandvoort Circuit | Race Date: August 23, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.07 (7% slower)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            ZandvoortGridPenalty, TyreDegScore,
            ReliabilityRiskScore, CircuitScore,
            SprintWinnerBoost, HomeRaceBoost
 Upgrades vs R11:
            + Russell SprintWinnerBoost + HomeRaceBoost
            + Norris pole — McLaren strong at Zandvoort
            + Verstappen home race boost
            + Tsunoda back in Racing Bulls
            + Lawson now in Red Bull (swap with Hadjar)
            + 11 rounds of 2026 CircuitScore data
            + Dry race (14%) vs wet qualifying (34%)
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 12: DUTCH GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 19    # °C — drizzle Saturday
RACE_TEMP        = 20    # °C — sunny intervals Sunday
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # +1°C
RAIN_PROBABILITY = 0.14  # 14% race day — essentially dry
HUMIDITY         = 70    # % estimated
WIND_SPEED       = 15    # km/h — coastal Zandvoort wind

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C 🌦️  →  Race: {RACE_TEMP}°C ⛅  (Δ+{TEMP_DELTA}°C)")
print(f"💨  Coastal wind: {WIND_SPEED}km/h — North Sea factor")
print(f"🌧️  Race rain: {int(RAIN_PROBABILITY*100)}% — essentially dry")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Norris POLE — McLaren flying at Zandvoort! 🟠
#    Top 7 covered by only 0.455s — incredibly tight!
#    Tsunoda back — Lawson now in Red Bull
# ══════════════════════════════════════════════════════════
POLE_TIME = 71.163  # Norris 1:11.163

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Lando Norris",
        "George Russell",
        "Kimi Antonelli",
        "Oscar Piastri",
        "Lewis Hamilton",
        "Charles Leclerc",
        "Max Verstappen",
        "Liam Lawson",
        "Gabriel Bortoleto",
        "Arvid Lindblad",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        71.163,   # 1:11.163 — Norris POLE 🌟
        71.265,   # 1:11.265 — Russell     +0.102s
        71.296,   # 1:11.296 — Antonelli   +0.133s
        71.305,   # 1:11.305 — Piastri     +0.142s
        71.494,   # 1:11.494 — Hamilton    +0.331s
        71.558,   # 1:11.558 — Leclerc     +0.395s
        71.618,   # 1:11.618 — Verstappen  +0.455s
        71.733,   # 1:11.733 — Lawson      +0.570s
        72.079,   # 1:12.079 — Bortoleto   +0.916s
        72.185,   # 1:12.185 — Lindblad    +1.022s
    ],
    "Team": [
        "McLaren",  "Mercedes",        "Mercedes",
        "McLaren",  "Ferrari",          "Ferrari",
        "Red Bull Racing", "Red Bull Racing", "Audi",
        "Racing Bulls",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    # Russell won Sprint 🏆
    "SprintWinnerBoost": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Lando Norris":     "NOR",
    "George Russell":   "RUS",
    "Kimi Antonelli":   "ANT",
    "Oscar Piastri":    "PIA",
    "Lewis Hamilton":   "HAM",
    "Charles Leclerc":  "LEC",
    "Max Verstappen":   "VER",
    "Liam Lawson":      "LAW",
    "Gabriel Bortoleto":"BOR",
    "Arvid Lindblad":   "LIN",
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
# 4. ADJUSTED TEAM SCORE — updated after 11 rounds
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.5,  # Russell Sprint win + P2 quali
    "McLaren":         9.3,  # Norris pole + Hungary win — surging!
    "Ferrari":         8.8,  # Consistent podiums all season
    "Red Bull Racing": 7.0,  # Verstappen P2 Hungary — improving
    "Racing Bulls":    6.0,  # Lindblad consistently in Q3
    "Audi":            5.5,  # Bortoleto P9 quali
    "Alpine":          5.0,
    "Haas":            4.5,
    "Aston Martin":    4.5,
    "Williams":        4.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR
#    Race is dry (14%) but qualifying was wet (34%)
#    Wet factor minimal for race prediction
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "NOR": 0.976,
    "RUS": 0.966,   # Elite wet — won Sprint in drizzle
    "ANT": 0.972,
    "PIA": 0.975,
    "HAM": 0.964,
    "LEC": 0.974,
    "VER": 0.966,   # Legendary Zandvoort wet history
    "LAW": 0.979,
    "BOR": 0.980,
    "LIN": 0.983,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — 14% rain < 60% threshold = zero
# ══════════════════════════════════════════════════════════
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY if (
        p == 1 and RAIN_PROBABILITY >= 0.60
    ) else 0.0
)

# ══════════════════════════════════════════════════════════
# 7. ZANDVOORT GRID PENALTY
#    Very hard to overtake — narrow banked circuit
#    Similar to Hungary but slightly easier than Monaco
#    0.12s per position penalty
# ══════════════════════════════════════════════════════════
qualifying_2026["ZandvoortGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.12
)

# ══════════════════════════════════════════════════════════
# 8. HOME RACE BOOST
#    Verstappen — Zandvoort is literally his home circuit
#    Russell + Norris — British drivers (mild boost)
# ══════════════════════════════════════════════════════════
HOME_BOOST = {
    "NOR": 0.5,    # British driver mild boost
    "RUS": 1.0,    # British driver + Sprint winner
    "ANT": 0.0,
    "PIA": 0.0,
    "HAM": 0.5,    # British driver
    "LEC": 0.0,
    "VER": 2.0,    # HOME RACE — Zandvoort is Max's circuit!
    "LAW": 0.0,
    "BOR": 0.0,
    "LIN": 0.5,    # British driver
}
qualifying_2026["HomeRaceBoost"] = qualifying_2026["DriverCode"].map(
    HOME_BOOST
)

# ══════════════════════════════════════════════════════════
# 9. TYRE DEG SCORE — 20°C cool race
# ══════════════════════════════════════════════════════════
TYRE_DEG = {
    "Mercedes":        2.0,
    "Ferrari":         2.5,
    "McLaren":         1.5,
    "Red Bull Racing": 2.0,
    "Racing Bulls":    3.0,
    "Audi":            3.5,
    "Alpine":          3.0,
    "Haas":            3.5,
    "Aston Martin":    2.5,
    "Williams":        3.5,
    "Cadillac":        4.5,
}
qualifying_2026["TyreDegScore"] = qualifying_2026["Team"].map(TYRE_DEG)

# ══════════════════════════════════════════════════════════
# 10. ERS DEPENDENCY (7MJ)
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
# 11. RELIABILITY RISK
# ══════════════════════════════════════════════════════════
RELIABILITY_RISK = {
    "Mercedes":        4.5,  # Antonelli 2x DNF still flagged
    "McLaren":         2.5,
    "Ferrari":         2.0,
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
# 12. CIRCUIT SCORE — 11 ROUNDS OF 2026 DATA
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS  CHN  JPN  MIA  CAN  MON  ESP  AUT  GBR  BEL  HUN
    "NOR":              [5,   20,  5,   2,   20,  20,  3,   20,  4,   7,   1],
    "RUS":              [1,   2,   4,   4,   20,  20,  2,   1,   2,   20,  7],
    "ANT":              [2,   1,   1,   1,   1,   1,   4,   3,   20,  1,   3],
    "PIA":              [22,  2,   2,   3,   20,  4,   20,  4,   20,  5,   20],
    "HAM":              [7,   3,   6,   7,   2,   2,   1,   5,   3,   4,   5],
    "LEC":              [3,   4,   3,   6,   4,   20,  20,  20,  1,   2,   4],
    "VER":              [20,  20,  8,   5,   3,   20,  5,   2,   20,  3,   2],
    "LAW":              [20,  20,  20,  20,  7,   5,   20,  20,  8,   20,  8],
    "BOR":              [20,  20,  20,  20,  20,  20,  20,  8,   8,   20,  20],
    "LIN":              [8,   20,  14,  20,  6,   6,   20,  20,  7,   9,   10],
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
#     Zandvoort split ratios (approximate)
#     S1: 32% — start to Hugenholtz
#     S2: 38% — middle section
#     S3: 30% — final banked corner to finish
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.32
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.38
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
    "AdjustedTeamScore", "HomeRaceBoost",
    "ZandvoortGridPenalty", "CircuitScore"
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
    "ZandvoortGridPenalty",  # narrow circuit — hard to overtake
    "TyreDegScore",
    "HomeRaceBoost",         # VER home race + RUS/NOR/HAM British
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",          # 11 rounds of 2026 data
    "SprintWinnerBoost",     # Russell won Sprint
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

# Zandvoort grid penalty
data["PredictedLapTime (s)"] += data["ZandvoortGridPenalty"] * 0.4

# Tyre deg — cool 20°C minimal
data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.01

# Home race boost
data["PredictedLapTime (s)"] -= data["HomeRaceBoost"] * 0.06

# Wet bonus — minimal at 14%
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
print("  🏁  2026 DUTCH GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Qualifying: {QUALIFYING_TEMP}°C 🌦️  →  Race: {RACE_TEMP}°C ⛅")
print(f"  🌧️  Race rain: {int(RAIN_PROBABILITY*100)}% — dry race expected")
print(f"  💨  Coastal wind: {WIND_SPEED}km/h — North Sea factor")
print(f"  🔋  ERS: 7MJ  |  ⚡ Boost cap: +150kW")
print(f"  🏆  Sprint winner: George Russell")
print(f"  🟠  Pole: Norris — McLaren flying at Zandvoort!")
print(f"  🔵  Verstappen home race — orange army LOUD!\n")

# ══════════════════════════════════════════════════════════
# 19. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 12: DUTCH GP\n"
    "ZANDVOORT CIRCUIT  |  AUGUST 23, 2026  |  ⛅ 20°C  |  💨 COASTAL WIND",
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
    "📊 Predicted Race Finishing Order  (🔵 Verstappen Home Race — Orange Army!)",
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

# ── Chart 2: Home Race Boost ──────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
home_sorted = qualifying_2026.sort_values("HomeRaceBoost", ascending=False)
home_colors = [TEAM_COLORS.get(t, "#FFF") for t in home_sorted["Team"]]
ax2.barh(
    home_sorted["Driver"][::-1],
    home_sorted["HomeRaceBoost"][::-1],
    color=home_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title(
    "🏠 Home Race Boost\n(VER 2.0x — Zandvoort his circuit! RUS 1.0x Sprint winner)",
    fontsize=9, fontweight="bold", color="white",
    fontfamily=FONT, pad=10
)
ax2.set_xlabel("Home Race Boost Score",
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
    "Zandvoort Grid", "Tyre Deg",
    "Home Boost 🏠", "Reliability",
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
ax5.set_title("🏆 Predicted Podium  🇳🇱",
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
    f"⛅ Race: {RACE_TEMP}°C  |  "
    f"💨 Wind: {WIND_SPEED}km/h  |  "
    f"🔋 ERS: 7MJ  |  "
    f"🏆 Sprint: Russell  |  "
    f"🟠 Pole: Norris  |  "
    f"🔵 VER home race boost 2.0x",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_12_netherlands_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_12_netherlands_prediction.png")
plt.show()