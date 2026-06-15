"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 7: SPANISH GP
 Circuit de Barcelona-Catalunya | Race Date: June 15, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.07 (7% slower)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            BoostCapScore, ReliabilityRiskScore,
            CircuitScore, SprintWinnerBoost,
            BarcelonaOvertakingScore
 Upgrades vs R06:
            + Dry hot race — tyre degradation key feature
            + Russell pole after 5 Antonelli poles — momentum shift
            + Leclerc P10 — penalty or pace issue factored
            + Hülkenberg P9 — Audi best result
            + 6 rounds of 2026 CircuitScore data
            + BarcelonaOvertakingScore — medium difficulty
 Author: Melisa Sever
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 7: SPANISH GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER — Hot dry weekend
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 29    # °C — partly cloudy
RACE_TEMP        = 29    # °C — sunny
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # 0°C
RAIN_PROBABILITY = 0.05  # essentially dry
HUMIDITY         = 45    # % estimated
WIND_SPEED       = 10    # km/h

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ⛅  →  Race: {RACE_TEMP}°C ☀️  (Δ{TEMP_DELTA}°C)")
print(f"☀️  Rain: {int(RAIN_PROBABILITY*100)}% — dry hot race")
print(f"🔥  29°C sunny — tyre degradation will be significant")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Russell breaks Antonelli's pole streak! 🌟
#    Leclerc P10 — surprise after Monaco P4
# ══════════════════════════════════════════════════════════
POLE_TIME = 74.679  # Russell 1:14.679

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "George Russell",
        "Lewis Hamilton",
        "Kimi Antonelli",
        "Lando Norris",
        "Max Verstappen",
        "Isack Hadjar",
        "Oscar Piastri",
        "Liam Lawson",
        "Nico Hulkenberg",
        "Charles Leclerc",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        74.679,   # 1:14.679 — Russell POLE 🌟
        74.743,   # 1:14.743 — Hamilton   +0.064s
        74.998,   # 1:14.998 — Antonelli  +0.319s
        75.001,   # 1:15.001 — Norris     +0.322s
        75.021,   # 1:15.021 — Verstappen +0.342s
        75.077,   # 1:15.077 — Hadjar     +0.398s
        75.090,   # 1:15.090 — Piastri    +0.411s
        76.542,   # 1:16.542 — Lawson     +1.863s
        76.657,   # 1:16.657 — Hulkenberg +1.978s
        75.281,   # 1:15.281 — Leclerc    +0.602s
    ],
    "Team": [
        "Mercedes", "Ferrari",        "Mercedes",
        "McLaren",  "Red Bull Racing", "Red Bull Racing",
        "McLaren",  "Racing Bulls",    "Audi",
        "Ferrari",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "SprintWinnerBoost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "George Russell":  "RUS",
    "Lewis Hamilton":  "HAM",
    "Kimi Antonelli":  "ANT",
    "Lando Norris":    "NOR",
    "Max Verstappen":  "VER",
    "Isack Hadjar":    "HAD",
    "Oscar Piastri":   "PIA",
    "Liam Lawson":     "LAW",
    "Nico Hulkenberg": "HUL",
    "Charles Leclerc": "LEC",
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
# 4. ADJUSTED TEAM SCORE — updated after 6 rounds
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.8,  # Dominant — 5 wins, Russell pole Spain
    "Ferrari":         8.5,  # Hamilton consistently P2/P3, strong pace
    "McLaren":         8.0,  # Norris P4 Spain quali — solid
    "Red Bull Racing": 7.5,  # VER improving, Hadjar P3 Monaco
    "Racing Bulls":    5.5,  # Lawson P5 Canada, Lindblad P6 Monaco
    "Audi":            5.0,  # Hulkenberg P9 — best result of 2026!
    "Alpine":          5.0,  # Gasly consistent scorer
    "Haas":            4.5,
    "Aston Martin":    4.0,
    "Williams":        4.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR — minimal impact (dry race)
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "RUS": 0.966,
    "HAM": 0.964,
    "ANT": 0.972,
    "NOR": 0.976,
    "VER": 0.968,
    "HAD": 0.980,
    "PIA": 0.975,
    "LAW": 0.979,
    "HUL": 0.978,
    "LEC": 0.974,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — essentially zero (5% rain, <60%)
# ══════════════════════════════════════════════════════════
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY if (
        p == 1 and RAIN_PROBABILITY >= 0.60
    ) else 0.0
)

# ══════════════════════════════════════════════════════════
# 7. BARCELONA OVERTAKING SCORE — NEW FEATURE 🆕
#    Barcelona is medium difficulty for overtaking
#    Long main straight (DRS/Boost zone) allows some moves
#    But T1 braking is the main opportunity
#    Less penalty than Monaco, more than Suzuka
#    Scale: penalty per grid position back (+0.08s)
# ══════════════════════════════════════════════════════════
qualifying_2026["BarcelonaGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.08
)

# ══════════════════════════════════════════════════════════
# 8. TYRE DEG SCORE — HOT DRY BARCELONA
#    29°C sunny — significant tyre degradation expected
#    Teams with better tyre management get advantage
#    Lower = better tyre management
# ══════════════════════════════════════════════════════════
TYRE_DEG = {
    "Mercedes":        2.0,  # Strong tyre management
    "Ferrari":         2.5,  # Good but Barcelona can expose deg
    "McLaren":         1.5,  # Best tyre management on grid
    "Red Bull Racing": 2.5,  # Medium tyre deg
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
# 9. ERS DEPENDENCY (7MJ limit)
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
# 10. RELIABILITY RISK (updated — Russell DNF Canada noted)
# ══════════════════════════════════════════════════════════
RELIABILITY_RISK = {
    "Mercedes":        3.0,  # Russell Canada DNF still noted
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
# 11. CIRCUIT SCORE — 6 ROUNDS OF 2026 DATA
#     AUS + CHN + JPN + MIA + CAN + MON
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS   CHN   JPN   MIA   CAN   MON
    "RUS":              [1,    2,    4,    4,    20,   20],  # 2x DNF hurts
    "HAM":              [7,    3,    6,    7,    2,    2],
    "ANT":              [2,    1,    1,    1,    1,    1],
    "NOR":              [5,    20,   5,    2,    20,   20],
    "VER":              [20,   20,   8,    5,    3,    20],  # Monaco DNF
    "HAD":              [20,   8,    9,    20,   5,    3],
    "PIA":              [22,   2,    2,    3,    20,   4],
    "LAW":              [20,   20,   20,   20,   7,    5],
    "HUL":              [20,   20,   20,   20,   20,   20],  # Limited data
    "LEC":              [3,    4,    3,    6,    4,    20],  # Monaco DNF
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
#     Barcelona split ratios (approximate)
#     S1: 30% — T1 to T4 (braking + technical)
#     S2: 42% — T5 to T13 (long technical middle)
#     S3: 28% — T14 to finish (fast sweepers)
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.30
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.42
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.28
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.07

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
    "AdjustedTeamScore", "TyreDegScore",
    "BarcelonaGridPenalty", "CircuitScore"
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
    "PoleWetBonus",
    "RainProbability",
    "Temperature",
    "TempDelta",
    "Humidity",
    "WindSpeed",
    "ERSDependencyScore",
    "BarcelonaGridPenalty",  # medium overtaking difficulty
    "TyreDegScore",          # 🆕 hot dry race — tyre deg critical
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",          # 6 rounds of 2026 data
    "SprintWinnerBoost",
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

# Barcelona grid penalty — medium overtaking difficulty
data["PredictedLapTime (s)"] += data["BarcelonaGridPenalty"] * 0.4

# Tyre deg penalty — hot race hurts teams with worse deg management
data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.03

# Wet bonus — minimal at 5% rain
data["WetBonus"] = (
    (1 - data["WetPerformanceFactor"]) * RAIN_PROBABILITY * 100
)
data["PredictedLapTime (s)"] -= data["WetBonus"]

# Pole wet bonus — zero (rain <60%)
data["PredictedLapTime (s)"] -= data["PoleWetBonus"]

# Sort by fastest predicted lap time
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 17. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 SPANISH GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Qualifying: {QUALIFYING_TEMP}°C ⛅  →  Race: {RACE_TEMP}°C ☀️")
print(f"  ☀️  Rain: {int(RAIN_PROBABILITY*100)}% — dry hot race")
print(f"  🔋  ERS limit: 7MJ  |  ⚡ Boost cap: +150kW")
print(f"  🔥  Tyre degradation — key factor at 29°C")
print(f"  🌟  Pole: George Russell — Antonelli streak broken!\n")

# ══════════════════════════════════════════════════════════
# 18. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 7: SPANISH GP\n"
    "CIRCUIT DE BARCELONA-CATALUNYA  |  JUNE 15, 2026  |  ☀️ 29°C DRY",
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
    "📊 Predicted Race Finishing Order  (☀️ Hot Dry Race — Tyre Deg Key)",
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
    "🔥 Tyre Degradation Score\n(lower = better tyre management at 29°C)",
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
    ax2.text(
        row["TyreDegScore"] + 0.05, i,
        f"{row['TyreDegScore']:.1f}",
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
    "BCN Grid Penalty", "Tyre Deg 🆕",
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
ax5.set_title("🏆 Predicted Podium  ☀️",
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
    f"☀️ Race: {RACE_TEMP}°C dry  |  "
    f"🔋 ERS: 7MJ  |  "
    f"🔥 TyreDeg: key feature  |  "
    f"🌟 Pole: Russell (Antonelli streak broken!)",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_07_spain_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_07_spain_prediction.png")
plt.show()