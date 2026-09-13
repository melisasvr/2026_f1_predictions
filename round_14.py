"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 14: MADRID GP
 Circuit de Madrid | Race Date: September 13, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.07 (7% slower)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            MadridGridPenalty, TyreDegScore,
            ReliabilityRiskScore, CircuitScore,
            SprintWinnerBoost, HomeRaceBoost
 Special  : BRAND NEW CIRCUIT — no historical F1 data!
            Model relies entirely on 2026 season form
            CircuitScore based on general 2026 results only
            No circuit-specific sector time history
 Upgrades vs R13:
            + Brand new Circuit de Madrid — first ever race
            + Humidity jump 13% → 46% (FP→Race day)
            + Norris pole — McLaren adapting fastest
            + Antonelli only 0.011s off pole
            + 13 rounds of 2026 CircuitScore data
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 14: MADRID GP")
print("=" * 62)
print("\n  🆕 BRAND NEW CIRCUIT — First ever F1 race at Madrid!")

# ══════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 31    # °C — sunny Saturday
RACE_TEMP        = 31    # °C — sunny Sunday
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # 0°C
RAIN_PROBABILITY = 0.00  # 0% — completely dry
HUMIDITY         = 46    # % race day (up from 13% Friday)
WIND_SPEED       = 6     # km/h — very light

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  Race: {RACE_TEMP}°C ☀️  (Δ{TEMP_DELTA}°C)")
print(f"☀️  0% rain — pure dry hot race")
print(f"💧  Humidity: {HUMIDITY}% — up from 13% on Friday")
print(f"🆕  BRAND NEW CIRCUIT — no historical data!")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Norris pole at brand new Madrid circuit! 🌟
#    Antonelli only 0.011s behind — razor thin!
# ══════════════════════════════════════════════════════════
POLE_TIME = 91.824  # Norris 1:31.824

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Lando Norris",
        "Kimi Antonelli",
        "Max Verstappen",
        "Lewis Hamilton",
        "Charles Leclerc",
        "George Russell",
        "Oscar Piastri",
        "Liam Lawson",
        "Franco Colapinto",
        "Arvid Lindblad",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        91.824,   # 1:31.824 — Norris POLE 🌟
        91.835,   # 1:31.835 — Antonelli   +0.011s
        91.964,   # 1:31.964 — Verstappen  +0.140s
        92.013,   # 1:32.013 — Hamilton    +0.189s
        92.019,   # 1:32.019 — Leclerc     +0.195s
        92.149,   # 1:32.149 — Russell     +0.325s
        92.294,   # 1:32.294 — Piastri     +0.470s
        92.316,   # 1:32.316 — Lawson      +0.492s
        92.903,   # 1:32.903 — Colapinto   +1.079s
        93.041,   # 1:33.041 — Lindblad    +1.217s
    ],
    "Team": [
        "McLaren",  "Mercedes",        "Red Bull Racing",
        "Ferrari",  "Ferrari",          "Mercedes",
        "McLaren",  "Red Bull Racing",  "Alpine",
        "Racing Bulls",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "SprintWinnerBoost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Lando Norris":     "NOR",
    "Kimi Antonelli":   "ANT",
    "Max Verstappen":   "VER",
    "Lewis Hamilton":   "HAM",
    "Charles Leclerc":  "LEC",
    "George Russell":   "RUS",
    "Oscar Piastri":    "PIA",
    "Liam Lawson":      "LAW",
    "Franco Colapinto": "COL",
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
# 4. ADJUSTED TEAM SCORE — updated after 13 rounds
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.5,  # Antonelli 7 wins — dominant
    "McLaren":         9.5,  # Norris pole Madrid — matching Mercedes!
    "Ferrari":         8.8,  # Hamilton wins Spain, Leclerc Britain
    "Red Bull Racing": 7.5,  # VER P3 Monza — improving every race
    "Alpine":          7.0,  # Gasly pole Monza — shock of season!
    "Racing Bulls":    6.0,  # Lindblad consistently Q3
    "Audi":            5.5,
    "Haas":            5.0,
    "Aston Martin":    4.5,
    "Williams":        4.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE — 0% rain, minimal impact
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "NOR": 0.976,
    "ANT": 0.972,
    "VER": 0.966,
    "HAM": 0.964,
    "LEC": 0.974,
    "RUS": 0.966,
    "PIA": 0.975,
    "LAW": 0.979,
    "COL": 0.978,
    "LIN": 0.983,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — 0% rain = zero
# ══════════════════════════════════════════════════════════
qualifying_2026["PoleWetBonus"] = 0.0

# ══════════════════════════════════════════════════════════
# 7. MADRID GRID PENALTY
#    Brand new circuit — overtaking difficulty unknown
#    Based on circuit layout description: medium difficulty
#    Mix of high speed and technical sections
#    Estimated: 0.10s per position
# ══════════════════════════════════════════════════════════
qualifying_2026["MadridGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.10
)

# ══════════════════════════════════════════════════════════
# 8. TYRE DEG SCORE — 31°C hot dry race
# ══════════════════════════════════════════════════════════
TYRE_DEG = {
    "Mercedes":        2.0,
    "Ferrari":         2.5,
    "McLaren":         1.5,  # Best tyre management
    "Red Bull Racing": 2.0,
    "Alpine":          3.0,
    "Racing Bulls":    3.0,
    "Audi":            3.5,
    "Haas":            3.5,
    "Aston Martin":    3.0,
    "Williams":        3.5,
    "Cadillac":        4.5,
}
qualifying_2026["TyreDegScore"] = qualifying_2026["Team"].map(TYRE_DEG)

# ══════════════════════════════════════════════════════════
# 9. HOME RACE BOOST
#    Alonso and Spanish fans — but neither in Q3
#    Colapinto is Argentine — no boost
#    No specific home drivers in top 10
# ══════════════════════════════════════════════════════════
qualifying_2026["HomeRaceBoost"] = 0.0

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
    "Mercedes":        4.5,
    "McLaren":         2.5,
    "Ferrari":         2.0,
    "Red Bull Racing": 3.0,
    "Alpine":          4.0,
    "Racing Bulls":    3.5,
    "Audi":            5.0,
    "Haas":            3.5,
    "Aston Martin":    3.5,
    "Williams":        4.0,
    "Cadillac":        5.0,
}
qualifying_2026["ReliabilityRiskScore"] = qualifying_2026["Team"].map(
    RELIABILITY_RISK
)

# ══════════════════════════════════════════════════════════
# 12. CIRCUIT SCORE — 13 ROUNDS OF 2026 DATA
#     No Madrid-specific history — using season form only
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS  CHN  JPN  MIA  CAN  MON  ESP  AUT  GBR  BEL  HUN  NED  ITA
    "NOR":              [5,   20,  5,   2,   20,  20,  3,   20,  4,   7,   1,   1,   4],
    "ANT":              [2,   1,   1,   1,   1,   1,   4,   3,   20,  1,   3,   2,   1],
    "VER":              [20,  20,  8,   5,   3,   20,  5,   2,   20,  3,   2,   20,  3],
    "HAM":              [7,   3,   6,   7,   2,   2,   1,   5,   3,   4,   5,   4,   6],
    "LEC":              [3,   4,   3,   6,   4,   20,  20,  20,  1,   2,   4,   5,   20],
    "RUS":              [1,   2,   4,   4,   20,  20,  2,   1,   2,   20,  7,   3,   2],
    "PIA":              [22,  2,   2,   3,   20,  4,   20,  4,   20,  5,   20,  6,   5],
    "LAW":              [20,  20,  20,  20,  7,   5,   20,  20,  8,   20,  8,   7,   20],
    "COL":              [20,  10,  20,  8,   20,  20,  20,  20,  20,  10,  15,  20,  9],
    "LIN":              [8,   20,  14,  20,  6,   6,   20,  20,  7,   9,   10,  10,  8],
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
#     Madrid — brand new circuit, estimated split ratios
#     Based on circuit description: mixed layout
#     S1: 31% — start/finish straight + first complex
#     S2: 41% — technical middle section
#     S3: 28% — final sector back to start
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.31
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.41
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.28
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
    "MadridGridPenalty", "CircuitScore"
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
    "MadridGridPenalty",     # brand new circuit — estimated
    "TyreDegScore",          # 31°C hot race
    "HomeRaceBoost",
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",          # 13 rounds 2026 data — no Madrid history
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

# Madrid grid penalty
data["PredictedLapTime (s)"] += data["MadridGridPenalty"] * 0.4

# Tyre deg — 31°C meaningful
data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.02

# Wet bonus — zero (0% rain)
data["PredictedLapTime (s)"] -= 0

# Sort
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 18. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 MADRID GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Race: {RACE_TEMP}°C ☀️  |  Rain: 0%  |  Wind: {WIND_SPEED}km/h")
print(f"  🆕  FIRST EVER F1 RACE AT CIRCUIT DE MADRID!")
print(f"  🟠  Pole: Norris — only 0.011s over Antonelli!")
print(f"  ⚠️  No historical circuit data — model uses 2026 form only\n")

# ══════════════════════════════════════════════════════════
# 19. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 14: MADRID GP\n"
    "CIRCUIT DE MADRID  |  SEP 13, 2026  |  ☀️ 31°C  |  🆕 FIRST EVER F1 RACE HERE!",
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
    "📊 Predicted Race Finishing Order  (🆕 Brand New Circuit — No Historical Data!)",
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

# ── Chart 2: 2026 Season Circuit Score ───────────────────
ax2 = fig.add_subplot(gs[1, 0])
cs_sorted = data.sort_values("CircuitScore")
cs_colors = [TEAM_COLORS.get(t, "#FFF") for t in cs_sorted["Team"]]
ax2.barh(
    cs_sorted["Driver"][::-1],
    cs_sorted["CircuitScore"][::-1],
    color=cs_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title(
    "📊 2026 Season Form Score\n(lower = stronger 2026 results — no Madrid history!)",
    fontsize=10, fontweight="bold", color="white",
    fontfamily=FONT, pad=10
)
ax2.set_xlabel("Circuit Score (lower = better 2026 form)",
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
ax3.set_title("⏱️  Qualifying Gap to Pole — First Ever Madrid Q3!",
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
    "Humidity", "Wind Speed", "ERS Dependency",
    "Madrid Grid", "Tyre Deg",
    "Home Boost", "Reliability",
    "Sector 1", "Sector 2", "Sector 3",
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
ax5.set_title("🏆 Predicted Podium  🇪🇸",
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
    f"🔋 ERS: 7MJ  |  "
    f"🆕 FIRST EVER F1 RACE AT CIRCUIT DE MADRID  |  "
    f"🟠 Pole: Norris (+0.011s over ANT!)",
    ha="center", fontsize=7, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_14_madrid_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_14_madrid_prediction.png")
plt.show()