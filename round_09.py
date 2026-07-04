"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 9: BRITISH GP
 Silverstone Circuit | Race Date: July 6, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.07 (7% slower)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            SilverstoneGridPenalty, TyreDegScore,
            ReliabilityRiskScore, CircuitScore,
            SprintWinnerBoost, HomeRaceBoost
 Upgrades vs R08:
            + Antonelli Sprint win → SprintWinnerBoost
            + Hamilton + Norris home race boost (Silverstone)
            + 26°C cloudy race — mild tyre deg vs Austria
            + Ferrari P2+P3 quali — very strong
            + Both Racing Bulls in Q3 again
            + 8 rounds of 2026 CircuitScore data
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 9: BRITISH GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 25    # °C — partly cloudy
RACE_TEMP        = 26    # °C — cloudy
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # +1°C
RAIN_PROBABILITY = 0.15  # cloudy but no confirmed rain
HUMIDITY         = 60    # % estimated — cloudy day
WIND_SPEED       = 12    # km/h — Silverstone is exposed

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ⛅  →  Race: {RACE_TEMP}°C ☁️  (Δ+{TEMP_DELTA}°C)")
print(f"☁️  Cloudy race — mild tyre deg vs Austria's 33°C")
print(f"🌧️  Rain: {int(RAIN_PROBABILITY*100)}% — low but cloudy")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Antonelli — 6th pole of the season! 🌟
#    Ferrari P2+P3 — Hamilton home race crowd 🔴
#    Both Racing Bulls in Q3 again
# ══════════════════════════════════════════════════════════
POLE_TIME = 88.111  # Antonelli 1:28.111

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Kimi Antonelli",
        "Charles Leclerc",
        "Lewis Hamilton",
        "George Russell",
        "Isack Hadjar",
        "Lando Norris",
        "Max Verstappen",
        "Oscar Piastri",
        "Arvid Lindblad",
        "Liam Lawson",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        88.111,   # 1:28.111 — Antonelli POLE 🌟
        88.286,   # +0.175s  — Leclerc
        88.458,   # +0.347s  — Hamilton
        88.481,   # +0.370s  — Russell
        88.746,   # +0.635s  — Hadjar
        88.877,   # +0.766s  — Norris
        88.893,   # +0.782s  — Verstappen
        89.032,   # +0.921s  — Piastri
        89.305,   # +1.194s  — Lindblad
        89.716,   # +1.605s  — Lawson
    ],
    "Team": [
        "Mercedes", "Ferrari",         "Ferrari",
        "Mercedes", "Red Bull Racing",  "McLaren",
        "Red Bull Racing", "McLaren",   "Racing Bulls",
        "Racing Bulls",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # Antonelli won Sprint 🏆
    "SprintWinnerBoost": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Kimi Antonelli":  "ANT",
    "Charles Leclerc": "LEC",
    "Lewis Hamilton":  "HAM",
    "George Russell":  "RUS",
    "Isack Hadjar":    "HAD",
    "Lando Norris":    "NOR",
    "Max Verstappen":  "VER",
    "Oscar Piastri":   "PIA",
    "Arvid Lindblad":  "LIN",
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
# 4. ADJUSTED TEAM SCORE — updated after 8 rounds
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.8,  # Dominant — 6 wins, Antonelli unstoppable
    "Ferrari":         9.0,  # Hamilton Spain win — Ferrari surging strongly
    "McLaren":         8.0,  # Consistent pace — Silverstone suits them
    "Red Bull Racing": 7.0,  # VER P2 Austria — improving
    "Racing Bulls":    5.5,  # Both in Q3 consistently
    "Alpine":          5.0,
    "Haas":            4.5,
    "Aston Martin":    4.0,
    "Williams":        4.5,
    "Audi":            5.0,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "ANT": 0.972,
    "LEC": 0.974,
    "HAM": 0.964,
    "RUS": 0.966,
    "HAD": 0.980,
    "NOR": 0.976,
    "VER": 0.968,
    "PIA": 0.975,
    "LIN": 0.983,
    "LAW": 0.979,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — low rain (15% < 60% threshold)
# ══════════════════════════════════════════════════════════
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY if (
        p == 1 and RAIN_PROBABILITY >= 0.60
    ) else 0.0
)

# ══════════════════════════════════════════════════════════
# 7. SILVERSTONE GRID PENALTY
#    Silverstone has good overtaking opportunities
#    Club corner, Stowe, Vale — multiple spots
#    Medium difficulty — less than Monaco, similar to Austria
#    0.09s per position penalty
# ══════════════════════════════════════════════════════════
qualifying_2026["SilverstoneGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.09
)

# ══════════════════════════════════════════════════════════
# 8. TYRE DEG SCORE — 26°C CLOUDY
#    Significantly milder than Austria (33°C)
#    Less tyre deg pressure — pace more equal
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
# 9. HOME RACE BOOST — Hamilton AND Norris at Silverstone
#    Both British drivers at their home race
#    Hamilton: legendary Silverstone record — 8x winner
#    Norris: massive home crowd support
# ══════════════════════════════════════════════════════════
qualifying_2026["HomeRaceBoost"] = qualifying_2026["DriverCode"].apply(
    lambda d: 1.5 if d == "HAM" else (1.0 if d == "NOR" else 0)
)

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
    "Red Bull Racing": 3.5,
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
# 12. CIRCUIT SCORE — 8 ROUNDS OF 2026 DATA
#     AUS+CHN+JPN+MIA+CAN+MON+ESP+AUT
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS  CHN  JPN  MIA  CAN  MON  ESP  AUT
    "ANT":              [2,   1,   1,   1,   1,   1,   4,   3],
    "LEC":              [3,   4,   3,   6,   4,   20,  20,  20],
    "HAM":              [7,   3,   6,   7,   2,   2,   1,   5],
    "RUS":              [1,   2,   4,   4,   20,  20,  2,   1],
    "HAD":              [20,  8,   9,   20,  5,   3,   6,   20],
    "NOR":              [5,   20,  5,   2,   20,  20,  3,   20],
    "VER":              [20,  20,  8,   5,   3,   20,  5,   2],
    "PIA":              [22,  2,   2,   3,   20,  4,   20,  4],
    "LIN":              [8,   20,  14,  20,  6,   6,   20,  20],
    "LAW":              [20,  20,  20,  20,  7,   5,   20,  20],
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
#     Silverstone split ratios (approximate)
#     S1: 31% — Copse to Maggotts/Becketts
#     S2: 40% — Hangar straight to Stowe/Vale
#     S3: 29% — Club corner to finish
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.31
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.40
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.29
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
    "HomeRaceBoost", "CircuitScore"
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
    "SilverstoneGridPenalty",
    "TyreDegScore",
    "HomeRaceBoost",         # Hamilton + Norris home boost
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",          # 8 rounds of 2026 data
    "SprintWinnerBoost",     # Antonelli Sprint win
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

# Silverstone grid penalty
data["PredictedLapTime (s)"] += data["SilverstoneGridPenalty"] * 0.4

# Tyre deg — milder at 26°C vs Austria
data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.02

# Home race boost — Hamilton and Norris
data["PredictedLapTime (s)"] -= data["HomeRaceBoost"] * 0.06

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
# 18. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 BRITISH GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Qualifying: {QUALIFYING_TEMP}°C ⛅  →  Race: {RACE_TEMP}°C ☁️")
print(f"  ☁️  Cloudy — mild tyre deg conditions")
print(f"  🔋  ERS: 7MJ  |  ⚡ Boost cap: +150kW")
print(f"  🏆  Sprint winner: Kimi Antonelli")
print(f"  🌟  Pole: Antonelli — 6th pole of 2026!")
print(f"  🏠  Home race: Hamilton + Norris (crowd will be LOUD!)\n")

# ══════════════════════════════════════════════════════════
# 19. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 9: BRITISH GP\n"
    "SILVERSTONE CIRCUIT  |  JULY 6, 2026  |  ☁️ 26°C",
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
    "📊 Predicted Race Finishing Order  (🏠 Hamilton & Norris Home Race)",
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

# ── Chart 2: Home Race Boost ──────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
home_colors = [TEAM_COLORS.get(t, "#FFF") for t in qualifying_2026["Team"]]
ax2.barh(
    qualifying_2026["Driver"][::-1],
    qualifying_2026["HomeRaceBoost"][::-1],
    color=home_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title(
    "🏠 Home Race Boost\n(Hamilton 1.5x · Norris 1.0x — Silverstone crowd!)",
    fontsize=10, fontweight="bold", color="white",
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
    "Silverstone Grid", "Tyre Deg",
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
ax5.set_title("🏆 Predicted Podium  🇬🇧",
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
    f"☁️ Race: {RACE_TEMP}°C  |  "
    f"🔋 ERS: 7MJ  |  "
    f"🏆 Sprint: Antonelli  |  "
    f"🌟 Pole: Antonelli (6th!)  |  "
    f"🏠 Home: Hamilton + Norris",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_09_britain_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_09_britain_prediction.png")
plt.show()