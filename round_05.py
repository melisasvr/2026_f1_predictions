"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 5: CANADIAN GP
 Circuit Gilles Villeneuve | Race Date: May 25, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.07 (7% slower)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            MiamiBoostCapScore, ReliabilityRiskScore,
            CircuitScore, SprintWinnerBoost
 Upgrades vs R04:
            + PoleWetBonus factor 0.10 → 0.20 (Miami lesson)
            + 95% rain — WetPerformanceFactor #1 feature
            + 14°C cold wet race — extreme conditions
            + CircuitScore now 4 full rounds of 2026 data
            + Russell: Sprint pole + Sprint win + GP pole
            + Hamilton Montreal history factored in
 Author   : F1 Predictions 2026
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 5: CANADIAN GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 21    # °C Saturday
RACE_TEMP        = 14    # °C Sunday — COLD wet race
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # -7°C
RAIN_PROBABILITY = 0.95  # 95% 🚨🚨 — almost certain rain
HUMIDITY         = 41    # %
WIND_SPEED       = 16    # km/h

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ☁️  →  Race: {RACE_TEMP}°C 🌧️  (Δ{TEMP_DELTA}°C)")
print(f"🌧️  Rain: {int(RAIN_PROBABILITY*100)}% 🚨🚨  |  💧 Humidity: {HUMIDITY}%  |  💨 Wind: {WIND_SPEED}km/h")
print(f"⚠️  COLD WET RACE — 14°C + 95% RAIN — EXTREME CONDITIONS")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    Russell — Sprint pole + Sprint win + GP pole 🌟
# ══════════════════════════════════════════════════════════
POLE_TIME = 72.578  # Russell 1:12.578

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "George Russell",
        "Kimi Antonelli",
        "Lando Norris",
        "Oscar Piastri",
        "Lewis Hamilton",
        "Max Verstappen",
        "Isack Hadjar",
        "Charles Leclerc",
        "Arvid Lindblad",
        "Franco Colapinto",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        72.578,   # 1:12.578 — Russell POLE 🌟
        72.646,   # 1:12.646 — Antonelli   +0.068s
        72.729,   # 1:12.729 — Norris      +0.151s
        72.781,   # 1:12.781 — Piastri     +0.203s
        72.868,   # 1:12.868 — Hamilton    +0.290s
        72.907,   # 1:12.907 — Verstappen  +0.329s
        72.935,   # 1:12.935 — Hadjar      +0.357s
        72.976,   # 1:12.976 — Leclerc     +0.398s
        73.280,   # 1:13.280 — Lindblad    +0.702s
        73.697,   # 1:13.697 — Colapinto   +1.119s
    ],
    "Team": [
        "Mercedes", "Mercedes", "McLaren",
        "McLaren",  "Ferrari",  "Red Bull Racing",
        "Red Bull Racing", "Ferrari",
        "Racing Bulls", "Alpine",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # Russell won Sprint 🏆
    "SprintWinnerBoost": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "George Russell":   "RUS",
    "Kimi Antonelli":   "ANT",
    "Lando Norris":     "NOR",
    "Oscar Piastri":    "PIA",
    "Lewis Hamilton":   "HAM",
    "Max Verstappen":   "VER",
    "Isack Hadjar":     "HAD",
    "Charles Leclerc":  "LEC",
    "Arvid Lindblad":   "LIN",
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
# 4. ADJUSTED TEAM SCORE
#    Updated after 4 rounds of 2026 data
#    AUS: Merc 1-2 | CHN: Merc 1-2 | JPN: Merc 1
#    MIA: Ant 1, NOR 2, PIA 3 | Sprint CAN: RUS 1
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.5,  # Dominant — 4 wins from 5 rounds
    "McLaren":         8.5,  # Miami 2-3, Sprint Canada P2+P4 — strong
    "Ferrari":         7.5,  # Consistent but no wins yet
    "Red Bull Racing": 6.5,  # VER improving — P6 Canada quali
    "Racing Bulls":    5.0,  # Lindblad consistently in Q3
    "Alpine":          5.5,  # Colapinto P10 — Q3 again
    "Haas":            4.5,
    "Aston Martin":    4.0,
    "Williams":        4.0,
    "Audi":            3.5,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. WET PERFORMANCE FACTOR
#    95% rain + 14°C — cold wet conditions
#    MOST IMPORTANT FEATURE THIS ROUND
#    Lower = better wet driver
#    Hamilton historically dominant at Montreal in wet
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "RUS": 0.966,   # ELITE — one of best wet drivers on grid
    "ANT": 0.972,   # Strong wet ability
    "NOR": 0.976,   # Decent but inconsistent
    "PIA": 0.975,   # Good wet driver
    "HAM": 0.964,   # ALL TIME greatest wet driver + loves Montreal
    "VER": 0.968,   # Legendary in wet — Spa 2021
    "HAD": 0.980,   # Limited cold wet F1 experience
    "LEC": 0.974,   # Good wet performer
    "LIN": 0.983,   # Rookie — limited wet F1 data
    "COL": 0.977,   # Argentina wet experience helps
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — UPGRADED FROM MIAMI
#    Miami lesson: 0.10 was too conservative
#    Now 0.20 for rain >75%
#    At 95% rain: 0.20 * 0.95 = 0.19s advantage
#    Russell on pole — biggest pole bonus of the season
# ══════════════════════════════════════════════════════════
POLE_WET_BONUS_FACTOR = 0.20  # upgraded from 0.10 after Miami

qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY if (
        p == 1 and RAIN_PROBABILITY >= 0.60
    ) else 0.0
)

# ══════════════════════════════════════════════════════════
# 7. ERS DEPENDENCY SCORE (7MJ limit continues)
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
# 8. MONTREAL BOOST CAP SCORE
#    Circuit Gilles Villeneuve: long straights + tight
#    hairpins. Low downforce circuit.
#    +150kW boost cap still applies
# ══════════════════════════════════════════════════════════
MONTREAL_BOOST_CAP = {
    "Mercedes":        6.5,
    "McLaren":         6.0,
    "Ferrari":         5.5,
    "Red Bull Racing": 5.0,
    "Alpine":          6.5,
    "Racing Bulls":    5.5,
    "Haas":            6.0,
    "Aston Martin":    7.0,
    "Williams":        7.0,
    "Audi":            6.5,
    "Cadillac":        6.5,
}
qualifying_2026["BoostCapScore"] = qualifying_2026["Team"].map(
    MONTREAL_BOOST_CAP
)

# ══════════════════════════════════════════════════════════
# 9. RELIABILITY RISK SCORE (updated after 4 rounds)
# ══════════════════════════════════════════════════════════
RELIABILITY_RISK = {
    "Mercedes":        1.5,
    "McLaren":         3.5,  # Improved since China — Miami P2+P3
    "Ferrari":         2.0,
    "Red Bull Racing": 3.5,
    "Racing Bulls":    3.5,
    "Alpine":          4.0,
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
# 10. CIRCUIT SCORE — 4 ROUNDS OF 2026 DATA
#     AUS + CHN + JPN + MIA actual finishing positions
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS   CHN   JPN   MIA
    "RUS":              [1,    2,    4,    4],
    "ANT":              [2,    1,    1,    1],
    "NOR":              [5,    20,   5,    2],
    "PIA":              [22,   2,    2,    3],
    "HAM":              [7,    3,    6,    7],
    "VER":              [20,   20,   8,    5],
    "HAD":              [20,   8,    9,    20],
    "LEC":              [3,    4,    3,    6],
    "LIN":              [8,    20,   14,   20],
    "COL":              [20,   10,   20,   8],
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
# 11. SYNTHETIC SECTOR TIMES
#     Montreal split ratios (approximate)
#     S1: 32% — hairpin + acceleration
#     S2: 40% — casino straight + chicane
#     S3: 28% — final chicane + pit straight
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.32
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.40
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.28
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.07

# ══════════════════════════════════════════════════════════
# 12. WEATHER FEATURES
# ══════════════════════════════════════════════════════════
qualifying_2026["RainProbability"] = RAIN_PROBABILITY
qualifying_2026["Temperature"]     = RACE_TEMP
qualifying_2026["TempDelta"]       = TEMP_DELTA
qualifying_2026["Humidity"]        = HUMIDITY
qualifying_2026["WindSpeed"]       = WIND_SPEED

print("\n📊 Full Feature Set:")
print(qualifying_2026[[
    "Driver", "QualifyingTime (s)", "GapFromPole (s)",
    "AdjustedTeamScore", "WetPerformanceFactor",
    "PoleWetBonus", "CircuitScore"
]].to_string(index=False))

# ══════════════════════════════════════════════════════════
# 13. FEATURE COLUMNS
# ══════════════════════════════════════════════════════════
FEATURE_COLS = [
    "QualifyingTime (s)",
    "GapFromPole (s)",
    "AdjustedTeamScore",
    "GridPenalty (s)",
    "WetPerformanceFactor",   # 🚨 95% rain — #1 feature
    "PoleWetBonus",           # 0.19s at 95% rain — upgraded
    "RainProbability",
    "Temperature",            # 14°C cold race
    "TempDelta",              # -7°C — coldest delta all season
    "Humidity",
    "WindSpeed",
    "ERSDependencyScore",
    "BoostCapScore",
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",           # 4 rounds of 2026 data
    "SprintWinnerBoost",      # Russell won Sprint
]
TARGET = "RacePace (s)"

# ══════════════════════════════════════════════════════════
# 14. TRAIN MODEL
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
# 15. PREDICT RACE
# ══════════════════════════════════════════════════════════
data = qualifying_2026.copy()
data["PredictedLapTime (s)"] = model.predict(X)

# Wet race adjustment
data["WetBonus"] = (
    (1 - data["WetPerformanceFactor"]) * RAIN_PROBABILITY * 100
)
data["PredictedLapTime (s)"] -= data["WetBonus"]

# Pole wet bonus
data["PredictedLapTime (s)"] -= data["PoleWetBonus"]

# Sort by fastest predicted lap time
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 16. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 CANADIAN GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Qualifying: {QUALIFYING_TEMP}°C ☁️  →  Race: {RACE_TEMP}°C 🌧️  (Δ{TEMP_DELTA}°C)")
print(f"  🌧️  Rain: {int(RAIN_PROBABILITY*100)}% 🚨🚨")
print(f"  🔋  ERS limit: 7MJ")
print(f"  ⚡  Boost cap: +150kW")
print(f"  🏆  Sprint winner: George Russell")
print(f"  🌟  GP Pole: George Russell — Sprint pole + win + GP pole!")
print(f"  🌧️  PoleWetBonus: {POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY:.3f}s\n")

# ══════════════════════════════════════════════════════════
# 17. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 5: CANADIAN GP\n"
    "CIRCUIT GILLES VILLENEUVE  |  MAY 25, 2026  |  🌧️ 95% RAIN  |  14°C",
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
ax1.set_title("📊 Predicted Race Finishing Order  (🌧️ 95% Rain | 14°C)",
              fontsize=13, fontweight="bold", color="white",
              fontfamily=FONT, pad=12)
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
        data["PredictedLapTime (s)"].min() * 0.9998, i, label,
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

# ── Chart 2: Wet Performance Factor ──────────────────────
ax2 = fig.add_subplot(gs[1, 0])
wet_sorted  = data.sort_values("WetPerformanceFactor")
wet_colors  = [TEAM_COLORS.get(t, "#FFF") for t in wet_sorted["Team"]]
ax2.barh(
    wet_sorted["Driver"][::-1],
    wet_sorted["WetPerformanceFactor"][::-1],
    color=wet_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title("💧 Wet Performance Factor\n(lower = elite wet driver — 95% rain today!)",
              fontsize=10, fontweight="bold", color="white",
              fontfamily=FONT, pad=10)
ax2.set_xlabel("Wet Factor (lower = better)",
               color="#AAAAAA", fontsize=8, fontfamily=FONT)
ax2.tick_params(colors="white", labelsize=8)
ax2.set_facecolor("#1a1a1a")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333333")
for i, (_, row) in enumerate(wet_sorted[::-1].iterrows()):
    ax2.text(
        row["WetPerformanceFactor"] + 0.0001, i,
        f"{row['WetPerformanceFactor']:.3f}",
        va="center", fontsize=7.5,
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
        row["GapFromPole (s)"] + 0.002, i,
        f"+{row['GapFromPole (s)']:.3f}s",
        va="center", fontsize=7.5,
        color="white", fontfamily=FONT
    )

# ── Chart 4: Feature Importance ──────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
feat_labels = [
    "Qualifying Time", "Gap From Pole", "Team Score",
    "Grid Penalty", "Wet Factor 🚨", "Pole Wet Bonus",
    "Rain Prob", "Temperature", "Temp Delta",
    "Humidity", "Wind Speed", "ERS Dependency",
    "Boost Cap", "Reliability", "Sector 1",
    "Sector 2", "Sector 3", "Circuit Score",
    "Sprint Boost"
]
feat_import  = model.feature_importances_
sorted_idx   = np.argsort(feat_import)
sorted_labels = [feat_labels[i] for i in sorted_idx]
sorted_values = feat_import[sorted_idx]
colors_bar   = plt.cm.Blues(np.linspace(0.3, 0.95, len(sorted_values)))
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
ax5.set_title("🏆 Predicted Podium  🌧️",
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
    f"🌧️ Rain: {int(RAIN_PROBABILITY*100)}% 🚨  |  "
    f"🌡️ Race: {RACE_TEMP}°C  |  "
    f"🌧️ PoleWetBonus: {POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY:.3f}s  |  "
    f"🏆 Sprint: Russell  |  "
    f"🌟 Pole: Russell",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_05_canada_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_05_canada_prediction.png")
plt.show()