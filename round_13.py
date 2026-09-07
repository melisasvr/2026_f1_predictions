"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 13: ITALIAN GP
 Autodromo Nazionale Monza | Race Date: September 6, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.04 (Monza slower ~4%)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, PoleWetBonus,
            RainProbability, Temperature, TempDelta,
            Humidity, WindSpeed, ERSDependencyScore,
            MonzaGridPenalty, TyreDegScore,
            ReliabilityRiskScore, CircuitScore,
            SprintWinnerBoost, HomeRaceBoost,
            MonzaSlipstreamBoost
 Upgrades vs R12:
            + GASLY SHOCK POLE — Alpine first pole of 2026!
            + MonzaSlipstreamBoost — overtaking more possible
            + Ferrari HomeRaceBoost — Leclerc + Hamilton + ANT
            + 0% rain — pure pace race
            + 32°C hottest of season alongside Austria
            + ERS dependency most critical here (power circuit)
            + 12 rounds of 2026 CircuitScore data
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 13: ITALIAN GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER — Perfect sunny weekend
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 31    # °C — sunny Saturday
RACE_TEMP        = 32    # °C — sunny Sunday
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # +1°C
RAIN_PROBABILITY = 0.00  # 0% — completely dry all weekend
HUMIDITY         = 40    # % estimated — dry conditions
WIND_SPEED       = 8     # km/h — light winds

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  Race: {RACE_TEMP}°C ☀️  (Δ+{TEMP_DELTA}°C)")
print(f"☀️  0% rain — perfect sunny Monza weekend")
print(f"⚡  FASTEST CIRCUIT — ERS dependency critical!")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA
#    GASLY POLE — Alpine shock! 🇫🇷
#    Top 10 covered by only 0.500s — tightest of season!
#    Ferrari P4+P5 — disappointment at home
#    Antonelli P7 — Italian home race
# ══════════════════════════════════════════════════════════
POLE_TIME = 81.786  # Gasly 1:21.786

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Pierre Gasly",
        "George Russell",
        "Oscar Piastri",
        "Charles Leclerc",
        "Lewis Hamilton",
        "Max Verstappen",
        "Kimi Antonelli",
        "Franco Colapinto",
        "Lando Norris",
        "Arvid Lindblad",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        81.786,   # 1:21.786 — Gasly POLE 🌟 SHOCK!
        81.846,   # +0.060s  — Russell
        81.966,   # +0.180s  — Piastri
        82.004,   # +0.218s  — Leclerc
        82.011,   # +0.225s  — Hamilton
        82.070,   # +0.284s  — Verstappen
        82.093,   # +0.307s  — Antonelli
        82.220,   # +0.434s  — Colapinto
        82.256,   # +0.470s  — Norris
        82.286,   # +0.500s  — Lindblad
    ],
    "Team": [
        "Alpine",   "Mercedes",        "McLaren",
        "Ferrari",  "Ferrari",          "Red Bull Racing",
        "Mercedes", "Alpine",           "McLaren",
        "Racing Bulls",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "SprintWinnerBoost": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Pierre Gasly":     "GAS",
    "George Russell":   "RUS",
    "Oscar Piastri":    "PIA",
    "Charles Leclerc":  "LEC",
    "Lewis Hamilton":   "HAM",
    "Max Verstappen":   "VER",
    "Kimi Antonelli":   "ANT",
    "Franco Colapinto": "COL",
    "Lando Norris":     "NOR",
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
# 4. ADJUSTED TEAM SCORE — updated after 12 rounds
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.5,
    "McLaren":         9.3,  # Norris 2 wins — title contender
    "Ferrari":         9.0,  # Home race motivation
    "Red Bull Racing": 7.0,
    "Alpine":          7.5,  # GASLY POLE — massive upgrade!
    "Racing Bulls":    6.0,
    "Audi":            5.5,
    "Haas":            5.0,  # Bearman P12 — Q3!
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
    "GAS": 0.977,
    "RUS": 0.966,
    "PIA": 0.975,
    "LEC": 0.974,
    "HAM": 0.964,
    "VER": 0.966,
    "ANT": 0.972,
    "COL": 0.978,
    "NOR": 0.976,
    "LIN": 0.983,
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. POLE WET BONUS — 0% rain = zero
# ══════════════════════════════════════════════════════════
POLE_WET_BONUS_FACTOR = 0.20
qualifying_2026["PoleWetBonus"] = 0.0

# ══════════════════════════════════════════════════════════
# 7. MONZA GRID PENALTY
#    Monza has good overtaking — long straights, slipstream
#    Easier than most circuits but still meaningful
#    0.07s per position — most overtaking-friendly circuit
# ══════════════════════════════════════════════════════════
qualifying_2026["MonzaGridPenalty"] = qualifying_2026["GridPosition"].apply(
    lambda p: (p - 1) * 0.07
)

# ══════════════════════════════════════════════════════════
# 8. MONZA SLIPSTREAM BOOST — NEW FEATURE 🆕
#    Monza's long straights create slipstream opportunities
#    Lower grid positions can benefit from tow
#    Makes Monza unique — grid position less dominant
#    Drivers with high straight line speed benefit most
# ══════════════════════════════════════════════════════════
SLIPSTREAM = {
    "GAS": 0.8,   # Renault PU good straight line
    "RUS": 0.6,   # Mercedes PU strong
    "PIA": 0.6,   # McLaren strong straight line
    "LEC": 0.9,   # Ferrari PU elite at Monza
    "HAM": 0.9,   # Ferrari PU elite at Monza
    "VER": 1.0,   # Ford PU less ERS = more pure ICE speed
    "ANT": 0.6,   # Mercedes PU
    "COL": 0.8,   # Renault PU
    "NOR": 0.6,   # McLaren strong straight line
    "LIN": 0.7,   # Ford PU customer
}
qualifying_2026["MonzaSlipstreamBoost"] = qualifying_2026["DriverCode"].map(
    SLIPSTREAM
)

# ══════════════════════════════════════════════════════════
# 9. HOME RACE BOOST
#    Ferrari home race — Leclerc, Hamilton (Ferrari driver)
#    Antonelli — Italian, home country
#    Gasly — French but Monza special for Alpine pole
# ══════════════════════════════════════════════════════════
HOME_BOOST = {
    "GAS": 0.5,   # Shock pole momentum
    "RUS": 0.0,
    "PIA": 0.0,
    "LEC": 1.5,   # Ferrari home race — Tifosi!
    "HAM": 0.8,   # Ferrari driver at Monza — special
    "VER": 0.0,
    "ANT": 1.2,   # Italian — home country crowd!
    "COL": 0.0,
    "NOR": 0.0,
    "LIN": 0.0,
}
qualifying_2026["HomeRaceBoost"] = qualifying_2026["DriverCode"].map(
    HOME_BOOST
)

# ══════════════════════════════════════════════════════════
# 10. ERS DEPENDENCY — MOST CRITICAL AT MONZA
#     Longest full-throttle sections of any circuit
#     7MJ limit is most punishing here
#     Low ERS dependency teams benefit MOST at Monza
# ══════════════════════════════════════════════════════════
ERS_DEPENDENCY = {
    "Mercedes":        9.0,
    "McLaren":         9.0,
    "Ferrari":         6.5,   # Less hurt — Ferrari PU balanced
    "Red Bull Racing": 5.5,   # Ford PU benefits most at Monza!
    "Alpine":          6.0,   # Renault PU — decent
    "Racing Bulls":    5.5,   # Ford PU customer
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
# 11. TYRE DEG — 32°C hot race
# ══════════════════════════════════════════════════════════
TYRE_DEG = {
    "Mercedes":        2.0,
    "Ferrari":         2.0,   # Good at Monza historically
    "McLaren":         1.5,
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
# 12. RELIABILITY RISK
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
# 13. CIRCUIT SCORE — 12 ROUNDS OF 2026 DATA
# ══════════════════════════════════════════════════════════
RESULTS_2026 = {
    #                    AUS  CHN  JPN  MIA  CAN  MON  ESP  AUT  GBR  BEL  HUN  NED
    "GAS":              [10,  6,   10,  20,  8,   7,   20,  20,  20,  20,  12,  10],
    "RUS":              [1,   2,   4,   4,   20,  20,  2,   1,   2,   20,  7,   3],
    "PIA":              [22,  2,   2,   3,   20,  4,   20,  4,   20,  5,   20,  6],
    "LEC":              [3,   4,   3,   6,   4,   20,  20,  20,  1,   2,   4,   5],
    "HAM":              [7,   3,   6,   7,   2,   2,   1,   5,   3,   4,   5,   4],
    "VER":              [20,  20,  8,   5,   3,   20,  5,   2,   20,  3,   2,   20],
    "ANT":              [2,   1,   1,   1,   1,   1,   4,   3,   20,  1,   3,   2],
    "COL":              [20,  10,  20,  8,   20,  20,  20,  20,  20,  10,  15,  20],
    "NOR":              [5,   20,  5,   2,   20,  20,  3,   20,  4,   7,   1,   1],
    "LIN":              [8,   20,  14,  20,  6,   6,   20,  20,  7,   9,   10,  10],
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
#     Monza split ratios (approximate)
#     S1: 32% — start to Variante del Rettifilo
#     S2: 38% — Lesmo curves to Variante Ascari
#     S3: 30% — Parabolica to finish line
# ══════════════════════════════════════════════════════════
qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.32
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.38
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * 0.30
# Monza race pace only ~4% slower than qualifying
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.04

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
    "AdjustedTeamScore", "ERSDependencyScore",
    "MonzaSlipstreamBoost", "HomeRaceBoost", "CircuitScore"
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
    "ERSDependencyScore",     # MOST CRITICAL at Monza
    "MonzaGridPenalty",       # easier to overtake than most
    "TyreDegScore",           # 32°C hot race
    "MonzaSlipstreamBoost",   # 🆕 straight line speed advantage
    "HomeRaceBoost",          # Ferrari + Antonelli home race
    "ReliabilityRiskScore",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)",
    "CircuitScore",           # 12 rounds of 2026 data
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

# Monza grid penalty — easier to overtake
data["PredictedLapTime (s)"] += data["MonzaGridPenalty"] * 0.35

# Tyre deg — 32°C meaningful
data["PredictedLapTime (s)"] += data["TyreDegScore"] * 0.02

# Slipstream boost — benefits low ERS, high straight line speed
data["PredictedLapTime (s)"] -= data["MonzaSlipstreamBoost"] * 0.04

# Home race boost
data["PredictedLapTime (s)"] -= data["HomeRaceBoost"] * 0.05

# Wet bonus — zero (0% rain)
data["WetBonus"] = 0.0
data["PredictedLapTime (s)"] -= data["WetBonus"]

# Sort
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 19. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 ITALIAN GP — PREDICTED RACE RESULT")
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
print(f"  ☀️  0% rain — perfect sunny Monza weekend")
print(f"  ⚡  FASTEST CIRCUIT — ERS dependency #1 feature")
print(f"  🔴  Ferrari home race — Tifosi will be LOUD!")
print(f"  😱  GASLY POLE — Alpine shock of the season!")
print(f"  🇮🇹  Antonelli home country race!\n")

# ══════════════════════════════════════════════════════════
# 20. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 13: ITALIAN GP\n"
    "AUTODROMO NAZIONALE MONZA  |  SEP 6, 2026  |  ☀️ 32°C  |  ⚡ TEMPLE OF SPEED",
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
    "📊 Predicted Race Finishing Order  (😱 GASLY POLE — Ferrari Home Race!)",
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

# ── Chart 2: ERS Dependency — KEY feature at Monza ───────
ax2 = fig.add_subplot(gs[1, 0])
ers_sorted = data.sort_values("ERSDependencyScore")
ers_colors = [TEAM_COLORS.get(t, "#FFF") for t in ers_sorted["Team"]]
ax2.barh(
    ers_sorted["Driver"][::-1],
    ers_sorted["ERSDependencyScore"][::-1],
    color=ers_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title(
    "🔋 ERS Dependency Score\n(lower = less hurt by 7MJ limit — KEY at Monza!)",
    fontsize=10, fontweight="bold", color="white",
    fontfamily=FONT, pad=10
)
ax2.set_xlabel("ERS Dependency (lower = benefits more at Monza)",
               color="#AAAAAA", fontsize=8, fontfamily=FONT)
ax2.tick_params(colors="white", labelsize=8)
ax2.set_facecolor("#1a1a1a")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333333")
for i, (_, row) in enumerate(ers_sorted[::-1].iterrows()):
    ax2.text(row["ERSDependencyScore"] + 0.05, i,
             f"{row['ERSDependencyScore']:.1f}",
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
ax3.set_title("⏱️  Qualifying Gap to Pole — Top 10 in 0.500s!",
              fontsize=10, fontweight="bold", color="white",
              fontfamily=FONT, pad=10)
ax3.set_xlabel("Gap to Pole (seconds)", color="#AAAAAA",
               fontsize=9, fontfamily=FONT)
ax3.tick_params(colors="white", labelsize=8)
ax3.set_facecolor("#1a1a1a")
for spine in ax3.spines.values():
    spine.set_edgecolor("#333333")
for i, (_, row) in enumerate(qual_sorted[::-1].iterrows()):
    ax3.text(row["GapFromPole (s)"] + 0.003, i,
             f"+{row['GapFromPole (s)']:.3f}s",
             va="center", fontsize=7.5,
             color="white", fontfamily=FONT)

# ── Chart 4: Feature Importance ──────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
feat_labels = [
    "Qualifying Time", "Gap From Pole", "Team Score",
    "Grid Penalty", "Wet Factor", "Pole Wet Bonus",
    "Rain Prob", "Temperature", "Temp Delta",
    "Humidity", "Wind Speed", "ERS Dependency ⚡",
    "Monza Grid", "Tyre Deg",
    "Slipstream 🆕", "Home Boost 🏠",
    "Reliability", "Sector 1", "Sector 2", "Sector 3",
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
ax5.set_title("🏆 Predicted Podium  🇮🇹",
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
    f"⚡ ERS: 7MJ — most critical at Monza  |  "
    f"😱 Gasly pole!  |  "
    f"🔴 Ferrari home race",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_13_italy_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_13_italy_prediction.png")
plt.show()