"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 4: MIAMI GP
 Miami International Autodrome | Race Date: May 3, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Race pace = qualifying * 1.07 (7% slower)
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            WetPerformanceFactor, RainProbability,
            Temperature, TempDelta, Humidity, WindSpeed,
            ERSDependencyScore, ReliabilityRiskScore,
            CircuitScore, SprintWinnerBoost,
            MiamiBoostCapScore
 Upgrades vs R03:
            + 80% rain probability — WetFactor now #1 feature
            + New ERS limit: 7MJ (was 8MJ in Japan)
            + Boost capped at +150kW in race
            + MGU-K deployment zones modelled
            + Verstappen P2 — Red Bull Ford resurgence
            + Colapinto P8 — Alpine surprise
            + Circuit scores from AUS + CHN + JPN (3 rounds)
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
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 4: MIAMI GP")
print("=" * 62)

# ══════════════════════════════════════════════════════════
# 1. WEATHER  — UPDATED race day forecast
#    Originally forecast 55% rain — now confirmed 80%!
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 32    # °C — hot sunny Saturday
RACE_TEMP        = 26    # °C — Sunday rain brings temp down
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP   # -6°C (cooler race!)
RAIN_PROBABILITY = 0.80  # 80% 🚨 — highest all season
HUMIDITY         = 81    # % — very high
WIND_SPEED       = 16    # km/h

print(f"\n🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  Race: {RACE_TEMP}°C 🌧️  (Δ{TEMP_DELTA}°C)")
print(f"🌧️  Rain: {int(RAIN_PROBABILITY*100)}% 🚨  |  💧 Humidity: {HUMIDITY}%  |  💨 Wind: {WIND_SPEED}km/h")
print(f"⚠️  WET RACE CONDITIONS HIGHLY LIKELY")

# ══════════════════════════════════════════════════════════
# 2. 2026 Q3 QUALIFYING DATA  (real times)
#    Antonelli — 3rd pole in 4 races 🌟
#    Verstappen P2 — Red Bull Ford resurgence! 😱
#    Colapinto P8 — Alpine surprise 🌟
# ══════════════════════════════════════════════════════════
POLE_TIME = 87.798  # Antonelli 1:27.798

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Kimi Antonelli",
        "Max Verstappen",
        "Charles Leclerc",
        "Lando Norris",
        "George Russell",
        "Lewis Hamilton",
        "Oscar Piastri",
        "Franco Colapinto",
        "Isack Hadjar",
        "Pierre Gasly",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        87.798,   # 1:27.798 — Antonelli POLE 🌟
        87.964,   # +0.166s  — Verstappen
        88.143,   # +0.345s  — Leclerc
        88.183,   # +0.385s  — Norris
        88.197,   # +0.399s  — Russell
        88.319,   # +0.521s  — Hamilton
        88.500,   # +0.702s  — Piastri
        88.762,   # +0.964s  — Colapinto
        88.789,   # +0.991s  — Hadjar
        88.810,   # +1.012s  — Gasly
    ],
    "Team": [
        "Mercedes", "Red Bull Racing", "Ferrari",
        "McLaren",  "Mercedes",        "Ferrari",
        "McLaren",  "Alpine",          "Red Bull Racing",
        "Alpine",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    # Norris won Sprint — SprintWinnerBoost
    "SprintWinnerBoost": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
})

DRIVER_CODES = {
    "Kimi Antonelli":   "ANT",
    "Max Verstappen":   "VER",
    "Charles Leclerc":  "LEC",
    "Lando Norris":     "NOR",
    "George Russell":   "RUS",
    "Lewis Hamilton":   "HAM",
    "Oscar Piastri":    "PIA",
    "Franco Colapinto": "COL",
    "Isack Hadjar":     "HAD",
    "Pierre Gasly":     "GAS",
}
qualifying_2026["DriverCode"] = qualifying_2026["Driver"].map(DRIVER_CODES)
qualifying_2026["GapFromPole (s)"] = (
    qualifying_2026["QualifyingTime (s)"] - POLE_TIME
)

# ── Pole Wet Bonus ────────────────────────────────────────
# In wet races (>60% rain), pole is exponentially more valuable:
# - Clean air, no spray blindness for the leader
# - Controls pace under safety cars / VSC restarts
# - Overtaking nearly impossible when visibility is near zero
# Conservative formula: 0.10 * RainProbability (only activates >60%)
# Applied only to P1 qualifier — scales with rain severity
# At 80% rain: bonus = 0.10 * 0.80 = 0.080s advantage per lap
POLE_WET_BONUS_FACTOR = 0.10  # conservative — tune upward if underfitting

qualifying_2026["PoleWetBonus"] = qualifying_2026["GridPosition"].apply(
    lambda p: POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY if (
        p == 1 and RAIN_PROBABILITY >= 0.60
    ) else 0.0
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
#    Updated after 3 rounds of real 2026 data
#    AUS: Mercedes 1-2 | CHN: Mercedes 1-2 | JPN: Mercedes 1
#    Sprint Miami: McLaren 1-2 | VER P5 — Red Bull improving
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.5,  # Dominant race pace but ERS cuts hurting
    "Ferrari":         8.0,  # Consistent podiums, strong wet package
    "McLaren":         8.0,  # Sprint 1-2 Miami — back to form
    "Red Bull Racing": 7.0,  # VER P2 quali! Major improvement with new regs
    "Alpine":          5.5,  # Colapinto P8 quali — best result of 2026
    "Racing Bulls":    4.5,
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
#    80% rain — THIS IS THE MOST IMPORTANT FEATURE THIS ROUND
#    Lower = better in wet (elite wet drivers closer to 0.965)
# ══════════════════════════════════════════════════════════
WET_PERFORMANCE = {
    "ANT": 0.972,   # Strong wet driver from junior categories
    "VER": 0.968,   # LEGENDARY wet driver — Spa 2021 etc
    "LEC": 0.974,   # Good wet performer
    "NOR": 0.976,   # Decent but inconsistent in wet
    "RUS": 0.967,   # ELITE wet driver — one of the best
    "HAM": 0.965,   # ALL TIME greatest wet driver
    "PIA": 0.975,   # Good wet driver
    "COL": 0.978,   # Limited wet F1 data — Argentina background helps
    "HAD": 0.982,   # Rookie — limited wet F1 experience
    "GAS": 0.977,   # Decent wet performer
}
qualifying_2026["WetPerformanceFactor"] = qualifying_2026["DriverCode"].map(
    WET_PERFORMANCE
)

# ══════════════════════════════════════════════════════════
# 6. ERS DEPENDENCY SCORE
#    3rd consecutive ERS cut: 9MJ → 8MJ → 7MJ
#    + Boost now CAPPED at +150kW in race
#    + MGU-K limited to 250kW outside acceleration zones
#    Higher = more hurt by these changes
# ══════════════════════════════════════════════════════════
ERS_DEPENDENCY = {
    "Mercedes":        9.0,  # Most ERS dependent — 3rd cut hurts most
    "McLaren":         9.0,  # Mercedes PU customer — same impact
    "Ferrari":         6.5,  # Balanced — less hurt by ERS cuts
    "Red Bull Racing": 5.5,  # Ford PU — less ERS reliant, benefits most
    "Alpine":          6.5,  # Renault PU — moderate
    "Racing Bulls":    5.5,  # Ford PU customer
    "Haas":            6.5,  # Ferrari PU customer
    "Aston Martin":    9.0,  # Mercedes PU customer
    "Williams":        9.0,  # Mercedes PU customer
    "Audi":            7.0,  # New PU — unknown
    "Cadillac":        6.5,  # Ferrari PU customer
}
qualifying_2026["ERSDependencyScore"] = qualifying_2026["Team"].map(
    ERS_DEPENDENCY
)

# ══════════════════════════════════════════════════════════
# 7. MIAMI BOOST CAP SCORE
#    New rule: Boost capped at +150kW in race (was uncapped)
#    MGU-K: 350kW in acceleration zones, 250kW elsewhere
#    Miami layout: tight hairpins + long straights
#    Teams with better mechanical grip benefit more
#    Lower score = better adapted to new boost rules
# ══════════════════════════════════════════════════════════
MIAMI_BOOST_CAP = {
    "Mercedes":        7.5,  # Was relying on high boost — now capped
    "McLaren":         6.0,  # Good mechanical balance
    "Ferrari":         5.5,  # Strong mechanical grip — benefits
    "Red Bull Racing": 5.0,  # Mechanical grip strength — benefits most
    "Alpine":          6.5,  # Moderate
    "Racing Bulls":    5.5,  # Ford PU mechanical strength
    "Haas":            6.0,  # Ferrari PU
    "Aston Martin":    7.0,
    "Williams":        7.0,
    "Audi":            6.5,
    "Cadillac":        6.5,
}
qualifying_2026["MiamiBoostCapScore"] = qualifying_2026["Team"].map(
    MIAMI_BOOST_CAP
)

# ══════════════════════════════════════════════════════════
# 8. RELIABILITY RISK SCORE
#    Updated after 3 rounds
#    McLaren reliability has improved since China DNS
# ══════════════════════════════════════════════════════════
RELIABILITY_RISK = {
    "Mercedes":        1.5,  # Still very reliable
    "Ferrari":         2.0,  # Solid
    "McLaren":         5.0,  # Improved from China DNS — still watchful
    "Red Bull Racing": 3.5,  # New Ford PU bedding in
    "Alpine":          4.5,  # Unknown reliability
    "Racing Bulls":    3.5,  # Ford PU
    "Haas":            3.0,  # Ferrari PU solid
    "Aston Martin":    3.5,
    "Williams":        4.0,
    "Audi":            5.5,  # New PU
    "Cadillac":        5.0,
}
qualifying_2026["ReliabilityRiskScore"] = qualifying_2026["Team"].map(
    RELIABILITY_RISK
)

# ══════════════════════════════════════════════════════════
# 9. CIRCUIT SCORE
#    Now using AUS + CHN + JPN 2026 actual results
#    3 rounds of real data — much more reliable than before
# ══════════════════════════════════════════════════════════
# Actual 2026 finishing positions per driver per round
# DNF/DNS = 20, DNE = 22
RESULTS_2026 = {
    #                    AUS   CHN   JPN
    "ANT":              [2,    1,    1],
    "VER":              [20,   20,   8],   # Struggled early, recovering
    "LEC":              [3,    4,    3],
    "NOR":              [5,    20,   5],   # China DNS
    "RUS":              [1,    2,    4],
    "HAM":              [7,    3,    6],
    "PIA":              [22,   2,    2],   # AUS DNE, China P2, Japan P2
    "COL":              [20,   10,   20],  # Limited top 10 finishes
    "HAD":              [20,   8,    9],
    "GAS":              [10,   6,    10],
}

circuit_scores = {}
for code, results in RESULTS_2026.items():
    avg = np.mean(results)
    # Normalize to 1-5 scale
    normalized = 1 + (avg - 1) * (4 / 19)
    circuit_scores[code] = round(normalized, 3)

qualifying_2026["CircuitScore"] = qualifying_2026["DriverCode"].map(
    circuit_scores
).fillna(3.5)

# ══════════════════════════════════════════════════════════
# 10. SYNTHETIC SECTOR TIMES
#     Miami circuit split ratios (approximate)
#     S1: 28% — tight hairpins and acceleration
#     S2: 45% — longest section, stadium complex
#     S3: 27% — final chicane and back straight
# ══════════════════════════════════════════════════════════
S1_RATIO = 0.28
S2_RATIO = 0.45
S3_RATIO = 0.27

qualifying_2026["Sector1Time (s)"] = qualifying_2026["QualifyingTime (s)"] * S1_RATIO
qualifying_2026["Sector2Time (s)"] = qualifying_2026["QualifyingTime (s)"] * S2_RATIO
qualifying_2026["Sector3Time (s)"] = qualifying_2026["QualifyingTime (s)"] * S3_RATIO

# Race pace = qualifying * 1.07 (7% slower on race tyres)
# Additional wet penalty applied below
qualifying_2026["RacePace (s)"]    = qualifying_2026["QualifyingTime (s)"] * 1.07

# ══════════════════════════════════════════════════════════
# 11. WEATHER FEATURES
# ══════════════════════════════════════════════════════════
qualifying_2026["RainProbability"]   = RAIN_PROBABILITY
qualifying_2026["Temperature"]       = RACE_TEMP
qualifying_2026["TempDelta"]         = TEMP_DELTA
qualifying_2026["Humidity"]          = HUMIDITY
qualifying_2026["WindSpeed"]         = WIND_SPEED

print("\n📊 Full Feature Set:")
print(qualifying_2026[[
    "Driver", "QualifyingTime (s)", "GapFromPole (s)",
    "AdjustedTeamScore", "WetPerformanceFactor",
    "ERSDependencyScore", "MiamiBoostCapScore",
    "PoleWetBonus", "CircuitScore"
]].to_string(index=False))

# ══════════════════════════════════════════════════════════
# 12. FEATURE COLUMNS
# ══════════════════════════════════════════════════════════
FEATURE_COLS = [
    "QualifyingTime (s)",     # race pace baseline
    "GapFromPole (s)",        # qualifying hierarchy
    "AdjustedTeamScore",      # team strength — 3 rounds of data
    "GridPenalty (s)",        # grid penalty
    "WetPerformanceFactor",   # 🚨 80% rain — most critical feature
    "RainProbability",        # 80%
    "Temperature",            # 26°C race
    "TempDelta",              # -6°C (cooler race vs qualifying)
    "Humidity",               # 81%
    "WindSpeed",              # 16 km/h
    "ERSDependencyScore",     # 7MJ limit — 3rd consecutive cut
    "MiamiBoostCapScore",     # new +150kW race boost cap
    "ReliabilityRiskScore",   # mechanical risk
    "Sector1Time (s)",        # Miami S1 — hairpins
    "Sector2Time (s)",        # Miami S2 — stadium section
    "Sector3Time (s)",        # Miami S3 — back straight
    "CircuitScore",           # avg 2026 AUS+CHN+JPN finishes
    "SprintWinnerBoost",      # Norris won Sprint
    "PoleWetBonus",           # pole advantage amplified in wet (80% rain)
]
TARGET = "RacePace (s)"

# ══════════════════════════════════════════════════════════
# 13. TRAIN MODEL
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
# 14. PREDICT RACE
# ══════════════════════════════════════════════════════════
data = qualifying_2026.copy()
data["PredictedLapTime (s)"] = model.predict(X)

# Wet race adjustment — drivers with better wet performance
# get a bonus proportional to rain probability
data["WetBonus"] = (
    (1 - data["WetPerformanceFactor"]) * RAIN_PROBABILITY * 100
)
data["PredictedLapTime (s)"] -= data["WetBonus"]

# Pole wet bonus — pole sitter gets additional lap time advantage
# in heavy rain (>60%) due to clean air, spray, visibility advantage
# Conservative: 0.08s per lap at 80% rain — enough to tip close calls
data["PredictedLapTime (s)"] -= data["PoleWetBonus"]

# Sort by fastest predicted lap time
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 15. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 62)
print("  🏁  2026 MIAMI GP — PREDICTED RACE RESULT")
print("=" * 62)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 62)
print(f"\n  🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  Race: {RACE_TEMP}°C 🌧️")
print(f"  🌧️  Rain: {int(RAIN_PROBABILITY*100)}% 🚨  |  "
      f"💧 Humidity: {HUMIDITY}%  |  💨 Wind: {WIND_SPEED}km/h")
print(f"  🔋  ERS limit: 7MJ (3rd consecutive cut)")
print(f"  ⚡  Boost cap: +150kW race (new Miami rule)")
print(f"  🏆  Sprint winner: Lando Norris (McLaren)")
print(f"  🌟  Pole: Kimi Antonelli — 3 poles in 4 races!\n")

# ══════════════════════════════════════════════════════════
# 16. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 28), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 4: MIAMI GP\n"
    "MIAMI INTERNATIONAL AUTODROME  |  MAY 3, 2026  |  🌧️ 80% RAIN",
    fontsize=18, fontweight="bold", color="white",
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
ax1.set_title("📊 Predicted Race Finishing Order  (🌧️ Wet Race Conditions)",
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
wet_sorted = data.sort_values("WetPerformanceFactor")
wet_colors = [TEAM_COLORS.get(t, "#FFF") for t in wet_sorted["Team"]]
bars = ax2.barh(
    wet_sorted["Driver"][::-1],
    wet_sorted["WetPerformanceFactor"][::-1],
    color=wet_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title("💧 Wet Performance Factor\n(lower = better in wet — 80% rain today!)",
              fontsize=10, fontweight="bold", color="white",
              fontfamily=FONT, pad=10)
ax2.set_xlabel("Wet Factor (lower = elite wet driver)",
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
        row["GapFromPole (s)"] + 0.005, i,
        f"+{row['GapFromPole (s)']:.3f}s",
        va="center", fontsize=7.5,
        color="white", fontfamily=FONT
    )

# ── Chart 4: Feature Importance ──────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
feat_labels = [
    "Qualifying Time", "Gap From Pole", "Team Score",
    "Grid Penalty", "Wet Factor 🚨", "Rain Prob",
    "Temperature", "Temp Delta", "Humidity",
    "Wind Speed", "ERS Dependency", "Boost Cap",
    "Reliability", "Sector 1", "Sector 2", "Sector 3",
    "Circuit Score", "Sprint Boost", "Pole Wet Bonus 🌧️"  # ← add this
]
feat_import = model.feature_importances_
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

podium = data[data["PredictedPosition"] <= 3].sort_values(
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
    f"💧 Humidity: {HUMIDITY}%  |  "
    f"🔋 ERS: 7MJ  |  "
    f"⚡ Boost cap: +150kW  |  "
    f"🌧️ PoleWetBonus: {POLE_WET_BONUS_FACTOR * RAIN_PROBABILITY:.3f}s  |  "
    f"🏆 Sprint: Norris  |  "
    f"🌟 Pole: Antonelli",
    ha="center", fontsize=7.5, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_04_miami_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_04_miami_prediction.png")
plt.show()