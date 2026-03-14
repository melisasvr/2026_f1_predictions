"""
============================================================
 F1 PREDICTIONS 2026 — ROUND 2: CHINESE GP
 Shanghai International Circuit | Race Date: March 15, 2026
============================================================
 Model    : Gradient Boosting Regressor
 Target   : Average race LapTime (s) from 2025 Chinese GP
 Features : QualifyingTime (s), GapFromPole (s),
            AdjustedTeamScore, GridPenalty (s),
            RainProbability, TempDelta,
            Sector1Time (s), Sector2Time (s),
            Sector3Time (s), CircuitScore (normalized),
            SprintWinnerBoost
 Author   : F1 Predictions 2026
============================================================
"""

import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings("ignore")

# ── FastF1 Cache ───────────────────────────────────────────
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

print("=" * 60)
print("  🏎️  F1 PREDICTIONS 2026 — ROUND 2: CHINESE GP")
print("=" * 60)

# ══════════════════════════════════════════════════════════
# 1. WEATHER
# ══════════════════════════════════════════════════════════
QUALIFYING_TEMP  = 17    # °C — sunny
RACE_TEMP        = 17    # °C — cloudy
TEMP_DELTA       = RACE_TEMP - QUALIFYING_TEMP  # 0°C
RAIN_PROBABILITY = 0.25  # 25% — low impact expected

# ══════════════════════════════════════════════════════════
# 2. 2026 GP QUALIFYING DATA  (real Q3 results)
#    Antonelli — youngest polesitter in F1 history 🌟
# ══════════════════════════════════════════════════════════
POLE_TIME = 92.064  # Antonelli 1:32.064

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "Kimi Antonelli",
        "George Russell",
        "Lewis Hamilton",
        "Charles Leclerc",
        "Oscar Piastri",
        "Lando Norris",
        "Pierre Gasly",
        "Max Verstappen",
        "Isack Hadjar",
        "Oliver Bearman",
    ],
    "GridPosition": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "QualifyingTime (s)": [
        92.064,   # 1:32.064 — Antonelli POLE 🌟
        92.286,   # 1:32.286 — Russell
        92.415,   # 1:32.415 — Hamilton
        92.428,   # 1:32.428 — Leclerc
        92.550,   # 1:32.550 — Piastri
        92.608,   # 1:32.608 — Norris
        92.873,   # 1:32.873 — Gasly
        93.002,   # 1:33.002 — Verstappen
        93.121,   # 1:33.121 — Hadjar
        93.292,   # 1:33.292 — Bearman
    ],
    "Team": [
        "Mercedes", "Mercedes", "Ferrari",
        "Ferrari",  "McLaren",  "McLaren",
        "Alpine",   "Red Bull Racing",
        "Red Bull Racing", "Haas",
    ],
    "GridPenalty (s)":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "IsRookie":          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "SprintWinnerBoost": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Russell won sprint
    "PolesitterBoost":   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Antonelli pole
})

DRIVER_CODES = {
    "Kimi Antonelli":  "ANT",
    "George Russell":  "RUS",
    "Lewis Hamilton":  "HAM",
    "Charles Leclerc": "LEC",
    "Oscar Piastri":   "PIA",
    "Lando Norris":    "NOR",
    "Pierre Gasly":    "GAS",
    "Max Verstappen":  "VER",
    "Isack Hadjar":    "HAD",
    "Oliver Bearman":  "BEA",
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
#    Higher = stronger. Weighted 40% 2025 + 60% 2026 R01
# ══════════════════════════════════════════════════════════
ADJUSTED_TEAM_SCORE = {
    "Mercedes":        9.5,  # Dominant — R01 1-2, ANT pole + RUS sprint R02
    "Ferrari":         8.5,  # R01 P3 + unique rear wing aero advantage R02
    "McLaren":         5.5,  # 2025 champs but underperforming 2026 regs
    "Red Bull Racing": 5.5,  # Struggling with 2026 regs
    "Alpine":          5.0,  # Gasly Q3 — better than expected
    "Racing Bulls":    4.5,
    "Haas":            4.5,  # Bearman Q3 — upgraded
    "Aston Martin":    4.0,
    "Williams":        3.5,
    "Audi":            3.0,
    "Cadillac":        2.5,
}
qualifying_2026["AdjustedTeamScore"] = qualifying_2026["Team"].map(
    ADJUSTED_TEAM_SCORE
)

# ══════════════════════════════════════════════════════════
# 5. LOAD 2025 CHINESE GP RACE DATA
# ══════════════════════════════════════════════════════════
print("\n📡 Loading 2025 Chinese GP race data...")
session_2025 = fastf1.get_session(2025, "China", "R")
session_2025.load(telemetry=False, weather=False, messages=False)

# ── Finishing positions → CircuitScore (normalized 1-5) ──
results_2025 = session_2025.results[["Abbreviation", "Position"]].copy()
results_2025["Position"] = pd.to_numeric(
    results_2025["Position"], errors="coerce"
)
results_2025 = results_2025.rename(
    columns={"Abbreviation": "DriverCode", "Position": "CircuitScore"}
)

for driver, code in DRIVER_CODES.items():
    row = results_2025[results_2025["DriverCode"] == code]
    pos = (f"P{int(row['CircuitScore'].values[0])}"
           if not row.empty and
           not pd.isna(row["CircuitScore"].values[0])
           else "N/A")
    print(f"   2025 Chinese GP | {driver:<22} → {pos}")

# ── Sector + lap times ────────────────────────────────────
laps_2025 = session_2025.laps[[
    "Driver", "LapTime",
    "Sector1Time", "Sector2Time", "Sector3Time"
]].copy()
laps_2025.dropna(inplace=True)

for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2025[f"{col} (s)"] = laps_2025[col].dt.total_seconds()

# Average times per driver (training target = avg LapTime)
sector_avg = laps_2025.groupby("Driver")[[
    "Sector1Time (s)", "Sector2Time (s)",
    "Sector3Time (s)", "LapTime (s)"
]].mean().reset_index().rename(columns={"Driver": "DriverCode"})

print("\n✅ 2025 sector + lap times loaded successfully")

# ══════════════════════════════════════════════════════════
# 6. BUILD TRAINING SET
#    X = qualifying features + sector times + circuit history
#    y = average race LapTime (s) from 2025
#    This way the model learns pace → lap time
#    NOT history → history
# ══════════════════════════════════════════════════════════
train = sector_avg.copy()

# Merge in circuit score
train = train.merge(results_2025, on="DriverCode", how="left")
# Normalize CircuitScore 1-20 → 1-5 (just a mild historical signal)
train["CircuitScore"] = train["CircuitScore"].fillna(15)
train["CircuitScore"] = 1 + (train["CircuitScore"] - 1) * (4 / 19)

# Merge in 2026 qualifying features for matched drivers
train = train.merge(
    qualifying_2026[[
        "DriverCode", "QualifyingTime (s)", "GapFromPole (s)",
        "AdjustedTeamScore", "GridPenalty (s)", "SprintWinnerBoost",
        "PolesitterBoost"
    ]],
    on="DriverCode", how="left"
)

# Fill missing (drivers not in top 10 quali)
train["QualifyingTime (s)"]  = train["QualifyingTime (s)"].fillna(93.5)
train["GapFromPole (s)"]     = train["GapFromPole (s)"].fillna(1.5)
train["AdjustedTeamScore"]   = train["AdjustedTeamScore"].fillna(4.0)
train["GridPenalty (s)"]     = train["GridPenalty (s)"].fillna(0)
train["SprintWinnerBoost"]   = train["SprintWinnerBoost"].fillna(0)
train["PolesitterBoost"]     = train["PolesitterBoost"].fillna(0)
train["RainProbability"]     = RAIN_PROBABILITY
train["TempDelta"]           = TEMP_DELTA

train.dropna(subset=["LapTime (s)"], inplace=True)

# ══════════════════════════════════════════════════════════
# 7. FEATURE COLUMNS
#    y = LapTime (s) — race pace target
#    CircuitScore is just one small feature, not the target
# ══════════════════════════════════════════════════════════
FEATURE_COLS = [
    "QualifyingTime (s)",   # strong correlation with race pace
    "GapFromPole (s)",      # reinforces qualifying hierarchy
    "AdjustedTeamScore",    # team strength (higher = better)
    "GridPenalty (s)",      # grid penalty
    "RainProbability",      # weather — rain chance
    "TempDelta",            # weather — temp shift
    "Sector1Time (s)",      # acceleration zones
    "Sector2Time (s)",      # technical corners
    "Sector3Time (s)",      # high speed / straights
    "CircuitScore",         # mild historical signal (normalized)
    "SprintWinnerBoost",    # sprint winner pace signal
    "PolesitterBoost",      # polesitter momentum (Antonelli)
]
TARGET = "LapTime (s)"     # ← race pace, NOT finishing position

X = train[FEATURE_COLS].fillna(0)
y = train[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=38
)

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=38,
)
model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"\n🔍 Model MAE on test set: {mae:.2f} seconds")

# ══════════════════════════════════════════════════════════
# 8. MERGE PREDICTION FEATURES
# ══════════════════════════════════════════════════════════
data = qualifying_2026.copy()
data = data.merge(results_2025, on="DriverCode", how="left")
data["CircuitScore"] = data["CircuitScore"].fillna(15)
data["CircuitScore"] = 1 + (data["CircuitScore"] - 1) * (4 / 19)
data = data.merge(sector_avg, on="DriverCode", how="left")

sector_cols = [
    "Sector1Time (s)", "Sector2Time (s)",
    "Sector3Time (s)", "LapTime (s)"
]
for col in sector_cols:
    data[col] = data[col].fillna(data[col].mean())

data["RainProbability"]   = RAIN_PROBABILITY
data["TempDelta"]         = TEMP_DELTA
data["PolesitterBoost"]   = data["PolesitterBoost"].fillna(0)

print("\n📊 Full Feature Set:")
print(data[[
    "Driver", "QualifyingTime (s)", "GapFromPole (s)",
    "AdjustedTeamScore", "RainProbability",
    "CircuitScore", "SprintWinnerBoost", "PolesitterBoost"
]].to_string(index=False))

# ══════════════════════════════════════════════════════════
# 9. PREDICT 2026 RACE — predicted lap time → rank
# ══════════════════════════════════════════════════════════
X_pred = data[FEATURE_COLS].fillna(0)
data["PredictedLapTime (s)"] = model.predict(X_pred)

# ── 2026 Form Correction ──────────────────────────────────
# The model trains on 2025 sector times. At China 2025,
# McLaren dominated (Piastri P1, Norris P2) and Ferrari
# was off-pace (Leclerc P18, Hamilton P19).
# In 2026 the picture has flipped: Mercedes leads, Ferrari
# has a rear-wing aero advantage, McLaren is struggling.
# We apply a lap-time correction (seconds) per team to
# reflect 2026 form vs. their 2025 China baseline.
# Negative = faster than 2025 form suggests; positive = slower.
FORM_CORRECTION_2026 = {
    "Mercedes":        -0.40,  # dominant; pole + sprint
    "Ferrari":         -0.35,  # rear wing aero advantage
    "McLaren":         +0.55,  # struggling with 2026 regs
    "Red Bull Racing": +0.10,  # finding their feet
    "Alpine":          -0.05,  # Gasly solid
    "Haas":            +0.00,  # neutral
}
data["FormCorrection"] = data["Team"].map(FORM_CORRECTION_2026).fillna(0)
data["PredictedLapTime (s)"] += data["FormCorrection"]

# Rookie penalty — add time
data.loc[data["IsRookie"] == 1, "PredictedLapTime (s)"] += 0.5

# Sort by fastest predicted lap time
data = data.sort_values("PredictedLapTime (s)").reset_index(drop=True)
data["PredictedPosition"] = data.index + 1

# ══════════════════════════════════════════════════════════
# 10. PRINT RESULTS
# ══════════════════════════════════════════════════════════
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
print("\n" + "=" * 60)
print("  🏁  2026 CHINESE GP — PREDICTED RACE RESULT")
print("=" * 60)
print(f"  {'Pos':<5} {'Driver':<22} {'Team':<18} {'Pred Lap (s)':>12}")
print("  " + "-" * 60)
for _, row in data.iterrows():
    pos  = int(row["PredictedPosition"])
    icon = medals.get(pos, f"P{pos} ")
    print(f"  {icon:<5} {row['Driver']:<22} {row['Team']:<18}"
          f" {row['PredictedLapTime (s)']:>12.3f}")
print("=" * 60)
print(f"\n  🌡️  Qualifying: {QUALIFYING_TEMP}°C ☀️  →  "
      f"Race Day: {RACE_TEMP}°C ☁️  (Δ {TEMP_DELTA}°C)")
print(f"  🌧️  Rain probability: {int(RAIN_PROBABILITY*100)}% "
      f"(low impact expected)")
print(f"  🏆  Sprint winner: George Russell")
print(f"  🌟  Pole: Kimi Antonelli — youngest polesitter in F1 history!\n")

# ══════════════════════════════════════════════════════════
# 11. VISUALISATIONS
# ══════════════════════════════════════════════════════════
plt.style.use("dark_background")
FONT = "monospace"

driver_colors = [TEAM_COLORS.get(t, "#FFFFFF") for t in data["Team"]]

fig = plt.figure(figsize=(20, 26), facecolor="#0f0f0f")
fig.suptitle(
    "🏎️  F1 2026 — ROUND 2: CHINESE GP\n"
    "SHANGHAI INTERNATIONAL CIRCUIT  |  MARCH 15, 2026",
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
ax1.set_title("📊 Predicted Race Finishing Order (by Race Lap Time)",
              fontsize=13, fontweight="bold", color="white",
              fontfamily=FONT, pad=12)
ax1.set_xlabel("Predicted Average Lap Time (s) — lower = faster",
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

# ── Chart 2: Qualifying Gap to Pole ──────────────────────
ax2 = fig.add_subplot(gs[1, 0])
qual_sorted = qualifying_2026.sort_values("GapFromPole (s)")
qual_colors = [TEAM_COLORS.get(t, "#FFF") for t in qual_sorted["Team"]]
ax2.barh(
    qual_sorted["Driver"][::-1],
    qual_sorted["GapFromPole (s)"][::-1],
    color=qual_colors[::-1],
    edgecolor="white", linewidth=0.4, height=0.65
)
ax2.set_title("⏱️  Qualifying Gap to Pole (Real Q3 Times)",
              fontsize=11, fontweight="bold", color="white",
              fontfamily=FONT, pad=10)
ax2.set_xlabel("Gap to Pole (seconds)", color="#AAAAAA",
               fontsize=9, fontfamily=FONT)
ax2.tick_params(colors="white", labelsize=8)
ax2.set_facecolor("#1a1a1a")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333333")
for i, (_, row) in enumerate(qual_sorted[::-1].iterrows()):
    ax2.text(
        row["GapFromPole (s)"] + 0.005, i,
        f"+{row['GapFromPole (s)']:.3f}s",
        va="center", fontsize=7.5,
        color="white", fontfamily=FONT
    )

# ── Chart 3: Sector Times ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
sectors = data[[
    "Driver", "Sector1Time (s)",
    "Sector2Time (s)", "Sector3Time (s)"
]].set_index("Driver")
x     = np.arange(len(sectors))
width = 0.25
ax3.bar(x - width, sectors["Sector1Time (s)"],
        width, label="S1", color="#FF8000", alpha=0.85)
ax3.bar(x,          sectors["Sector2Time (s)"],
        width, label="S2", color="#00D2BE", alpha=0.85)
ax3.bar(x + width, sectors["Sector3Time (s)"],
        width, label="S3", color="#DC0000", alpha=0.85)
ax3.set_title("⏱️  2025 Avg Sector Times (Training Data)",
              fontsize=11, fontweight="bold", color="white",
              fontfamily=FONT, pad=10)
ax3.set_ylabel("Time (seconds)", color="#AAAAAA",
               fontsize=9, fontfamily=FONT)
ax3.set_xticks(x)
ax3.set_xticklabels(
    [d.split()[-1] for d in sectors.index],
    rotation=45, ha="right", fontsize=8, color="white"
)
ax3.tick_params(colors="white", labelsize=8)
ax3.set_facecolor("#1a1a1a")
ax3.legend(fontsize=8, facecolor="#1a1a1a",
           edgecolor="#444", labelcolor="white")
for spine in ax3.spines.values():
    spine.set_edgecolor("#333333")

# ── Chart 4: Feature Importance ──────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
feat_labels = [
    "Qualifying Time", "Gap From Pole", "Team Score",
    "Grid Penalty", "Rain Prob", "Temp Delta",
    "Sector 1", "Sector 2", "Sector 3",
    "Circuit Score", "Sprint Boost", "Polesitter Boost"
]
feat_import  = model.feature_importances_
feat_colors2 = [
    "#FF8000", "#00D2BE", "#DC0000", "#3671C6",
    "#6692FF", "#FF87BC", "#B6BABD", "#64C4FF",
    "#358C75", "#B8B8B8", "#C8102E", "#FFD700"
]
sorted_idx    = np.argsort(feat_import)
sorted_labels = [feat_labels[i] for i in sorted_idx]
sorted_values = feat_import[sorted_idx]
sorted_colors = [feat_colors2[i] for i in sorted_idx]
ax4.barh(sorted_labels, sorted_values,
         color=sorted_colors,
         edgecolor="white", linewidth=0.4, height=0.5)
ax4.set_title("🤖 Model Feature Importance",
              fontsize=11, fontweight="bold", color="white",
              fontfamily=FONT, pad=10)
ax4.set_xlabel("Importance Score", color="#AAAAAA",
               fontsize=9, fontfamily=FONT)
ax4.tick_params(colors="white", labelsize=8)
ax4.set_facecolor("#1a1a1a")
for spine in ax4.spines.values():
    spine.set_edgecolor("#333333")
for i, v in enumerate(sorted_values):
    ax4.text(v + 0.002, i, f"{v:.3f}",
             va="center", fontsize=8,
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
ax5.set_title("🏆 Predicted Podium",
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
             fontsize=13, fontweight="bold",
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
    f"🔍 Model MAE: {mae:.2f}s  |  "
    f"🌧️ Rain: {int(RAIN_PROBABILITY*100)}%  |  "
    f"🌡️ Temp Δ: {TEMP_DELTA}°C  |  "
    f"🏆 Sprint: Russell  |  "
    f"🌟 Pole: Antonelli (youngest ever!)  |  🔴 Ferrari rear wing advantage",
    ha="center", fontsize=8, color="#888888", fontfamily=FONT
)

plt.savefig(
    "round_02_china_prediction.png",
    dpi=150, bbox_inches="tight",
    facecolor="#0f0f0f"
)
print("✅ Chart saved → round_02_china_prediction.png")
plt.show()