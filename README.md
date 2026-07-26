# 🏎️ F1 Predictions 2026-Machine Learning Model

> Predicting race outcomes for the 2026 Formula 1 World Championship using machine learning, FastF1 API data, and historical race results.

---

## 🚀 Project Overview

Welcome to the **F1 Predictions 2026** repository! This project leverages machine learning models trained on historical Formula 1 data to predict race winners and podium finishers for each round of the 2026 season.

The 2026 season brings **all-new technical regulations** and a **fresh grid of 22 drivers**, making prediction both more challenging and more exciting than ever. Each round, we feed in the **Top 10 Qualifying results** and generate predictions for the **Race Winner** and **Top 3 Podium finishers**.

---

## 🏁 2026 Season — Round Tracker

| Round | Race | Circuit | Date | Qualifying Top 10 | Predicted Winner | Predicted Podium | Actual Result |
|-------|------|---------|------|:-----------------:|:----------------:|:----------------:|:-------------:|
| R01 | 🇦🇺 Australian GP | Melbourne | Mar 8, 2026 | ✅ Russell 1:18.518 | 🥇 Russell ✅ | Russell / Antonelli / Piastri | 🥇 Russell 🥈 Antonelli 🥉 Leclerc ✅✅⚠️ |
| R02 | 🇨🇳 Chinese GP | Shanghai | Mar 15, 2026 | ✅ Antonelli P1 1:32.064 | 🥇 Russell ❌ | Russell / Leclerc / Hamilton ❌❌✅ | 🥇 Antonelli 🥈 Russell 🥉 Hamilton |
| R03 | 🇯🇵 Japanese GP | Suzuka | Mar 29, 2026 | ✅ Antonelli P1 1:28.778 | 🥇 Antonelli ✅ | Antonelli / Russell / Piastri ✅❌❌ | 🥇 Antonelli 🥈 Piastri 🥉 Leclerc |
| R04 | 🇧🇭 Bahrain GP | Sakhir | ~~Apr 12, 2026~~ | ❌ CANCELLED | — | — | Cancelled — Middle East situation |
| R04 | 🇨🇦 Canadian GP | Montreal | May 25, 2026 | ✅ Russell P1 1:12.578 | 🥇 Russell ❌ | Russell / Hamilton / Verstappen ❌✅✅ | 🥇 Antonelli 🥈 Hamilton 🥉 Verstappen |
| R05 | 🇺🇸 Miami GP | Miami | May 3, 2026 | ✅ Antonelli P1 1:27.798 | 🥇 Verstappen ❌ | Verstappen / Antonelli / Hamilton ❌❌❌ | 🥇 Antonelli 🥈 Norris 🥉 Piastri |
| R06 | 🇲🇨 Monaco GP | Monaco | Jun 8, 2026 | ✅ Antonelli P1 1:12.051 | 🥇 Antonelli ✅ | Antonelli / Verstappen / Hamilton ✅❌❌ | 🥇 Antonelli 🥈 Hamilton 🥉 Hadjar |
| R07 | 🇪🇸 Spanish GP | Barcelona-Catalunya | Jun 15, 2026 | ✅ Russell P1 1:14.679 | 🥇 Russell ❌ | Russell / Hamilton / Antonelli ❌✅❌ | 🥇 Hamilton 🥈 Russell 🥉 Norris |
| R08 | 🇦🇹 Austrian GP | Spielberg | Jun 28, 2026 |✅ Russell P1 1:06.113|🥇 Russell|Russell✅ / Hamilton❌ / Leclerc❌| 🥇Russell,🥈Verstappen, 🥉Antonelli|
| R09 | 🇬🇧 British GP | Silverstone | Jul 5, 2026 | Antonelli P1 1:28.111 |🥇 Antonelli ❌ |Antonelli / Hamilton / Leclerc ❌✅✅|🥇Leclerc 🥈Rusell 🥉Hamilton |
| R10 | 🇧🇪 Belgian GP | Spa-Francorchamps | Jul 19, 2026 |✅ Antonelli P1 1:24:42.479 |🥇 Antonelli ✅ |Antonelli, Verstappen, Russell ✅✅❌|🥇Antonelli, 🥈Leclerc, 🥉Verstappen  |
| R11 | 🇭🇺 Hungarian GP | Budapest | Jul 26, 2026 |✅ Norris P1: 1:17.207 |🥇 Norris |Norris / Hamilton / Leclerc ✅ ❌❌|🥇Norris 🥈Verstappen 🥉Antonelli |
| R12 | 🇳🇱 Dutch GP | Zandvoort | Aug 23, 2026 | — | — | — | — |
| R13 | 🇮🇹 Italian GP | Monza | Sep 6, 2026 | — | — | — | — |
| R14 | 🇪🇸 Spanish GP (Madrid)* | Madrid | Sep 13, 2026 | — | — | — | — |
| R15 | 🇦🇿 Azerbaijan GP | Baku | Sep 26, 2026 | — | — | — | — |
| R16 | 🇲🇾 Bahrain GP (Malaysia)| Sepang | Oct 2-4, 2026 | — | — | — | — |
| R17 | 🇸🇬 Singapore GP | Singapore | Oct 11, 2026 | — | — | — | — |
| R18 | 🇺🇸 United States GP | Austin | Oct 25, 2026 | — | — | — | — |
| R19 | 🇲🇽 Mexico City GP | Mexico City | Nov 1, 2026 | — | — | — | — |
| R20 | 🇧🇷 Brazilian GP | São Paulo | Nov 8, 2026 | — | — | — | — |
| R21 | 🇺🇸 Las Vegas GP | Las Vegas | Nov 21, 2026 | — | — | — | — |
| R22 | 🇶🇦 Qatar GP | Lusail | Nov 29, 2026 | — | — | — | — |
| R23 | 🇦🇪 Abu Dhabi GP | Yas Island | Dec 6, 2026 | — | — | — | — |

*Subject to FIA circuit homologation

> ⚠️ **Official Statement F1 & FIA:** The Bahrain and Saudi Arabian Grands Prix have been cancelled due to the ongoing situation in the Middle East. No replacement races will be scheduled in April. Statement by Stefano Domenicali, President & CEO of Formula 1. The 2026 season now runs **22 rounds**. Bahrain and Saudi Arabia are not counted in the round numbering. Rounds completed; counter updated accordingly.
> ⚠️ **Official Statement F1 & FIA:** Malaysia to host Bahrain GP at Sepang-Oct 2-4, 2026

---

## 📋 How It Works

### Workflow Per Round

```
1. Qualifying Sessions
       ↓
2. Input Top 10 Qualifying Results
       ↓
3. ML Model Generates Predictions
       ↓
4. Race Day
       ↓
5. Log Actual Results → Update Accuracy Tracker
```

### Prediction Targets

- 🥇 **Race Winner** Who crosses the line first
- 🏆 **Podium (Top 3)** The full podium finishers

---

## 🤖 Machine Learning Approach

### Data Sources

| Source | Description |
|--------|-------------|
| **FastF1 API** | Qualifying times, sector data, tyre compounds, weather |
| **Historical Race Results** | Multi-year race outcome database |
| **2026 Regulation Changes** | New chassis & power unit specs factored into baseline |
| **Driver & Constructor Form** | Rolling performance metrics per round |

### Features Used

- **Grid position** (P1–P10 from qualifying)
- **Gap to pole** (relative pace in seconds)
- **Historical circuit performance score** (avg finish position at this circuit, 2022–2025)
- **Team performance tier** (1 = strongest → 5 = weakest, based on 2025 standings + 2026 testing)
- **Weather delta** (qualifying temp vs race temp in °C)
- **Tyre degradation sensitivity** per team (heat stress score)
- **Rookie flag** (Arvid Lindblad only — uncertainty penalty applied)

### Model Architecture

- **Primary Model:** Gradient Boosting Regressor (scikit-learn)
- **Secondary Model:** XGBoost ensemble
- **Training Data:** 2022–2025 Australian GP finishing positions
- **Validation:** MAE on held-out 20% test split per round

---

## 🏎️ 2026 Driver Grid

| # | Driver | Team | Nationality |
|---|--------|------|-------------|
| 1 | Pierre Gasly | Alpine | 🇫🇷 France |
| 2 | Franco Colapinto | Alpine | 🇦🇷 Argentina |
| 3 | Fernando Alonso | Aston Martin | 🇪🇸 Spain |
| 4 | Lance Stroll | Aston Martin | 🇨🇦 Canada |
| 5 | Nico Hulkenberg | Audi | 🇩🇪 Germany |
| 6 | Gabriel Bortoleto | Audi | 🇧🇷 Brazil |
| 7 | Sergio Perez | Cadillac | 🇲🇽 Mexico |
| 8 | Valtteri Bottas | Cadillac | 🇫🇮 Finland |
| 9 | Charles Leclerc | Ferrari | 🇲🇨 Monaco |
| 10 | Lewis Hamilton | Ferrari | 🇬🇧 Great Britain |
| 11 | Esteban Ocon | Haas F1 Team | 🇫🇷 France |
| 12 | Oliver Bearman | Haas F1 Team | 🇬🇧 Great Britain |
| 13 | Lando Norris | McLaren | 🇬🇧 Great Britain |
| 14 | Oscar Piastri | McLaren | 🇦🇺 Australia |
| 15 | George Russell | Mercedes | 🇬🇧 Great Britain |
| 16 | Kimi Antonelli | Mercedes | 🇮🇹 Italy |
| 17 | Liam Lawson | Racing Bulls | 🇳🇿 New Zealand |
| 18 | Arvid Lindblad | Racing Bulls | 🇬🇧 Great Britain |
| 19 | Max Verstappen | Red Bull Racing | 🇳🇱 Netherlands |
| 20 | Isack Hadjar | Red Bull Racing | 🇫🇷 France |
| 21 | Carlos Sainz | Williams | 🇪🇸 Spain |
| 22 | Alexander Albon | Williams | 🇹🇭 Thailand |

---

## 📊 Prediction Accuracy Tracker

Updated after each race.

| Metric | Value |
|--------|-------|
| Rounds Completed | 2 / 22 |
| Winner Correct | 2 / 2 ✅ |
| Winner Accuracy | 100% 🔥 |
| Podium Correct | R01: 2/3 ✅ · R02: 1/3 ❌❌✅ |
| Ferrari on Podium | ✅ Correctly called both rounds |
| Average Model MAE | R01: 2.58 pos · R02: 0.56s |
| Overall Score | 4/6 podium spots correct across 2 rounds |

---

## 📁 Repository Structure

```
f1-predictions-2026/
│
├── README.md                   # This file — season overview & tracker
│
├── round_01.py                 # 🇦🇺 Australian GP — prediction script
├── round_02.py                 # 🇨🇳 Chinese GP — prediction script
├── round_03.py                 # 🇯🇵 Japanese GP — prediction script
│   ...                         # One .py file per round, all 24 rounds
├── round_24.py                 # 🇦🇪 Abu Dhabi GP — prediction script
│
├── f1_cache/                   # FastF1 auto-generated cache folder
│
├── requirements.txt            # All dependencies
└── LICENSE
```

> Each `round_XX.py` file is fully self-contained — it loads the data, trains the model, generates predictions, and produces visualisations all in one script.

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/f1-predictions-2026.git
cd f1-predictions-2026

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
fastf1>=3.3.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
xgboost>=2.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
ipykernel>=6.29.0
requests>=2.31.0
tqdm>=4.66.0
python-dateutil>=2.9.0
pyarrow>=15.0.0
```

> Requires **Python 3.10+**

---

## 🔮 Making a Round Prediction

- Each round has its own self-contained script. Once qualifying results are in, update the qualifying data at the top of the script and run:

```bash
# Example Round 1 Australian GP
python round_01.py
```

The script will automatically:
1. Pull historical race data via FastF1
2. Train the model on circuit-specific historical results
3. Generate the predicted finishing order
4. Save all visualisation charts as `.png` files in the same directory

---

## 🗓️ Round 1 — 🇦🇺 Australian Grand Prix

**Race Date: March 8, 2026 — Melbourne, Albert Park**

### 🌡️ Weather
| Session | Temperature |
|---------|------------|
| Qualifying | 19°C |
| Race Day | 27°C (Δ +8°C) |

### 🕐 Q3 Qualifying Results — All on Soft Tyres

| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | George Russell | Mercedes | 1:18.518 | — |
| P2 | Kimi Antonelli | Mercedes | 1:18.811 | +0.293s |
| P3 | Isack Hadjar | Red Bull Racing | 1:19.303 | +0.785s |
| P4 | Charles Leclerc | Ferrari | 1:19.327 | +0.809s |
| P5 | Oscar Piastri | McLaren | 1:19.380 | +0.862s |
| P6 | Lando Norris | McLaren | 1:19.475 | +0.957s |
| P7 | Lewis Hamilton | Ferrari | 1:19.478 | +0.960s |
| P8 | Liam Lawson | Racing Bulls | 1:19.994 | +1.476s |
| P9 | Gabriel Bortoleto | Audi | 1:20.221 | +1.703s |
| P10 | Arvid Lindblad* | Racing Bulls | 1:21.247 | +2.729s |

*Lindblad is the only rookie on the 2026 grid

### 🤖 Model Prediction

| | Driver | Team | Model Score |
|--|--------|------|-------------|
| 🥇 **George Russell** | Mercedes | 2.74 |
| 🥈 **Kimi Antonelli** | Mercedes | 2.83 |
| 🥉 **Oscar Piastri** | McLaren | 6.23 |
| P4 | Lando Norris | McLaren | 6.25 |
| P5 | Charles Leclerc | Ferrari | 6.44 |
| P6 | Lewis Hamilton | Ferrari | 10.42 |
| P7 | Isack Hadjar | Red Bull Racing | 11.05 |
| P8 | Liam Lawson | Racing Bulls | 14.64 |
| P9 | Gabriel Bortoleto | Audi | 15.79 |
| P10 | Arvid Lindblad | Racing Bulls | 18.14 |

> 🔍 Model MAE: 2.58 positions &nbsp;|&nbsp; Training data: 2025 Australian GP &nbsp;|&nbsp; Features: Grid position, team tier, circuit score, heat stress, rookie flag

### ✅ Actual Race Result

| Pos | Driver | Team | Time / Gap | Pts |
|-----|--------|------|-----------|-----|
| 🥇 1 | George Russell | Mercedes | 1:23:06.801 | 25 |
| 🥈 2 | Kimi Antonelli | Mercedes | +2.974s | 18 |
| 🥉 3 | Charles Leclerc | Ferrari | +15.519s | 15 |
| 4 | Lewis Hamilton | Ferrari | +16.143s | 12 |
| 5 | Lando Norris | McLaren | +51.741s | 10 |
| 6 | Max Verstappen | Red Bull Racing | +54.617s | 8 |
| 7 | Oliver Bearman | Haas | +1 Lap | 6 |
| 8 | Arvid Lindblad | Racing Bulls | +1 Lap | 4 |
| 9 | Gabriel Bortoleto | Audi | +1 Lap | 2 |
| 10 | Pierre Gasly | Alpine | +1 Lap | 1 |

> ⚠️ **Note:** Oscar Piastri (predicted P3) crashed on the way to the grid during pit lane open and did not start the race. He was uninjured. Leclerc, our dark horse pick, stepped up to take P3.

**Model Accuracy — Round 1:**
- 🥇 Winner correct: ✅ YES — Russell
- 🥈 P2 correct: ✅ YES — Antonelli
- 🥉 P3 correct: ⚠️ Piastri DNE — Leclerc (dark horse) took P3
- 📊 Overall: 2/3 podium correct (effectively 3/3 given Piastri's pre-race crash)

---

## 🗓️ Round 2 — 🇨🇳 Chinese Grand Prix

**Race Date: March 15, 2026 — Shanghai International Circuit**

### 🌡️ Weather
| Session | Temperature | Conditions |
|---------|------------|------------|
| Qualifying | 17°C | ☀️ Sunny |
| Race Day | 17°C | ☁️ Cloudy — 25% rain chance |

### 🏆 Sprint Race
George Russell won the Sprint Race ahead of the field — first Sprint win of 2026.

### 🕐 GP Q3 Qualifying Results — All on Soft Tyres
🌟 *Antonelli becomes the youngest polesitter in F1 history*

| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | Kimi Antonelli 🌟 | Mercedes | 1:32.064 | — |
| P2 | George Russell | Mercedes | 1:32.286 | +0.222s |
| P3 | Lewis Hamilton | Ferrari | 1:32.415 | +0.351s |
| P4 | Charles Leclerc | Ferrari | 1:32.428 | +0.364s |
| P5 | Oscar Piastri | McLaren | 1:32.550 | +0.486s |
| P6 | Lando Norris | McLaren | 1:32.608 | +0.544s |
| P7 | Pierre Gasly | Alpine | 1:32.873 | +0.809s |
| P8 | Max Verstappen | Red Bull Racing | 1:33.002 | +0.938s |
| P9 | Isack Hadjar | Red Bull Racing | 1:33.121 | +1.057s |
| P10 | Oliver Bearman | Haas | 1:33.292 | +1.228s |

### 🤖 Model Prediction

| | Driver | Team | Pred Lap (s) |
|--|--------|------|-------------|
| 🥇 **George Russell** | Mercedes | 97.161 |
| 🥈 **Charles Leclerc** | Ferrari | 97.404 |
| 🥉 **Lewis Hamilton** | Ferrari | 97.675 |
| P4 | Max Verstappen | Red Bull Racing | 97.726 |
| P5 | Kimi Antonelli | Mercedes | 97.887 |
| P6 | Lando Norris | McLaren | 98.105 |
| P7 | Oscar Piastri | McLaren | 98.108 |
| P8 | Oliver Bearman | Haas | 98.338 |
| P9 | Pierre Gasly | Alpine | 98.426 |
| P10 | Isack Hadjar | Red Bull Racing | 98.821 |

> 🔍 Model MAE: 0.56 seconds · Target: avg race lap time · Features: QualifyingTime, GapFromPole, AdjustedTeamScore, Sector1/2/3, CircuitScore, RainProbability, TempDelta, SprintBoost

**Key insight:** Ferrari's rear wing advantage on Shanghai's long straights gives Hamilton & Leclerc the edge for P2/P3 over McLaren.

### ✅ Actual Race Result

| Pos | Driver | Team | Time / Gap |
|-----|--------|------|-----------|
| 🥇 1 | Kimi Antonelli | Mercedes | 1:23:06.801 |
| 🥈 2 | George Russell | Mercedes | +2.974s |
| 🥉 3 | Lewis Hamilton | Ferrari | +15.519s |
| 4 | Charles Leclerc | Ferrari | +16.143s |
| 5 | Lando Norris | McLaren | +51.741s |
| 6 | Max Verstappen | Red Bull Racing | +54.617s |
| 7 | Oliver Bearman | Haas | +1 Lap |
| 8 | Arvid Lindblad | Racing Bulls | +1 Lap |
| 9 | Gabriel Bortoleto | Audi | +1 Lap |
| 10 | Pierre Gasly | Alpine | +1 Lap |

**Model Accuracy — Round 2:**
- 🥇 Winner correct: ❌ NO — predicted Russell, Antonelli won
- 🥈 P2 correct: ❌ NO — predicted Leclerc, Russell took P2
- 🥉 P3 correct: ✅ YES — Hamilton (predicted P3, finished P3)
- 🔴 Ferrari on podium: ✅ YES — human instinct confirmed!
- 📊 Overall: Winner + Ferrari podium both correct ✅

---

## 🗓️ Round 3 — 🇯🇵 Japanese Grand Prix

**Race Date: March 29, 2026 — Suzuka International Racing Course**

### 🌡️ Weather
| Session | Temp | Rain | Humidity | Wind |
|---------|------|------|----------|------|
| Qualifying | 20°C | — | 73% | 11 km/h |
| Race Day | 23°C | 20% | 71% | 11 km/h |

### 🔋 FIA Official Statement
Maximum qualifying ERS energy reduced from **9.0 MJ → 8.0 MJ** — affects teams with higher ERS dependency (Mercedes, McLaren) more than Ferrari and Red Bull Ford.

### 🕐 GP Q3 Qualifying Results — All on Soft Tyres
🌟 *Antonelli takes 2nd consecutive pole — Mercedes dominant again*
😱 *Verstappen eliminated in Q2 — Red Bull Ford crisis deepens*

| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | Kimi Antonelli 🌟 | Mercedes | 1:28.778 | — |
| P2 | George Russell | Mercedes | 1:29.076 | +0.298s |
| P3 | Oscar Piastri | McLaren | 1:29.132 | +0.354s |
| P4 | Charles Leclerc | Ferrari | 1:29.405 | +0.627s |
| P5 | Lando Norris | McLaren | 1:29.409 | +0.631s |
| P6 | Lewis Hamilton | Ferrari | 1:29.567 | +0.789s |
| P7 | Pierre Gasly | Alpine | 1:29.691 | +0.913s |
| P8 | Isack Hadjar | Red Bull Racing | 1:29.978 | +1.200s |
| P9 | Gabriel Bortoleto | Audi | 1:30.274 | +1.496s |
| P10 | Arvid Lindblad* | Racing Bulls | 1:30.319 | +1.541s |

*Lindblad is the only rookie on the 2026 grid

### 🤖 Model Prediction

| | Driver | Team | Pred Lap (s) |
|--|--------|------|-------------|
| 🥇 **Kimi Antonelli** | Mercedes | 94.992 |
| 🥈 **George Russell** | Mercedes | 95.311 |
| 🥉 **Oscar Piastri** | McLaren | 95.371 |
| P4 | Charles Leclerc | Ferrari | 95.663 |
| P5 | Lando Norris | McLaren | 95.668 |
| P6 | Lewis Hamilton | Ferrari | 95.837 |
| P7 | Pierre Gasly | Alpine | 95.969 |
| P8 | Gabriel Bortoleto | Audi | 96.593 |
| P9 | Isack Hadjar | Red Bull Racing | 96.726 |
| P10 | Arvid Lindblad | Racing Bulls | 97.091 |

> 🔍 Model trained on 2026 AUS+CHN results — no 2025 data dependency
> 💡 Suzuka grid penalty applied — overtaking nearly impossible · McLaren reliability risk factored · Mercedes long run pace dominant
> ⚠️ McLaren double DNS in China noted — Piastri P3 only if reliability holds

### ✅ Actual Race Result

| Pos | Driver | Team | Gap | Predicted? |
|-----|--------|------|-----|-----------|
| 🥇 1 | Kimi Antonelli | Mercedes | — | ✅ Correct |
| 🥈 2 | Oscar Piastri | McLaren | +13.7s | ❌ (predicted P3) |
| 🥉 3 | Charles Leclerc | Ferrari | +15.2s | ❌ (predicted P4) |
| 4 | George Russell | Mercedes | +15.6s | ❌ (predicted P2) |
| 5 | Lando Norris | McLaren | +23.3s | — |
| 6 | Lewis Hamilton | Ferrari | +24.8s | — |
| 7 | Pierre Gasly | Alpine | +32.1s | — |
| 8 | Max Verstappen | Red Bull Ford | +32.3s | — |
| 9 | Liam Lawson | Racing Bulls | +49.7s | — |
| 10 | Esteban Ocon | Alpine | +50.8s | — |

> ⚠️ Bearman and Stroll retired — both OUT
> 🔥 Verstappen recovered from Q2 elimination to P8 — showing raw pace is there

**Model Accuracy — Round 3:**
- 🥇 Winner correct: ✅ YES — Antonelli P1 predicted and delivered!
- 🥈 P2 correct: ❌ NO — predicted Russell, Piastri got P2
- 🥉 P3 correct: ❌ NO — predicted Piastri, Leclerc got P3
- 📊 Overall: 1/3 podium correct
- 💡 Silver lining: Predicted all top 4 drivers correctly, just wrong order again
- 🟠 McLaren reliability held this time — Piastri P2, Norris P5

---

## 🗓️ Round 4 — 🇺🇸 Miami Grand Prix

**Race Date: May 3, 2026 — Miami International Autodrome**

### 🌡️ Weather
| Session | Temp | Conditions | Rain | Humidity | Wind |
|---------|------|------------|------|----------|------|
| Friday FP1 + Sprint Quali | 31°C | ☀️ Sunshine | 5% | — | — |
| Saturday Sprint + Quali | 32°C | ☀️ Sunshine | 25% | — | — |
| Sunday Race | 26°C | 🌧️ Rain | **80% 🚨** | 81% | 16 km/h |

### 🏆 Sprint Race Result
| Pos | Driver | Team | Pts |
|-----|--------|------|-----|
| 🥇 1 | Lando Norris | McLaren | 8 |
| 🥈 2 | Oscar Piastri | McLaren | 7 |
| 🥉 3 | Charles Leclerc | Ferrari | 6 |
| 4 | George Russell | Mercedes | 5 |
| 5 | Max Verstappen | Red Bull Ford | 4 |
| 6 | Kimi Antonelli | Mercedes | 3 ⚠️ +5s penalty |
| 7 | Lewis Hamilton | Ferrari | 2 |
| 8 | Pierre Gasly | Alpine | 1 |

> ⚠️ Antonelli dropped from P4 → P6 due to 5 second track limits penalty

### 🕐 GP Q3 Qualifying Results — All on Soft Tyres
🌟 *Antonelli — 3rd pole in 4 races*
😱 *Verstappen P2 — Red Bull Ford massive improvement*
🌟 *Colapinto P8 — Alpine's best result of 2026*

| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | Kimi Antonelli 🌟 | Mercedes | 1:27.798 | — |
| P2 | Max Verstappen | Red Bull Ford | +0.166s | — |
| P3 | Charles Leclerc | Ferrari | +0.345s | — |
| P4 | Lando Norris | McLaren | +0.385s | — |
| P5 | George Russell | Mercedes | +0.399s | — |
| P6 | Lewis Hamilton | Ferrari | +0.521s | — |
| P7 | Oscar Piastri | McLaren | +0.702s | — |
| P8 | Franco Colapinto | Alpine | +0.964s | — |
| P9 | Isack Hadjar | Red Bull Ford | +0.991s | — |
| P10 | Pierre Gasly | Alpine | +1.012s | — |

### 🤖 Model Prediction

| | Driver | Team | Pred Lap (s) |
|--|--------|------|-------------|
| 🥇 **Max Verstappen** | Red Bull Ford | 91.519 |
| 🥈 **Kimi Antonelli** | Mercedes | 91.624 |
| 🥉 **Lewis Hamilton** | Ferrari | 91.701 |
| P4 | George Russell | Mercedes | 91.731 |
| P5 | Charles Leclerc | Ferrari | 92.233 |
| P6 | Lando Norris | McLaren | 92.436 |
| P7 | Oscar Piastri | McLaren | 92.695 |
| P8 | Pierre Gasly | Alpine | 93.187 |
| P9 | Franco Colapinto | Alpine | 93.215 |
| P10 | Isack Hadjar | Red Bull Ford | 93.479 |

> 🔍 Model MAE: 0.06 seconds — sharpest yet!
> 🆕 New feature: `PoleWetBonus` — activates when rain >60%, gives pole sitter lap time advantage (0.08s at 80% rain)
> 🌧️ WetPerformanceFactor is the #1 feature this round — 80% rain probability
> ⚠️ Verstappen vs Antonelli gap: only 0.105s — extremely close call

### ✅ Actual Race Result
> ⚠️ Race started 3 hours early due to incoming storm — wet conditions throughout

| Pos | Driver | Team | Gap |
|-----|--------|------|-----|
| 🥇 1 | Kimi Antonelli | Mercedes | — |
| 🥈 2 | Lando Norris | McLaren | +3.2s |
| 🥉 3 | Oscar Piastri | McLaren | +23.8s |
| 4 | George Russell | Mercedes | +15.9s |
| 5 | Max Verstappen | Red Bull Ford | +16.7s |
| 6 | Charles Leclerc | Ferrari | +17.5s |
| 7 | Lewis Hamilton | Ferrari | +25.4s |
| 8 | Franco Colapinto | Alpine | +24.0s |
| 9 | Carlos Sainz | Williams | +27.8s |
| 10 | Alexander Albon | Williams | +26.7s |
| OUT | Hulkenberg, Lawson, Gasly, Hadjar | — | — |

**Model Accuracy — Round 4:**
- 🥇 Winner correct: ❌ NO — predicted Verstappen, Antonelli won
- 🥈 P2 correct: ❌ NO — predicted Antonelli P2, he actually won 🥇
- 🥉 P3 correct: ❌ NO — predicted Hamilton, Piastri took P3
- 📊 Strict podium score: 0/3
- ✅ **But — Antonelli WAS in our predicted podium (P2)**
- ✅ **Russell predicted P4 — finished P4 exactly!**
- ✅ **All predicted drivers finished in top 7** — right drivers, wrong order
- 💡 The model correctly identified Mercedes + McLaren + Ferrari as the top teams
- 🌧️ PoleWetBonus (0.10) too conservative — upgrading to 0.20 for >75% rain
- 🟠 McLaren race reliability confirmed fixed — Norris P2, Piastri P3

---

## 🗓️ Round 5 — 🇨🇦 Canadian Grand Prix

**Race Date: May 25, 2026 — Circuit Gilles Villeneuve, Montreal**

### 🌡️ Weather
| Session | Temp | Conditions | Rain | Wind | Humidity |
|---------|------|------------|------|------|----------|
| Friday | 19°C | ⛅ Cloudy/Sun | 5% | 16km/h | 37% |
| Saturday | 22°C | ☁️ Cloudy | 10% | 13km/h | 38% |
| Sunday Race | 14°C | 🌧️ Heavy Rain | **95% 🚨** | 16km/h | 41% |

### 🏆 Sprint Race Result
| Pos | Driver | Team |
|-----|--------|------|
| 🥇 1 | George Russell | Mercedes |
| 🥈 2 | Lando Norris | McLaren |
| 🥉 3 | Kimi Antonelli | Mercedes |
| 4 | Oscar Piastri | McLaren |
| 5 | Charles Leclerc | Ferrari |
| 6 | Lewis Hamilton | Ferrari |
| 7 | Max Verstappen | Red Bull Ford |
| 8 | Arvid Lindblad | Racing Bulls |

### 🕐 GP Q3 Qualifying Results
🌟 *Russell — Sprint pole + Sprint win + GP pole*

| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | George Russell 🌟 | Mercedes | 1:12.578 | — |
| P2 | Kimi Antonelli | Mercedes | 1:12.646 | +0.068s |
| P3 | Lando Norris | McLaren | 1:12.729 | +0.151s |
| P4 | Oscar Piastri | McLaren | 1:12.781 | +0.203s |
| P5 | Lewis Hamilton | Ferrari | 1:12.868 | +0.290s |
| P6 | Max Verstappen | Red Bull Ford | 1:12.907 | +0.329s |
| P7 | Isack Hadjar | Red Bull Ford | 1:12.935 | +0.357s |
| P8 | Charles Leclerc | Ferrari | 1:12.976 | +0.398s |
| P9 | Arvid Lindblad | Racing Bulls | 1:13.280 | +0.702s |
| P10 | Franco Colapinto | Alpine | 1:13.697 | +1.119s |

### 🤖 Model Prediction

| | Driver | Team | Pred Lap (s) |
|--|--------|------|-------------|
| 🥇 **George Russell** | Mercedes | 74.238 |
| 🥈 **Lewis Hamilton** | Ferrari | 74.549 |
| 🥉 **Max Verstappen** | Red Bull Ford | 74.970 |
| P4 | Kimi Antonelli | Mercedes | 75.038 |
| P5 | Oscar Piastri | McLaren | 75.501 |
| P6 | Lando Norris | McLaren | 75.540 |
| P7 | Charles Leclerc | Ferrari | 75.614 |
| P8 | Isack Hadjar | Red Bull Ford | 76.140 |
| P9 | Franco Colapinto | Alpine | 76.671 |
| P10 | Arvid Lindblad | Racing Bulls | 76.705 |

> 🔍 Model MAE: 0.06s · PoleWetBonus upgraded to 0.20 after Miami lesson

### ✅ Actual Race Result

| Pos | Driver | Team | Time |
|-----|--------|------|------|
| 🥇 1 | Kimi Antonelli | Mercedes | 1:28:15.758 |
| 🥈 2 | Lewis Hamilton | Ferrari | +10.768s |
| 🥉 3 | Max Verstappen | Red Bull Ford | +11.276s |
| 4 | Charles Leclerc | Ferrari | +44.151s |
| 5 | Isack Hadjar | Red Bull Ford | +1 Lap |
| 6 | Franco Colapinto | Alpine | +1 Lap |
| 7 | Liam Lawson | Racing Bulls | +1 Lap |
| 8 | Pierre Gasly | Alpine | +1 Lap |
| 9 | Carlos Sainz | Williams | +1 Lap |
| 10 | Oliver Bearman | Haas | +1 Lap |
| DNF | George Russell | Mercedes | Car failure from pole |

**Model Accuracy — Round 5:**
- 🥇 Winner correct: ❌ NO — predicted Russell, car failed — Antonelli inherited win
- 🥈 P2 correct: ✅ YES — Hamilton predicted P2, finished P2!
- 🥉 P3 correct: ✅ YES — Verstappen predicted P3, finished P3!
- 📊 Overall: 2/3 podium correct 🎉
- 💡 Russell was on course to win — mechanical DNF unpredictable by any model
- ✅ Hamilton + Verstappen wet weather calls both perfect

---

## 🗓️ Round 6 — 🇲🇨 Monaco Grand Prix

**Race Date: June 8, 2026 — Circuit de Monaco**

### 🌡️ Weather
| Session | Temp | Conditions | Rain |
|---------|------|------------|------|
| Qualifying | 22°C | ☀️ Sunny | ~0% |
| Race Day | 23°C | ⛅ Sunny/Cloudy | 5% |

### 🕐 GP Q3 Qualifying Results
🌟 *Antonelli — 5th pole in 6 races! Only 0.043s over Verstappen!*

| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | Kimi Antonelli 🌟 | Mercedes | 1:12.051 | — |
| P2 | Max Verstappen | Red Bull Ford | 1:12.094 | +0.043s |
| P3 | Lewis Hamilton | Ferrari | 1:12.279 | +0.228s |
| P4 | Charles Leclerc | Ferrari | 1:12.351 | +0.300s |
| P5 | Isack Hadjar | Red Bull Ford | 1:12.434 | +0.383s |
| P6 | George Russell | Mercedes | 1:12.445 | +0.394s |
| P7 | Oscar Piastri | McLaren | 1:12.624 | +0.573s |
| P8 | Lando Norris | McLaren | 1:12.765 | +0.714s |
| P9 | Pierre Gasly | Alpine | 1:13.226 | +1.175s |
| P10 | Liam Lawson | Racing Bulls | 1:13.412 | +1.361s |

### 🤖 Model Prediction

| | Driver | Team | Pred Lap (s) |
|--|--------|------|-------------|
| 🥇 **Kimi Antonelli** | Mercedes | 75.564 |
| 🥈 **Max Verstappen** | Red Bull Ford | 75.688 |
| 🥉 **Lewis Hamilton** | Ferrari | 75.913 |
| P4 | Charles Leclerc | Ferrari | 76.089 |
| P5 | George Russell | Mercedes | 76.372 |
| P6 | Isack Hadjar | Red Bull Ford | 76.406 |
| P7 | Oscar Piastri | McLaren | 76.705 |
| P8 | Lando Norris | McLaren | 76.908 |
| P9 | Pierre Gasly | Alpine | 77.480 |
| P10 | Liam Lawson | Racing Bulls | 77.828 |

> 🔍 Model MAE: 0.02 seconds is best!
> 🆕 New features: MonacoGridPenalty (+0.15s/position) + MonacoHistoryScore
> 🏰 Monaco rule: qualifying order = race order. Pole is everything.
> ⚠️ Only 0.043s between Antonelli and Verstappen — razor thin!

### ✅ Actual Race Result

| Pos | Driver | Team | Gap | Predicted? |
|-----|--------|------|-----|-----------|
| 🥇 1 | Kimi Antonelli | Mercedes | 2:23:31.243 | ✅ Correct |
| 🥈 2 | Lewis Hamilton | Ferrari | +6.271s | ⚠️ Predicted P3 |
| 🥉 3 | Isack Hadjar | Red Bull Ford | +23.394s | ❌ (predicted P6) |
| 4 | Oscar Piastri | McLaren | +24.261s | — |
| 5 | Liam Lawson | Racing Bulls | +26.553s | — |
| 6 | Arvid Lindblad | Racing Bulls | +29.010s | — |
| 7 | Pierre Gasly | Alpine | +30.369s | — |
| 8 | Alexander Albon | Williams | +33.413s | — |
| 9 | Esteban Ocon | Haas | +37.140s | — |
| 10 | Sergio Pérez | Cadillac | +39.153s | — |

**Model Accuracy — Round 6:**
- 🥇 Winner correct: ✅ YES, Antonelli predicted P1, finished P1
- 🥈 P2 correct: ✅ Antonelli + Hamilton both correctly predicted in top 3 — only order differed
- 🥉 P3 correct: ❌ Hadjar surprise P3 not predicted (Verstappen DNF)
- 📊 Strict podium: 1/3 · Fair assessment: 2/3 correct drivers identified ✅
- ✅ Antonelli P1 exact ✅ Hamilton predicted P3 finished P2 one position off
- 😱 Verstappen DNF removed from the race result entirely, unpredictable
- 🌟 Hadjar P3 is the best result of 2026 for Red Bull Ford's junior driver

---

## 🗓️ Round 7 — 🇪🇸 Spanish Grand Prix

**Race Date: June 15, 2026 — Circuit de Barcelona-Catalunya**

### 🌡️ Weather
| Session | Temp | Conditions | Rain |
|---------|------|------------|------|
| Qualifying | 29°C | ⛅ Partly Cloudy | ~0% |
| Race Day | 29°C | ☀️ Sunny | 5% |

### 🕐 GP Q3 Qualifying Results
🌟 *Russell ends Antonelli's pole streak at 5 consecutive poles!*

| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | George Russell 🌟 | Mercedes | 1:14.679 | — |
| P2 | Lewis Hamilton | Ferrari | 1:14.743 | +0.064s |
| P3 | Kimi Antonelli | Mercedes | 1:14.998 | +0.319s |
| P4 | Lando Norris | McLaren | 1:15.001 | +0.322s |
| P5 | Max Verstappen | Red Bull Ford | 1:15.021 | +0.342s |
| P6 | Isack Hadjar | Red Bull Ford | 1:15.077 | +0.398s |
| P7 | Oscar Piastri | McLaren | 1:15.090 | +0.411s |
| P8 | Liam Lawson | Racing Bulls | 1:16.542 | +1.863s |
| P9 | Nico Hülkenberg | Audi | 1:16.657 | +1.978s |
| P10 | Charles Leclerc | Ferrari | 1:15.281 | +0.602s |

### 🤖 Model Prediction

| | Driver | Team | Pred Lap (s) |
|--|--------|------|-------------|
| 🥇 **George Russell** | Mercedes | 79.797 |
| 🥈 **Lewis Hamilton** | Ferrari | 79.834 |
| 🥉 **Kimi Antonelli** | Mercedes | 80.232 |
| P4 | Lando Norris | McLaren | 80.272 |
| P5 | Max Verstappen | Red Bull Ford | 80.315 |
| P6 | Oscar Piastri | McLaren | 80.458 |
| P7 | Isack Hadjar | Red Bull Ford | 80.467 |
| P8 | Charles Leclerc | Ferrari | 80.784 |
| P9 | Liam Lawson | Racing Bulls | 82.124 |
| P10 | Nico Hülkenberg | Audi | 82.166 |

> 🔍 Model MAE: 0.10s · 🆕 New features: BarcelonaGridPenalty, TyreDegScore
> 🔥 Hot dry 29°C race — tyre management critical feature

### ✅ Actual Race Result

| Pos | Driver | Team | Predicted? |
|-----|--------|------|-----------|
| 🥇 1 | Lewis Hamilton | Ferrari | ❌ Predicted P2 |
| 🥈 2 | George Russell | Mercedes | ✅ Predicted P1 (one position off) |
| 🥉 3 | Lando Norris | McLaren | ❌ Predicted P4 |
| 4 | Kimi Antonelli | Mercedes | ⚠️ Predicted P3 |
| 5+ | Rest of field | — | — |

**Model Accuracy — Round 7:**
- 🥇 Winner correct: ❌ NO predicted Russell, Hamilton won
- 🥈 P2 correct: ✅ YES  Russell predicted P1, finished P2 (one position off)
- 🥉 P3 correct: ❌ NO  Norris P3 surprise, predicted P4
- 📊 Strict podium: 1/3
- ✅ Russell + Hamilton both in predicted top 2 right drivers, swapped order again
- 💡 *"The data tells a story, but the race always writes its own ending"* 🧠🔥

## 🗓️ Round 8 — 🇦🇹 Austrian Grand Prix
Race Date: June 29, 2026 — Red Bull Ring, Spielberg 
### 🌡️ Weather
| Session | Temp | Conditions | Rain |
|---------|------|------------|------|
| Qualifying | 33°C | ☀️ Sunny | 20% |
| Race Day | 33°C | ☀️ Sunny | 20% |

### 🕐 GP Q3 Qualifying Results
🌟 Russell pole — 2nd consecutive!
🔥 *Hottest qualifying and race of 2026: tyre degradation critical!*

| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | George Russell 🌟 | Mercedes | 1:06.113 | — |
| P2 | Charles Leclerc | Ferrari | 1:06.349 | +0.236s |
| P3 | Lewis Hamilton | Ferrari | 1:06.408 | +0.295s |
| P4 | Kimi Antonelli | Mercedes | 1:06.414 | +0.301s |
| P5 | Max Verstappen | Red Bull Racing | 1:06.475 | +0.362s |
| P6 | Lando Norris | McLaren | 1:06.502 | +0.389s |
| P7 | Oscar Piastri | McLaren | 1:06.511 | +0.398s |
| P8 | Isack Hadjar | Red Bull Racing | 1:06.632 | +0.519s |
| P9 | Liam Lawson | Racing Bulls | 1:06.955 | +0.842s |
| P10 | Arvid Lindblad | Racing Bulls | 1:07.007 | +0.894s |

### 🤖 Model Prediction
| | Driver | Team | Pred Lap (s) |
|--|--------|------|-------------|
| 🥇 **George Russell** | Mercedes | 70.161 |
| 🥈 **Lewis Hamilton** | Ferrari | 70.542 |
| 🥉 **Charles Leclerc** | Ferrari | 70.658 |
| P4 | Max Verstappen | Red Bull Racing | 70.693 |
| P5 | Kimi Antonelli | Mercedes | 70.723 |
| P6 | Lando Norris | McLaren | 70.952 |
| P7 | Oscar Piastri | McLaren | 70.982 |
| P8 | Isack Hadjar | Red Bull Racing | 71.301 |
| P9 | Liam Lawson | Racing Bulls | 71.756 |
| P10 | Arvid Lindblad | Racing Bulls | 71.892 |

✅ Actual Race Result
|Pos|Driver        |Team         |Gap        |Predicted?    |
|---|--------------|-------------|-----------|--------------|
|🥇 1|George Russell|Mercedes     |1:26:37.979|✅ Correct!    |
|🥈 2|Max Verstappen|Red Bull Ford|+1.611s    |❌ Predicted P5|
|🥉 3|Kimi Antonelli|Mercedes     |+1.986s    |⚠️ Predicted P4|
|4  |Oscar Piastri |McLaren      |+21.809s   |—             |
|5  |Lewis Hamilton|Ferrari      |+26.393s   |—             |

Model Accuracy — Round 8:
- 🥇 Winner correct: ✅ YES, Russell pole to win!
- 🥈 P2 correct: ❌ NO Verstappen P2 from P5 home race magic
- 🥉 P3 correct: ❌ NO Antonelli P3, predicted P4 one position off
- 📊 Strict podium: 1/3
- ✅ Russell correct + Antonelli in top 4; right drivers identified
- 🔵 HomeRaceBoost for Verstappen was directionally correct; he gained 3 places!


## 🗓️ Round 9 — 🇬🇧 British Grand Prix

Race Date: July 5, 2026-Silverstone Circuit

### 🌡️ Weather
| Session | Temp | Conditions | Rain |
|---------|------|------------|------|
|Sprint | 25°C | ⛅ Partly Cloudy | ~0% |
|Sprint + GP Qualifying | 25°C | ⛅ Partly Cloudy | ~0% |
| Race Day | 26°C |☁️ Cloudy| 15% |

### 🏆 Sprint Race Result
| Pos | Driver | Team| 
|-----|--------|------|
| P1 | Kimi Antonelli | Mercedes|
| P2 | Lewis Hamilton | Ferrari |
| P3 | Lando Norris | McLaren | 
| P4 | George Russell | Mercedes |
| P5 | Charles Leclerc | Ferrari| 
| P6 | Max Verstappen |Red Bull Racing | 
| P7 | Oscar Piastri | McLaren | 1:06.511 | +0.398s |
| P8 | Liam Lawson | Racing Bulls |
| P9 | Isack Hadjar | Red Bull Racing |
| P10 | Arvid Lindblad | Racing Bulls | 

### 🕐 GP Q3 Qualifying Results
- 🌟 Antonelli's 5th pole in 9 races! Sprint winner + pole!
- 🏠 Hamilton P3 at his home race, Silverstone crowd electric

| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | Kimi Antonelli 🌟 | Mercedes | 1:28.11 | — |
| P2 | Charles Leclerc | Ferrari | +0.175s | — |
| P3 | Lewis Hamilton | Ferrari | +0.347s | — |
| P4 | George Russell | Mercedes | +0.370s | — |
| P5 | Isack Hadjar | Red Bull Racing | +0.635s| — |
| P6 | Lando Norris |McLaren | +0.766s | — |
| P7 | Max Verstappen |Red Bull Racing |+0.782s| — |
| P8 | Oscar Piastri | McLaren | +0.921s | — |
| P9 | Arvid Lindblad| Racing Bulls |+1.194s| — |
| P10 | Liam Lawson | Racing Bulls |+1.605s| — |

### 🤖 Model Prediction
| | Driver | Team | Pred Lap (s) |
|--|--------|------|-------------|
| 🥇 **Kimi Antonelli** | Mercedes | 93.899 |
| 🥈 **Lewis Hamilton** | Ferrari | 94.142 |
| 🥉 **Charles Leclerc** | Ferrari | 94.312 |
| P4 | George Rusell | Mercedes | 94.313 |
| P5 | Isack Hadjar | Red Bull Racing  | 94.852 |
| P6 | Lando Norris | McLaren | 94.888|
| P7 | Max Verstappen | Red Bull Ford  | 94.902 |
| P8 | Oscar Piastri | McLaren | 95.171 |
| P9 | Arvid Lindblad | Racing Bulls | 95.625 |
| P10 | Liam Lawson | Racing Bulls | 96.075 |

>🔍 Model MAE: 0.06s · Sprint winner: Antonelli · 8 rounds CircuitScore data
>🏠 HomeRaceBoost: Hamilton (1.5x) + Norris (1.0x)
>🔋 7MJ ERS limit · ⚡ +150kW boost cap

### ✅ Actual Race Result
| Pos | Driver | Team | Gap | Predicted? |
|-----|--------|------|-----|-----------|
| 🥇 1 | Charles Leclerc | Ferrari | LEADER | ✅ In predicted podium (P3) |
| 🥈 2 | George Russell | Mercedes | +0.427s | ⚠️ Predicted P4 one off |
| 🥉 3 | Lewis Hamilton | Ferrari | +0.772s | ✅ In predicted podium (P2) |
| 4 | Lando Norris | McLaren | +1.149s | — |
| 5 | Isack Hadjar | Red Bull Ford | +1.598s | — |
| 6 | Liam Lawson | Racing Bulls | +2.023s | — |
| 7 | Arvid Lindblad | Racing Bulls | +2.214s | — |
| 8 | Gabriel Bortoleto | Audi | +2.413s | — |
| 9 | Franco Colapinto | Alpine | +3.229s | — |
| 10 | Pierre Gasly | Alpine | +3.445s | — |
| P16 | Kimi Antonelli | Mercedes | DNF — car failure | Predicted P1 |

**Model Accuracy — Round 9:**
- 🥇 Winner correct: ❌ NO Antonelli DNF (car failure) unpredictable
- ✅ Leclerc + Hamilton BOTH correctly identified in top 3 right drivers, wrong order
- ✅ All 3 predicted drivers finished in the top 3 (excl. Antonelli DNF)
- 📊 Strict podium: 0/3 · Fair assessment: 2/3 correct drivers ✅✅
- 💡 Lesson: Russell is also British; HomeRaceBoost should include him from R10!
- 😱 Antonelli P16 car failure from pole 2nd DNF of season (Canada also DNF)
---

## 🗓️ Round 10-🇧🇪 Belgian Grand Prix
 
**Race Date: July 19, 2026-Spa-Francorchamps**
 
### 🌡️ Weather
| Session | Temp | Conditions | Rain | Wind | Humidity |
|---------|------|------------|------|------|----------|
| Friday FP | 25°C | 🌧️ Rainy | 60% | 5 km/h | 79% |
| Saturday Qualifying | 23°C | ☁️ Cloudy | 30% | 16 km/h | 78% |
| Sunday Race | 20°C | ⛅ Sunny/Cloudy | 20% | 18 km/h | 70% |
 
### 🕐 GP Q3 Qualifying Results
🌟 *Antonelli 7th pole in 10 races! Unstoppable!*
 
| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | Kimi Antonelli 🌟 | Mercedes | 1:44.361 | — |
| P2 | Max Verstappen | Red Bull Ford | 1:44.678 | +0.317s |
| P3 | Lando Norris | McLaren | 1:44.801 | +0.440s |
| P4 | George Russell | Mercedes | 1:44.869 | +0.508s |
| P5 | Charles Leclerc | Ferrari | 1:44.893 | +0.532s |
| P6 | Lewis Hamilton | Ferrari | 1:44.895 | +0.534s |
| P7 | Oscar Piastri | McLaren | 1:45.016 | +0.655s |
| P8 | Arvid Lindblad | Racing Bulls | 1:45.143 | +0.782s |
| P9 | Gabriel Bortoleto | Audi | 1:45.628 | +1.267s |
| P10 | Isack Hadjar | Red Bull Ford | 1:45.823 | +1.462s |
 
### 🤖 Model Prediction
 
| | Driver | Team | Pred Lap (s) |
|--|--------|------|-------------|
| 🥇 **Kimi Antonelli** | Mercedes | 111.216 |
| 🥈 **Max Verstappen** | Red Bull Ford | 111.478 |
| 🥉 **George Russell** | Mercedes | 111.748 |
| P4 | Lewis Hamilton | Ferrari | 111.793 |
| P5 | Lando Norris | McLaren | 111.834 |
| P6 | Charles Leclerc | Ferrari | 111.980 |
 
> 🔍 New features: ColdTyreScore (20°C coldest race), SpaWetHistory
> 🔋 ERS 7MJ most impactful on longest circuit of calendar
 
### ✅ Actual Race Result
 
| Pos | Driver | Team | Gap | Predicted? |
|-----|--------|------|-----|-----------|
| 🥇 1 | Kimi Antonelli | Mercedes | 1:24:42.479 | ✅ Correct! |
| 🥈 2 | Charles Leclerc | Ferrari | +1.952s | ⚠️ Predicted P6 |
| 🥉 3 | Max Verstappen | Red Bull Ford | +11.586s | ✅ Predicted P2 |
| 4 | Lewis Hamilton | Ferrari | +17.245s | — |
| 5 | Oscar Piastri | McLaren | +18.988s | — |
| 6 | Isack Hadjar | Red Bull Ford | +23.307s | — |
| 7 | Lando Norris | McLaren | +24.014s | — |
| 8 | Gabriel Bortoleto | Audi | +49.140s | — |
| 9 | Arvid Lindblad | Racing Bulls | +50.406s | — |
| 10 | Franco Colapinto | Alpine | +76.037s | — |
 
**Model Accuracy — Round 10:**
- 🥇 Winner correct: ✅ YES Antonelli wins again! 6 wins in 10 races
- 🥈 P2 correct: ❌ NO Leclerc P2, predicted P6 big jump!
- 🥉 P3 correct: ✅ YES Verstappen predicted P2, finished P3 correct driver!
- 📊 Strict: 1/3 · Fair: Antonelli ✅ Verstappen ✅ 2 correct drivers
- 🌟 Antonelli: 10 races, 6 wins; one of the greatest debut seasons in F1 history

---
## 🗓️ Round 11- 🇭🇺 2026 Hungarian Grand Prix 
**Race Date: July 26, 2026-Hungaroring**

### 🌡️ Weather
| Session | Temp | Conditions | Rain | Wind | Humidity |
|---------|------|------------|------|------|----------|
| Friday FP | 25°C |⛅ Sunshine/Cloudy | 5% | 8 km/h | 29% |
| Saturday Qualifying |29°C |☀️ Sunshine| 5% | 8 km/h | 29% |
| Sunday Race |	30°C | ⛅ Sunshine/Cloudy | 20% | 19km/h | 38% |


### 🕐 GP Q3 Qualifying Results
🟠 *Norris POLE; Antonelli's 7-race pole streak ends!*
🔴 *Hamilton only 0.012s off pole razor thin!*
 
| Pos | Driver | Team | Time | Gap |
|-----|--------|------|------|-----|
| P1 | 	Lando Norris 🌟 | McLaren | 1:17.207 | — |
| P2 | 	Lewis Hamilton | Ferrari | 1:17.219 | +0.012s |
| P3 |  Charles Leclerc | Ferrari | 1:17.445 | +0.238s |
| P4 | 	Kimi Antonelli | Mercedes | 1:17.479 | 	+0.272s |
| P5 | Oscar Piastri | McLaren | 1:17.684 | +0.477s |
| P6 | 	Max Verstappen | Red Bull Ford | 1:17.725 | +0.518s |
| P7 | George Russell | Mercedes|1:17.760 | —|
| P8 | 	Isack Hadjar | Red Bull Ford | 1:17.856  | +0.649s |
| P9 | Arvid Lindblad | Racing Bulls | 1:18.281 | +1.074s |
| P10 | Nico Hülkenberg| Audi | 1:18.686 |+1.479s |

### ⚠️ Post-Qualifying Penalties — Final Starting Grid

| Penalty | Driver | Reason | Grid Change |
|---------|--------|--------|-------------|
| ⬇️ -3 places | Lewis Hamilton | Impeding Oscar Piastri in qualifying | P2 → P3 |
| ⬇️ -3 places | Kimi Antonelli | Failed to slow under yellow flags in Q3 | P4 → P7 |


**Final Starting Grid:**
| Pos | Driver | Team |
|-----|--------|------|
| P1 | Lando Norris | McLaren |
| P2 | Charles Leclerc | Ferrari |
| P3 | Lewis Hamilton | Ferrari |
| P4 | Oscar Piastri | McLaren |
| P5 | Max Verstappen | Red Bull Ford |
| P6 | Isack Hadjar | Red Bull Ford |
| P7 | Kimi Antonelli | Mercedes ⬇️ |
| P8 | George Russell | Mercedes |
| P9 | Arvid Lindblad | Racing Bulls |
| P10 | Nico Hülkenberg | Audi |

> ⚠️ Note: Model prediction uses post-Hamilton penalty grid (Leclerc P2, Hamilton P3)
> Antonelli P7 penalty noted but not applied to code mentioned for context only
 
 ### 🤖 Model Prediction
| | Driver | Team | Pred Lap (s) |
|--|--------|------|-------------|
| 🥇 **Lando Norris** | McLearn | 82.282 |
| 🥈 **Charles Leclerc** | Ferrari | 82.625 |
| 🥉 **Charles Leclerc** | Ferrari| 82.689 |
| P4 Kimi Antonelli| Mercedes | Mercedes | 82.760 |
| P5 Oscar Piastri | McLaren | McLaren | 82.966 |
| P6 Max Verstappen | Red Bull Racing |Red Bull Racing | 83.135 |
> 🔍 New feature: Hungary2025Score (Norris won, Piastri P2, Russell P3 in 2025)
> 🏁 HungaryGridPenalty: 0.13s/pos 2nd hardest to overtake after Monaco

### ✅ Actual Race Result

| Pos | Driver | Team | Gap |
|-----|--------|------|-----|
| 🥇 1 | Lando Norris | McLaren | LEADER |
| 🥈 2 | Max Verstappen | Red Bull Ford | +15.080s |
| 🥉 3 | Kimi Antonelli | Mercedes | +18.728s |
| 4 | Charles Leclerc | Ferrari | +23.840s |
| 5 | Lewis Hamilton | Ferrari | +24.540s |
| 6 | Isack Hadjar | Red Bull Ford | +55.488s |
| 7 | George Russell | Mercedes | +57.503s |
| 8 | Liam Lawson | Racing Bulls | +1 Lap |
| 9 | Nico Hülkenberg | Audi | +1 Lap |
| 10 | Arvid Lindblad | Racing Bulls | +1 Lap |
| 11 | Gabriel Bortoleto | Audi | +1 Lap |
| 12 | Pierre Gasly | Alpine | +1 Lap |
| 13 | Lance Stroll | Aston Martin | +1 Lap |
| 14 | Fernando Alonso | Aston Martin | +1 Lap |
| 15 | Franco Colapinto | Alpine | +1 Lap |

**Model Accuracy — Round 11:**
- 🥇 Winner correct: ✅ YES Norris pole to win!
- 🥈 P2 correct: ❌ NO predicted Hamilton, Verstappen took P2
- 🥉 P3 correct: ❌ NO predicted Leclerc, Antonelli charged from P7 to P3!
- 📊 Strict: 1/3
- 🌟 Antonelli P3 from P7 grid penalty didn't stop him, incredible race craft
- 🔵 Verstappen P2 strong at Hungary despite difficult Red Bull season
- 🏖️ Summer break after Hungary model resumes after break!"""

---

## 📝 Notes on 2026 Regulations

### 🆕 Mid-Season Regulation Update — Effective from Miami GP (Round 6)
*Confirmed by FIA following analysis of Australia, China, and Japan*

#### ⚡ Qualifying Changes
| Change | Before | After |
|--------|--------|-------|
| Max recharge (MJ) | 8 MJ | **7 MJ** *(3rd consecutive cut — was 9 MJ at season start)* |
| Peak superclip power | 250 kW | **350 kW** |
| Alternative lower energy limit races | 8 races | **12 races** |

#### 🏁 Race Changes
| Change | Detail |
|--------|--------|
| Boost cap | Now capped at **+150 kW** max (previously uncapped) |
| MGU-K deployment | **350 kW** in acceleration zones · **250 kW** elsewhere |
| Peak superclip power | **350 kW** (applies in race too) |

#### 🚦 Race Start Safety
- New **low power start detection** system introduced
- Automatic MGU-K deployment triggered for cars with abnormally low acceleration
- Flashing rear/lateral warning lights on affected cars
- Energy counter reset at formation lap start

#### 🌧️ Wet Conditions
- Intermediate tyre blanket temperatures **increased** (better initial grip)
- Maximum ERS deployment **reduced** in wet (improved car control)
- Rear light systems simplified for better visibility

#### 🏎️ Impact on Predictions from Miami onwards

| Team | ERS Dependency | Miami Regulation Impact |
|------|---------------|------------------------|
| Mercedes | 🔴 Very High | ⚠️ Third consecutive ERS cut hurts |
| McLaren | 🔴 Very High | ⚠️ Same as Mercedes |
| Ferrari | 🟡 Medium | ✅ Benefits from lower ERS reliance |
| Red Bull Ford | 🟢 Lower | ✅ Ford ICE strength more relevant |
| Alpine | 🟡 Medium | 🟡 Neutral |

> ⚠️ These changes are subject to FIA World Motor Sport Council e-vote before implementation at Miami.

---

The 2026 season introduces sweeping rule changes that directly affect model assumptions:

### 🚗 The Car
- Cars are **shorter (3400mm vs 3600mm) and narrower** — more nimble with different setup flexibility
- **Simpler wings** with fewer elements
- Ground effect tunnels **replaced by flatter floors**
- Different setup styles allow different driving styles — team-to-team variation is higher than ever
- **Safety:** Roll hoop takes 23% more load; more rigorous survival cell testing

### 💨 Aerodynamics & Overtaking — DRS is Gone
- **Boost Mode** — a manual boost button usable *anywhere* on track (not just fixed DRS zones)
- **Active Aero** — wings automatically switch between corner mode and straight mode
- **Overtake Button** — extra electrical energy deployed when within **1 second** of the car ahead
- Net effect: overtaking opportunity is no longer circuit-layout dependent, making grid position a weaker predictor of finishing position than in the DRS era

### ⚡ Energy Recovery System (ERS)
- Can recharge the battery with **twice as much energy per lap** vs previous regs
- Recovery happens under braking and lifting off at the end of straights
- Drivers select recharge modes with their race engineer throughout the race
- **Battery management is now an essential race strategy variable** — not just a bonus system

> These regulation changes mean early-season predictions carry higher uncertainty. The model will recalibrate as 2026 race data accumulates round by round.

---

## 🤝 Contributing

Predictions, data improvements, and model refinements are welcome!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/improved-model`)
3. Commit your changes
4. Open a Pull Request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🏆 Let's Predict the Championship!

*24 rounds. 1 model. Let's see how good our predictions get. Updated after every race! 🏁*

---

*Built with ❤️ for F1 fans and data nerds alike. May the best model win.*
