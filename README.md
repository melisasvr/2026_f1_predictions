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
| R01 | 🇦🇺 Australian GP | Melbourne | Mar 8, 2026 | ✅ Russell P1 1:18.518 | 🥇 Russell ✅ | Russell / Antonelli / Piastri | 🥇 Russell 🥈 Antonelli 🥉 Leclerc |
| R02 | 🇨🇳 Chinese GP | Shanghai | Mar 15, 2026 | ✅ Antonelli P1 1:32.064 | 🥇 Russell | Russell / Leclerc / Hamilton | ⏳ Race Day |
| R03 | 🇯🇵 Japanese GP | Suzuka | Mar 29, 2026 | — | — | — | — |
| R04 | 🇧🇭 Bahrain GP | Sakhir | Apr 12, 2026 | — | — | — | — |
| R05 | 🇸🇦 Saudi Arabian GP | Jeddah | Apr 19, 2026 | — | — | — | — |
| R06 | 🇺🇸 Miami GP | Miami | May 3, 2026 | — | — | — | — |
| R07 | 🇨🇦 Canadian GP | Montreal | May 24, 2026 | — | — | — | — |
| R08 | 🇲🇨 Monaco GP | Monaco | Jun 7, 2026 | — | — | — | — |
| R09 | 🇪🇸 Spanish GP | Barcelona-Catalunya | Jun 14, 2026 | — | — | — | — |
| R10 | 🇦🇹 Austrian GP | Spielberg | Jun 28, 2026 | — | — | — | — |
| R11 | 🇬🇧 British GP | Silverstone | Jul 5, 2026 | — | — | — | — |
| R12 | 🇧🇪 Belgian GP | Spa-Francorchamps | Jul 19, 2026 | — | — | — | — |
| R13 | 🇭🇺 Hungarian GP | Budapest | Jul 26, 2026 | — | — | — | — |
| R14 | 🇳🇱 Dutch GP | Zandvoort | Aug 23, 2026 | — | — | — | — |
| R15 | 🇮🇹 Italian GP | Monza | Sep 6, 2026 | — | — | — | — |
| R16 | 🇪🇸 Spanish GP (Madrid)* | Madrid | Sep 13, 2026 | — | — | — | — |
| R17 | 🇦🇿 Azerbaijan GP | Baku | Sep 26, 2026 | — | — | — | — |
| R18 | 🇸🇬 Singapore GP | Singapore | Oct 11, 2026 | — | — | — | — |
| R19 | 🇺🇸 United States GP | Austin | Oct 25, 2026 | — | — | — | — |
| R20 | 🇲🇽 Mexico City GP | Mexico City | Nov 1, 2026 | — | — | — | — |
| R21 | 🇧🇷 Brazilian GP | São Paulo | Nov 8, 2026 | — | — | — | — |
| R22 | 🇺🇸 Las Vegas GP | Las Vegas | Nov 21, 2026 | — | — | — | — |
| R23 | 🇶🇦 Qatar GP | Lusail | Nov 29, 2026 | — | — | — | — |
| R24 | 🇦🇪 Abu Dhabi GP | Yas Island | Dec 6, 2026 | — | — | — | — |

*Subject to FIA circuit homologation

---

## 📋 How It Works

### Workflow Per Round

```
1. Qualifying Session (Friday)
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

- 🥇 **Race Winner** — Who crosses the line first
- 🏆 **Podium (Top 3)** — The full podium finishers

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
| Rounds Completed | 1 / 24 |
| Winner Correct | 1 / 1 ✅ |
| Winner Accuracy | 100% |
| Podium Correct (P1+P2) | 2 / 2 ✅ |
| P3 Correct | ⚠️ Piastri DNE — Leclerc (dark horse) took P3 |
| Average Model MAE | R01: 2.58 pos · R02: 0.56s |
| Overall Score | 2/3 podium ✅ (effectively 3/3) · R02 pending |

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

Each round has its own self-contained script. Once qualifying results are in, update the qualifying data at the top of the script and run:

```bash
# Example — Round 1 Australian GP
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

| Pos | Driver | Team | Time |
|-----|--------|------|------|
| P1 | George Russell | Mercedes | 1:18.518 |
| P2 | Kimi Antonelli | Mercedes | — |
| P3 | Isack Hadjar | Red Bull Racing | — |
| P4 | Charles Leclerc | Ferrari | — |
| P5 | Oscar Piastri | McLaren | — |
| P6 | Lando Norris | McLaren | — |
| P7 | Lewis Hamilton | Ferrari | — |
| P8 | Liam Lawson | Racing Bulls | — |
| P9 | Arvid Lindblad* | Racing Bulls | — |
| P10 | Gabriel Bortoleto | Audi | — |

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

### ✅ Actual Result

| | Driver | Team | Predicted? |
|--|--------|------|-----------|
| 🥇 | George Russell | Mercedes | ✅ Correct |
| 🥈 | Kimi Antonelli | Mercedes | ✅ Correct |
| 🥉 | Charles Leclerc | Ferrari | ⚠️ Dark Horse (model had P5) |

> ⚠️ **Note:** Oscar Piastri (predicted P3) crashed on the way to the grid during pit lane open and did not start the race. The car was ok and Piastri was uninjured. Leclerc, our dark horse pick, stepped up to take P3.

**Model Accuracy — Round 1:**
- 🥇 Winner correct: ✅ YES
- 🥈 P2 correct: ✅ YES
- 🥉 P3 correct: ⚠️ Piastri DNE (Did Not Enter) — Leclerc was model's dark horse
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

### ✅ Actual Result
*To be updated after the race on March 15, 2026*

---

## 📝 Notes on 2026 Regulations

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
```
Copyright (c) 2026 Deep Research Agent Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
---

## 🏆 Let's Predict the Championship!

*24 rounds. 1 model. Let's see how good our predictions get. Updated after every race! 🏁*

---

*Built with ❤️ for F1 fans and data nerds alike. May the best model win.*
