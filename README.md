# Strategy Learner Project  
**Author:** Joshua Lutkemuller  
**GTID:** 904051695  
**GT Username:** `jlutkemuller3`  

---

## 🧠 Overview

This project implements a **machine learning–based trading strategy** (`StrategyLearner`) that automatically learns and executes trades based on technical indicators.  
It extends the manual strategy logic into a **Bagged Random Tree ensemble** that adapts to different market impacts and trading costs.  

The implementation is designed for **CS 7646 – Machine Learning for Trading**, covering:

- Manual and strategy-based trading policy generation  
- Random Tree and Bagged Ensemble learners  
- Portfolio simulation and evaluation  
- Experimentation with market impact and parameter sensitivity  

---

## 📂 Project Structure

```
.
├── StrategyLearner.py        # Main strategy learner class
├── RTLearner.py              # Random Tree learner
├── BagLearner.py             # Bootstrap Aggregation (Bagging) learner
├── indicators.py             # Technical indicators (BB%, RSI, MACD, etc.)
├── experiment1.py            # Manual strategy comparison (Experiment 1)
├── experiment2.py            # Market impact sensitivity (Experiment 2)
├── testproject.py            # Entry point / test runner
├── metadata.yml              # Project metadata (grading)
└── util.py                   # Helper functions (provided by course)
```

---

## ⚙️ Requirements

**Language:** Python 3.8+  
**Dependencies:**
```bash
pip install numpy pandas matplotlib
```

The provided `util.py` (from course materials) must be in the same directory, as it supplies:
- `get_data()` – for fetching price data  
- `plot_data()` – for visualization  

---

## 🧩 Key Components

### `StrategyLearner.py`
Implements the main **learning agent**, which:
1. Computes technical indicators (Bollinger %B, RSI, MACD, Golden Cross)
2. Trains a **bagged ensemble** of random trees (`BagLearner` + `RTLearner`)
3. Selects optimal hyperparameters (`horizon`, `threshold`, `leaf_size`) using validation CR
4. Outputs daily trade signals (`testPolicy`) for simulation

### `RTLearner.py`
Implements a **Random Tree regression learner** that splits randomly selected features at their median until leaf conditions are met.

### `BagLearner.py`
Implements **bootstrap aggregation** (bagging), training multiple `RTLearner` instances and averaging predictions to reduce variance.

### `indicators.py`
Contains all major **technical indicators** used for strategy signal generation:
- Bollinger %B  
- Golden Cross (50/200 SMA difference)  
- RSI (Wilder’s method)  
- MACD (12/26 EMA histogram)  
- Keltner Channels (optional visualization)

### `experiment1.py`
Compares **Manual Strategy vs. Strategy Learner** performance both in-sample and out-of-sample, producing:
- Portfolio value plots
- Entry/exit markers
- Performance metric tables (CR, ADR, SDDR)

### `experiment2.py`
Runs a **Market Impact Sensitivity Test**, analyzing the learner’s performance across different `impact` values and plotting:
- Cumulative Return
- Average Daily Return
- Annualized Sharpe Ratio
- Annualized Std. Dev.  
Results are visualized as both line plots and summary tables.

---

## 🚀 How to Run

### 1. Manual Strategy
```bash
python experiment1.py
```
Generates:
- In-sample and out-of-sample performance plots
- Metric comparison tables between Benchmark vs. Manual Strategy

### 2. Strategy Learner
```bash
python testproject.py
```
This script:
- Instantiates the `StrategyLearner`
- Runs training (`add_evidence`) and testing (`testPolicy`)
- Plots results versus the benchmark

### 3. Market Impact Analysis
```bash
python experiment2.py
```
Runs impact sensitivity across `[0.0, 0.0025, 0.005, 0.01]` and saves:
- `experiment2_impact_sensitivity_insample.png`
- `experiment2_results_table.png`

---

## 📊 Outputs

All plots and tables are saved automatically in the working directory:
- `Manual_vs_Benchmark_(In Sample).png`
- `Manual_vs_Benchmark_(Out Sample).png`
- `experiment2_impact_sensitivity_insample.png`
- `experiment2_results_table.png`

---

## 🧪 Experiments Summary

| Experiment | Objective | Description |
|-------------|------------|--------------|
| **1: Manual vs. Strategy** | Benchmark comparison | Evaluates the performance of manual rules vs. ML-based trades |
| **2: Market Impact Sensitivity** | Robustness test | Examines the effect of transaction cost and impact on performance metrics |

---

## 🧑‍💻 Implementation Highlights

- **Adaptive learning** via validation-based hyperparameter tuning  
- **Bagged ensemble** to smooth volatility and reduce overfitting  
- **Dynamic position sizing** (±1000 shares) with flattening at end date  
- **Automated experiment pipeline** for reproducibility  

---

## 🧾 License

This project is based on the **Georgia Tech CS 7646** template,  
© 2018 Georgia Institute of Technology. Redistribution or public posting is prohibited.  
