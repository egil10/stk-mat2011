# STK-MAT2011 — Project Work in Finance, Insurance, Risk and Data Analysis

> **University of Oslo** · Bachelor's Level · 10 Credits · Spring 2026

---

## About

This repository contains my project work for **STK-MAT2011**, a hands-on course where students apply mathematics and computational tools to solve real-world problems in finance, insurance, risk management, and data analysis.

The course focuses on:
- Problem analysis and structured problem-solving
- Independent research under supervision
- Scientific writing using **LaTeX**
- Communicating technical results

---

## Project: Machine Learning for High-Frequency Trading

This project explores **high-frequency forex tick data** (EUR/USD) using statistical analysis and machine learning techniques. The dataset contains over 1.5 million tick-level observations from January 2026.

### Key Analyses
- **Price dynamics** — Bid/ask spread behavior and mid-price evolution
- **Microstructure patterns** — Intraday spread variations across trading sessions
- **Volatility modeling** — Realized variance and return distributions
- **Trading activity** — Tick frequency and market liquidity patterns

---

## Repository Structure

```
stk-mat2011/
├── code/
│   ├── data/               # EUR/USD tick data (git-ignored)
│   │   ├── DAT_ASCII_EURUSD_T_202601.csv    # Combined bid/ask (56 MB)
│   │   ├── DAT_NT_EURUSD_T_ASK_202601.csv   # Ask prices (17 MB)
│   │   ├── DAT_NT_EURUSD_T_BID_202601.csv   # Bid prices (17 MB)
│   │   └── *.txt                             # Metadata & gap reports
│   ├── scripts/
│   │   └── visualize_forex.py   # Data visualization pipeline
│   ├── notebooks/          # Jupyter notebooks
│   ├── oblig/              # Mandatory assignments
│   └── plots/
│       └── eurusd_tick_analysis.pdf   # Generated visualizations
├── STK-MAT2011 - ML HFT.pdf   # Project report
├── .gitignore
└── README.md
```

---

## Quick Start

### Requirements
```
python >= 3.9
pandas
numpy
matplotlib
```

### Generate Visualizations
```bash
cd stk-mat2011
python code/scripts/visualize_forex.py
```

This produces a 6-panel PDF with:
1. Price overview (bid/ask/mid with spread shading)
2. Bid-ask spread distribution
3. Intraday spread patterns by hour
4. Log returns distribution vs. normal
5. Trading activity (tick frequency)
6. Daily realized volatility

---

## Data

The forex tick data is sourced from [HistData.com](https://www.histdata.com/) and contains millisecond-resolution EUR/USD quotes. Large CSV files are excluded from version control via `.gitignore`.

| Metric | Value |
|--------|-------|
| Time range | Jan 1 – Jan 30, 2026 |
| Total ticks | ~1.5 million |
| Bid range | 1.15778 – 1.20805 |
| Ask range | 1.15794 – 1.20809 |
| Avg spread | 0.33 pips |

---

## Prerequisites

| Required (at least one)                                      | Recommended                                  |
|--------------------------------------------------------------|----------------------------------------------|
| STK2100 – Machine Learning and Prediction                    | STK1100 – Probability and Statistical Modelling |
| STK2130 – Modelling by Stochastic Processes                  | MAT1100/MAT1110 – Calculus                   |
|                                                              | MAT1120/MAT1125 – Linear Algebra             |
|                                                              | IN1900/IN1910 – Scientific Programming       |

---

## Assessment

| Component              | Weight |
|------------------------|--------|
| Project Paper (LaTeX)  | 100%   |
| Oral Presentation      | Req.   |

*Grading: Pass/Fail*

---

## Resources

- [Course Page (UiO)](https://www.uio.no/studier/emner/matnat/math/STK-MAT2011/)
- [HistData – Free Forex Historical Data](https://www.histdata.com/)
- [LaTeX Documentation](https://www.overleaf.com/learn)

---

<sub>University of Oslo · Department of Mathematics · Spring 2026</sub>