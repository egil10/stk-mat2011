# Supervisor feedback (meeting summary)

Short summary of comments after the latest meeting, plus a concrete project roadmap aligned with this repo (`hw0`–`hw2`, `nnhmm.py`, AR(1)-HMM / ARCH extensions).

---

## 1. Prediction intervals (all methods)

- Build **prediction intervals** not only point forecasts, for **every** model you compare (single AR, HMM variants, mixture predictor, etc.).
- The **mixture** predictor needs a bit more work: use standard **double expectation / law of total variance** (totale forventning og varians) to get a mean and a variance for the predictive, then form an interval (often Gaussian; state clearly if it is an approximation).

---

## 2. Interval score (primary scalar metric)

- Look up **interval score** as a **strictly proper** way to combine **calibration** (coverage) and **interval width** into **one number**.
- Background reading tends to be dense; a standard reference is **Gneiting & Raftery**, *Strictly Proper Scoring Rules, Prediction, and Estimation*.
- Other scoring rules exist with similar goals; **for now, standardise on interval score** so comparisons stay simple.

---

## 3. General forecasting reference

- Consider skimming **Forecasting: Principles and Practice** (3rd ed.): [https://otexts.com/fpp3/](https://otexts.com/fpp3/) for interval construction, evaluation mindset, and reporting.

---

## 4. Simulation design: asymmetric stickiness + interpretable regimes

- Experiment with **asymmetric stickiness** in the Markov chain (one “normal” state that persists, rare switches to short-lived alternatives).
- Substantive idea: **normal** regime ≈ **low autocorrelation** in returns; **alternatives** ≈ **shorter spells** with **stronger** autocorrelation (**positive or negative**). This matches the economic story (noise vs momentum / microstructure) better than symmetric two-state toys.

---

## 5. Report and structure

- Once intervals + interval score + key empirical results exist, **feed them into the thesis** and revisit **overall chapter structure** so methods → experiments → results read as one line of argument.

---

## 6. Real data: MS-AR(1) on returns (EUR/USD)

- **Fit a Markov-switching AR(1)** (2–3 hidden states) on **real returns** (after your preprocessing).
- Vary **pre-averaging** strength (window / stride) and compare behaviour.
- Two **day-handling** setups (both worth reporting):
  1. **Clip** roughly the **first and last 2–3 hours** of each trading day (reduce open/close noise), then treat days as **separate segments**.
  2. **Concatenate** days into **one long series** (no clip) and fit on that.
- **Likelihood when days are clipped / separate:** total log-likelihood = **sum of log-likelihoods per day**, i.e. **conditional independence across days**. Supervisor: this is **acceptable** for your purpose.

---

## 7. Extension: ARCH / GARCH

- Extend toward **ARCH** and/or **GARCH** (start simple).
- Initial ARCH mean equation (example):  
  \(y_t = \beta_0 + \beta_1 y_{t-1} + \varepsilon_t\),  
  \(\varepsilon_t = \sigma_t z_t\), \(z_t \sim \mathcal{N}(0,1)\),  
  \(\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2\).  
  You may set **\(\beta_0 = 0\)** at first to keep parameters manageable.
- **Later:** allow **\(\beta_0\)** (intercept) to **depend on the hidden state** as well.

---

## 8. Trading strategy (later)

- After models and evaluation are in place, sketch a **trading strategy** (supervisor will refine). **Pairs trading** is mentioned as a natural candidate.

---

## Suggested next steps (ordered)

Priority is: **frozen data → MS-AR(1) on real EUR/USD → intervals + interval score → simulations with asymmetric stickiness → ARCH extension → report → trading idea.**

| Step | Task | Notes |
|------|------|--------|
| A | **Master series (`hw1`)** from Drive `processed` bid/ask → mid → returns; one clear pre-avg v1 | Unblocks all empirical work. |
| B | **Two day protocols** on the same pre-avg grid: (1) clip 2–3 h open/close per day, sum LL over days; (2) concatenate full days | Directly matches supervisor’s comparison. |
| C | **MS-AR(1), K ∈ {2,3}** on returns | Use proper MS-AR likelihood / EM or vetted package; avoid only a 2D Gaussian HMM on \((y_t,y_{t-1})\) as the final word unless you justify it as a proxy. |
| D | **Prediction intervals** for single AR, hard-switch HMM, mixture | Mixture: law of total variance; document Gaussian approximation if used. |
| E | **Interval score** on test windows | Same splits for all models; one table in thesis. |
| F | **Simulations** with **asymmetric P** and **low-ϕ vs high-\|ϕ\|** short spells | Links narrative to “normal vs burst” autocorrelation. |
| G | **ARCH(-MS)** layer | Start with \(\beta_0=0\); then state-dependent \(\beta_0\) if needed. |
| H | **Thesis** | Drop in results; tighten outline (data → model → intervals/scores → empirical → extensions). |
| I | **Trading / pairs** | After (C)–(E) are stable. |

---

## Repo pointers

- Raw / processed ticks: `hw0` notebook, Drive `.../data/processed`, `code/scripts/p_duka.py`.
- Neural HMM likelihood / decoding: `code/scripts/nnhmm.py`, `hw2_nnhmm_calibration.ipynb`.
- Classical HMM on features: `code/scripts/hmm.py`; MS-AR theory/status: `code/docs/HMMAR1.md`.
- Baseline scripts: `code/docs/MINIMAL_TS_MODELS.md` (ARIMAX/ARCH/GARCH demos).

This file is a **living checklist**: tick off rows as you implement and cite the supervisor’s wording where it matters in the thesis (day-independence likelihood, interval score, asymmetric stickiness).
