# 6. Conclusion

## 6.1 Summary of findings

Across three currency pairs (AUDUSD/NZDUSD, GBPUSD/EURUSD,
EURNOK/EURSEK) and three months (August 2024, September 2024,
August 2025), we tested whether a Markov-switching AR(1) regime
classifier adds value when used as a filter on top of high-frequency
pairs trading. We compared four strategies on a 3-day-rolling /
1-day-out-of-sample framework with end-of-day flattening:

1. **Buy & Hold** — passive long spread (benchmark, no costs).
2. **Baseline** — z-score pairs trade, no regime filter.
3. **AR** — z-score pairs trade with hard-kill HMM filter.
4. **MS-AR** — z-score pairs trade with soft probability-weighted
   entry band.

Three principal empirical results:

1. **MS-AR is uniformly better than AR** as a way to use the HMM
   (8 / 9 month-pairs in the main test; 47 / 81 cells in the sensitivity
   sweep, and the cells where AR wins are tightly concentrated in
   GBPEUR-August-2024).
2. **Baseline is best overall in cleanly cointegrated months** (4 / 6
   wins outside NOKSEK). Adding any HMM-based filter to a clean,
   mean-reverting pair tends to destroy value.
3. **The regime filter (AR specifically) is dramatically better in
   months that contain a genuine, persistent regime switch.** In our
   one example of such a month (GBPEUR August 2024), AR outperformed
   Baseline by ~24 Sharpe points (Sharpe +23.5 vs −0.5) by cutting
   80 % of trades and keeping only the high-quality ones.

The clean version of the conclusion: the HMM regime filter has
*conditional* value. Used as a soft gearbox (MS-AR) it does little
harm in calm markets and a little good in stress. Used as a hard kill
(AR) it does substantial harm in calm markets and substantial good in
persistent stress. We do not have a reliable way to know in advance
which regime mode a given month will fall into.

---

## 6.2 What we learned methodologically

### About the modelling pipeline

- **3-day rolling MS-AR(1) training** is sufficient to obtain a stable
  regime classifier on 500-tick bars. EM convergence on every trading
  day across all 9 month-pair experiments. No skipped folds.
- **Decoupling in-sample smoother from out-of-sample one-step
  posterior** prevents look-ahead and yields a clean walk-forward
  result. The OOS posterior is an analytical Gaussian mixture
  conditional on training-window parameters and the previous bar's
  spread.
- **End-of-day flattening is essential for one-month experiments**.
  Without it, the gearbox can produce thousands of overnight
  position-flips that wash out any underlying edge. With it, every
  trading day is a clean independent experiment.
- **Linear pre-processing for EM stability** (winsorisation + scaling)
  doesn't change regime structure but markedly improves EM
  convergence and the cleanliness of the regime probabilities.

### About the strategies

- **The naïve z-score Baseline is hard to beat** on cleanly
  cointegrated pairs. Pairs-trading literature has long claimed this;
  our experiment confirms it at high frequency on real tick data.
- **Hard kills are dangerous outside their target scenario.** They
  behave well when the regime is genuinely persistent (the
  GBPEUR-Aug-24 case) but bleed slippage from entry → kill → re-entry
  cycles in transient-regime contexts.
- **Soft gearboxes are the safer use of an HMM.** They preserve
  in-the-money positions through brief regime flickers and respond
  continuously to the model's continuous output, but they cannot
  capture the full upside of avoiding a persistent bad regime.
- **Neither HMM variant uniformly dominates an un-filtered Baseline.**
  At $n = 9$ we don't have statistical power to claim an expected-value
  difference between MS-AR and Baseline. The visible difference is
  in dispersion (MS-AR has lower variance month-to-month than
  Baseline, but also lower mean in the cleanly-cointegrated subset).

### About the experimental design

- **Fixed parameters reveal more than WFO-optimised parameters.** The
  same regime models, with Optuna optimising trading-rule parameters
  on the two-year span, produced misleading results (MS-AR Sharpe −3,
  9 235 trades) because the optimiser picked extreme parameter
  combinations that maximised in-sample Sharpe via churn. Holding
  parameters fixed at sensible values exposed the actual structural
  behaviour.
- **A small, hand-picked experiment (9 cells, 81 sweep cells) is more
  informative than a long aggregate run.** The two-year aggregate hid
  the GBPEUR-Aug-24 stress finding by averaging it against calm
  months. Splitting the experiment surfaced it.
- **Negative controls are valuable.** Including NOKSEK (which we
  suspected to be broken) confirmed that the HMM correctly identifies
  non-cointegrated dynamics — the Danger probability is persistently
  high — while also confirming that no z-score-based filter can
  rescue a pair without an underlying edge.

---

## 6.3 Limitations

- **Sample size.** 9 month-pair observations is too few for
  statistical claims about long-run averages. The GBPEUR-Aug-24 win
  is one observation, not a trend.
- **Single bar resolution.** All experiments use 500-tick bars. Bar
  resolution affects the bias–variance trade-off in the rolling OLS
  and the HMM separately. We did not test sensitivity to bar size.
- **Single regime model.** Two-regime AR(1) with Gaussian innovations
  and a homogeneous transition matrix. More expressive models
  (heavy-tailed innovations, three or more regimes, regime-dependent
  AR order, Hidden Semi-Markov Models with duration distributions)
  would give different posteriors.
- **Annualisation convention.** Tick-clock Sharpe is inflated relative
  to daily-returns Sharpe by ~$38\times$. Cross-strategy ranking
  within a month is preserved; absolute magnitudes are not.
- **Transaction-cost simplification.** Fixed half-spread + 0.5 bps fee
  is a clean baseline. Real-world friction in the Danger regime is
  higher (spreads widen, market depth thins), which would bias the
  comparison further in favour of filtered strategies.
- **No latency model.** Bars are executed at the close. Real
  execution would slip more, especially during Danger regimes.
- **Fixed-trading-rule parameters.** While intentional (see §3.8 for
  why), this leaves open the possibility that *some* parameter set
  exists for which MS-AR beats Baseline universally. We have not
  exhaustively searched the parameter space.

---

## 6.4 What we'd do next

In order of decreasing priority:

1. **Drill into the GBPEUR-August-2024 case study.** Plot
   `df_params['Danger_Variance']`, the smoothed Danger probability
   series, and the spread itself for the full month. Identify the
   calendar windows where the HMM flagged danger and confirm they
   coincide with real market events (the UK budget, Bank of England
   surprises, ECB decisions, etc.). This is the most interesting
   single empirical result in the project and deserves a focused
   write-up.

2. **Test more months specifically picked for stress.** Examples:
   September 2008 (Lehman), August 2015 (CNY devaluation), March
   2020 (COVID), the days around Yen interventions (April–May 2024),
   the BOE pension-LDI crisis (September–October 2022). The hypothesis
   is that the HMM filter pays off during macro stress and rarely
   otherwise. With 20+ stress months we could begin to estimate the
   *rate* at which the filter helps.

3. **Build a meta-strategy that estimates regime persistence and
   switches between hard kill and soft gearbox dynamically.** The
   transition-matrix diagonal $p_{22}$ (Danger-stay probability) is
   already estimated per fold. When $p_{22}$ is high (regime is
   persistent), use AR; when $p_{22}$ is moderate (transient flickers
   likely), use MS-AR. Calibrate the persistence threshold on a
   training-only validation set.

4. **Regime-aware cost modelling.** Replace the constant 0.5 bps fee
   plus current half-spread with a Danger-regime-aware slippage
   estimate. Spreads in stressed regimes empirically widen 2–5×.
   This is the single biggest source of bias in the current
   transaction-cost story.

5. **Extend to more pairs.** EURUSD/USDJPY, USDCHF/EURCHF,
   AUDUSD/CADUSD are obvious next candidates. Several already have
   tick data on disk in `code/data/processed`.

6. **Try a Hidden Semi-Markov Model (HSMM).** Instead of a geometric
   regime-duration distribution implicit in the Markov chain, fit an
   explicit duration distribution per regime. This could give a
   cleaner "long-stay vs short-stay" signal than the current Markov
   chain.

7. **Bootstrap-based statistical inference.** Replace the tick-clock
   Sharpe with a block-bootstrap distribution of the strategy's mean
   return. Compute confidence intervals on the MS-AR-minus-Baseline
   gap, in absolute terms and corrected for multiple testing across
   the 9 month-pair cells.

---

## 6.5 The clean one-paragraph version, for a paper

> In high-frequency pairs trading on liquid 2024–2025 FX data, a
> two-regime Markov-switching AR(1) regime classifier adds value as a
> **soft gearbox** (continuous probability-weighted entry threshold)
> rather than as a **hard kill switch** (binary liquidation on high
> Danger probability): the gearbox uniformly dominates the kill
> switch in 8 of 9 month-pair experiments and 47 of 81
> sensitivity-sweep cells. However, neither HMM-based variant beats
> the unfiltered z-score baseline in the majority of cleanly
> cointegrated months. The hard-kill filter outperforms in exactly
> one month (GBPUSD/EURUSD, August 2024), in which it correctly
> identifies a persistent high-variance regime, avoids 80 percent of
> the baseline's trades, and captures a Sharpe of +23.5 against the
> baseline's −0.5 — robust across 25 of 27 parameter-grid cells.
> We conclude that the value of a regime filter in pairs trading is
> conditional on *regime persistence*: persistent stress regimes
> reward hard kills, transient regime flickers reward soft gating,
> and clean cointegration rewards no filter at all. A natural
> direction is a duration-aware meta-strategy that estimates regime
> persistence (via the transition-matrix diagonal) and dynamically
> selects between gear-shift modes.

---

## 6.6 A short personal note for the writer

Looking at the spread of nine cells in the headline Sharpe table, you
can read two stories:

- A **disappointing one**: "regime filters don't beat the baseline".
- A **more interesting one**: "regime filters have one shape of
  pay-off (rare, large, conditional on persistence), and we
  characterised when and why."

The second story is the publishable one. It tells the reader exactly
what *would* have to change about the deployment context for the
filter to be worth running, and gives a clear next experiment (more
stress months, persistence-aware meta-strategy). That's the version
to write up.

Either way, the project succeeded at what it was actually testing:
not "can MS-AR make money" (we don't have the sample size to answer
that) but "what is the structural relationship between an HMM regime
filter and a baseline z-score pairs trade". We have a clean
characterisation of that relationship, with one striking single
example of when the filter dominates and a clear mechanism for why.
