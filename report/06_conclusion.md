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

Four principal empirical results:

1. **MS-AR is uniformly better than AR** as a way to use the HMM
   (8 / 9 month-pairs in the main test; 47 / 81 cells in the sensitivity
   sweep, and the cells where AR wins are tightly concentrated in
   GBPEUR-August-2024).
2. **Baseline is best overall in cleanly cointegrated months** (4 / 6
   wins outside NOKSEK). Adding any HMM-based filter to a clean,
   mean-reverting pair tends to destroy value.
3. **The regime filter (AR specifically) outperforms Baseline in a
   regime-switching month**, but not in absolute terms. In GBPEUR
   August 2024, AR cut 80 % of trades and beat Baseline by 50 bps
   (Sharpe gap +24), **but Buy & Hold beat AR by 172 bps in the same
   month** — the spread drifted up 2 % and the filter's value was
   "trade less, lose less to the drift", not "capture an edge".
4. **No traded strategy beat Buy & Hold in absolute PnL in any of
   the 9 month-pair cells.** Baseline beat BuyHold by a large margin
   in cleanly-cointegrated months (e.g. GBPEUR-Sep24 +322 vs +151
   bps), but in those months the strategy was running with the drift,
   not against it.

The clean version of the conclusion: the HMM regime filter has
*conditional, defensive* value. As a soft gearbox (MS-AR) it does
little harm in calm markets and a little good when the spread
behaves badly. As a hard kill (AR) it does substantial harm in calm
markets and substantial good (relative to Baseline) when the spread
drifts persistently. **Neither variant turns a losing pairs trade
into a winning one** — the underlying z-score rule is sign-blind to
drift direction, and the HMM cannot fix that. The filter is capital
preservation; the alpha (or lack thereof) is in the underlying
cointegration premise.

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

1. **Add a drift/stationarity gate upstream of the trading rule.**
   The BuyHold analysis (§4.6, §5.2) shows that every "profitable"
   month was a drift month — the strategies that won did so partly by
   *running with the drift*, and the strategies that lost did so by
   *fighting it*. A test for spread stationarity over the training
   window (e.g. ADF / KPSS / a slope test on the rolling spread) could
   gate the trading rule entirely: if the spread is non-stationary,
   either abstain or switch to a directional rule. This is the
   intervention most likely to materially improve real-money
   performance.

2. **Drill into the GBPEUR-August-2024 case study.** Plot
   `df_params['Danger_Variance']`, the smoothed Danger probability
   series, the spread itself, and BuyHold's running PnL for the full
   month. Identify the calendar windows where the HMM flagged danger
   and confirm they coincide with real market events (UK budget, BoE
   surprises, ECB decisions). Quantify how much of the +215.6 bps
   drift was captured / lost by each strategy.

3. **Test more months specifically picked for stress AND for clean
   stationarity.** Stress examples: September 2008 (Lehman), August
   2015 (CNY devaluation), March 2020 (COVID), April–May 2024 (Yen
   intervention), September–October 2022 (BoE LDI crisis). Stationary
   examples: long, quiet, low-vol windows in EURUSD or other liquid
   majors. The point is to characterise the filter's performance as a
   function of stationarity *and* persistence, not just persistence.

4. **Build a meta-strategy that selects gear-shift mode based on
   regime persistence AND drift sign.** Persistence comes from the
   transition-matrix diagonal $p_{22}$. Drift sign comes from a slope
   test on the spread's rolling regression. The meta-rule:
   stationary calm → Baseline; transient stress → MS-AR; persistent
   stress with drift → AR; non-stationary drift → trend-follow or
   abstain.

5. **Regime-aware cost modelling.** Replace the constant 0.5 bps fee
   plus current half-spread with a Danger-regime-aware slippage
   estimate. Spreads in stressed regimes empirically widen 2–5×.
   This is the single biggest source of bias in the current
   transaction-cost story and would penalise Baseline more (it's the
   one trading through the danger).

6. **Extend to more pairs.** EURUSD/USDJPY, USDCHF/EURCHF,
   AUDUSD/CADUSD are obvious next candidates. Several already have
   tick data on disk in `code/data/processed`.

7. **Try a Hidden Semi-Markov Model (HSMM).** Instead of a geometric
   regime-duration distribution implicit in the Markov chain, fit an
   explicit duration distribution per regime. This could give a
   cleaner "long-stay vs short-stay" signal than the current Markov
   chain — directly informative for the meta-strategy in (4).

8. **Bootstrap-based statistical inference.** Replace the tick-clock
   Sharpe with a block-bootstrap distribution of the strategy's mean
   return. Compute confidence intervals on the MS-AR-minus-Baseline
   gap, in absolute terms and corrected for multiple testing across
   the 9 month-pair cells.

---

## 6.5 The clean one-paragraph version, for a paper

> In high-frequency pairs trading on liquid 2024–2025 FX data, a
> two-regime Markov-switching AR(1) regime classifier adds value
> primarily as a **soft gearbox** (continuous probability-weighted
> entry threshold) rather than as a **hard kill switch** (binary
> liquidation on high Danger probability): the gearbox dominates the
> kill switch in 8 of 9 month-pair experiments and 47 of 81
> sensitivity-sweep cells. Neither HMM-based variant beats the
> unfiltered z-score baseline in the majority of cleanly cointegrated
> months. The hard-kill filter outperforms the baseline in exactly
> one month (GBPUSD/EURUSD, August 2024), in which it sat in cash 97
> percent of the time and captured a Sharpe of +23.5 against the
> baseline's −0.5 — robust across 25 of 27 parameter-grid cells.
> Crucially, in that same month a passive long Buy & Hold of the
> spread captured +215.6 bps while the best filtered strategy
> captured only +44.0 bps, indicating that the filter's contribution
> was **defensive damage control during a directional drift**, not
> alpha capture. Across all 9 month-pair cells no traded strategy
> outperformed Buy & Hold in absolute PnL terms, and the negative
> control (EURNOK/EURSEK August 2025) demonstrated that the
> symmetric z-score trading rule systematically loses money to
> directional drift even when passive exposure makes money. We
> conclude that the HMM regime filter is correctly *detecting*
> high-variance regimes but the underlying z-score trading rule is
> sign-blind to drift direction, so the filter delivers capital
> preservation rather than alpha capture. A natural next step is a
> duration- and direction-aware meta-strategy that estimates both
> regime persistence (via the transition-matrix diagonal) and drift
> sign (via a slope test on the spread), and dynamically switches
> between trading modes — short-mean-reversion, abstain, or trend-
> follow — accordingly.

---

## 6.6 A short personal note for the writer

Looking at the spread of nine cells once you include Buy & Hold, there
are three stories to tell:

- A **disappointing one**: "regime filters don't beat the baseline,
  and in our one stress month none of the trading strategies even
  matched a passive long".
- A **constructive one**: "regime filters detect what they're supposed
  to detect, but the underlying trading rule is the bottleneck — the
  z-score is sign-blind to drift, so the filter delivers damage
  control rather than alpha. The next experiment is to fix the
  trading rule."
- An **honest one** (the synthesis): "across 9 month-pair experiments
  the most consistent finding is that the spread drifts, the
  symmetric z-rule fights the drift, and any pairs-trading framework
  on this data is competing against passive exposure rather than
  capturing reversion. We characterised when and why each of the
  three strategies wins or loses, and identified the trading rule
  itself as the bottleneck."

The third story is the most publishable. It's a clean methodological
finding (the HMM works, the trading rule doesn't), it points
explicitly at the next experiment (drift-aware gating before the
trading rule), and it doesn't oversell the AR result by ignoring
BuyHold. That's the version to write up.

The project succeeded at what it was actually testing: not "can
MS-AR make money" (we don't have the sample size for that) but
"what is the structural relationship between an HMM regime filter
and a baseline z-score pairs trade, and where are the boundaries of
that relationship". We have a clean answer to that — including the
honest acknowledgement that the BuyHold benchmark beats every
trading strategy in the one month the filter "wins", which is the
most informative limitation in the dataset.
