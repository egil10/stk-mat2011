# 1. Introduction

## 1.1 Context

High-frequency pairs trading is one of the oldest applications of statistical
arbitrage. The setup is simple: two assets that share a common economic
driver tend to move together in the long run, even if they wander apart in
the short run. The arbitrageur trades the wandering — short the rich leg,
long the cheap leg — and waits for the spread to converge.

The mathematical structure underlying this trade is **cointegration**. Two
non-stationary log-price series $\log A_t$ and $\log B_t$ are cointegrated
if there exists a hedge ratio $\beta$ such that the spread

$$
S_t \;=\; \log A_t - \beta \log B_t
$$

is stationary. The stationary spread becomes the tradeable signal: it has a
well-defined mean, a well-defined volatility, and (under stationarity) a
finite mean-reversion half-life.

In high-frequency data two complications appear:

1. The hedge ratio $\beta$ is **not constant in time**. Intra-day flow,
   liquidity events and overnight repricing all drift $\beta$. We need a
   rolling estimate.
2. The spread does **not always mean-revert**. There are *stress regimes*
   where the spread keeps drifting and the textbook pairs trade racks up
   losses. Detecting these regimes is the second modelling challenge.

## 1.2 Research question

This project asks a single, narrow methodological question:

> **Does a Markov-switching regime classifier add value when used as a
> filter on top of standard pairs trading on tick data?**

The classifier we fit is a two-regime Markov-switching AR(1) on the
cointegration spread. At every bar it produces a posterior probability that
the spread is currently in a *Mean-Reverting* (MR / quiet) regime versus a
*Danger* (DR / volatile) regime. We then build three trading strategies
that use this signal differently, and compare them against a passive
benchmark:

- **Buy & Hold** — always long the spread. The do-nothing benchmark.
- **Baseline** — naïve z-score pairs trade. Ignores the regime classifier.
- **AR** — z-score pairs trade with a **hard kill switch**: liquidates and
  stays in cash whenever the Danger probability exceeds a threshold.
- **MS-AR** — z-score pairs trade with a **soft gearbox**: continuously
  widens the entry z-band as the Danger probability rises, but never kills
  an open position.

Comparing these four across multiple pairs and multiple months is the
experiment. The key claim under test is whether the *information* in the
HMM regime probabilities is usable in a way that improves on doing nothing
with that information (Baseline).

## 1.3 Why pairs, why tick data, why MS-AR?

**Pairs over single-asset.** Single-asset HF strategies require predicting
the sign of returns, which is famously hard. Pairs trading reduces the
problem to predicting *mean reversion of a spread*, which is significantly
more tractable: stationarity of the spread is a testable property and
gives a natural set of entry/exit thresholds (z-score bands).

**Tick data over OHLC.** Microstructure effects — bid-ask bounce, order-flow
clustering, fleeting arbitrage opportunities — are completely invisible at
daily resolution. The MS-AR(1) classifier specifically benefits from
high-resolution data because regime *posteriors* are estimated more
sharply with more observations, and the regime transition matrix is
identifiable on minutes/hours but not on days within a single calendar
month.

**MS-AR over GARCH-style filters.** GARCH models the conditional variance as
a single continuous process. MS-AR(1) models the spread as switching
between distinct discrete regimes, each with its own AR coefficient,
mean and innovation variance. This is closer to the textbook intuition of
"calm market vs stressed market" and produces a cleaner *gate* signal — a
single posterior probability between zero and one. That gate signal is
the natural input to a kill switch (AR) or to a probability-weighted
entry threshold (MS-AR).

## 1.4 Scope and limitations

The scope of this project is deliberately narrow:

- **Three pairs** — AUDUSD vs NZDUSD, GBPUSD vs EURUSD, EURNOK vs EURSEK.
  These were chosen to cover one strongly-cointegrated pair (AUDNZD), one
  marginally-cointegrated pair (GBPEUR) and one historically-cointegrated
  but now-broken pair (NOKSEK).
- **Three months across two years** — August 2024, September 2024, August
  2025. Months were chosen for data availability and to give one
  calm/normal pair (Sep-24), one mildly stressed (Aug-24) and one different
  macro vintage (Aug-25).
- **One bar resolution** — 500 tick bars throughout.
- **Fixed trading-rule parameters** — no per-month optimisation. This is
  deliberate: the goal is to test whether the regime filter has
  *structural* value, not whether you can tune three z-thresholds to a
  specific month. Optimised parameters introduce a confounding
  in-sample-fit signal that swamps the regime effect, as the earlier
  two-year runs in this repo showed.
- **One transaction-cost model** — flat 0.5 bps fee per trade plus
  half-spread slippage. Adequate for liquid majors in normal conditions,
  but understates costs in the Danger regime where real spreads widen.

This is a controlled experiment, not a production strategy. The point is
to test a single methodological hypothesis cleanly.

## 1.5 What we found, in two sentences

The soft gearbox (MS-AR) is uniformly a better way to use the HMM than the
hard kill (AR): MS-AR beats AR in 8 of 9 month-pairs and in 56 of 81
sensitivity-sweep cells. However, neither HMM-based strategy beats the
un-filtered Baseline in calm cointegration months — the regime filter pays
off only in months that contain a *genuine, persistent* regime switch, of
which we found exactly one (GBPUSD vs EURUSD, August 2024) in our 9
month-pair experiments.

The rest of this report unpacks those two sentences.
