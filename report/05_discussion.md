# 5. Discussion

This chapter is the analytical heart of the report. We use the 9
month-pair experiments and the 81 sensitivity-sweep cells from
Chapter 4 to answer:

1. Does the HMM regime filter add value over an unfiltered Baseline? (§5.1)
2. When does it pay off? (§5.2)
3. Why doesn't it pay off otherwise? (§5.3)
4. What's the structural difference between AR and MS-AR? (§5.4)
5. The NOKSEK negative control. (§5.5)
6. What we explicitly did *not* find. (§5.6)
7. Why are the absolute Sharpe values so large? (§5.7)
8. Specific vs general learnings. (§5.8)

---

## 5.1 Does the regime filter pay?

The single-question summary, on the 6 cleanly-cointegrated month-pairs
(excluding NOKSEK):

- **Baseline** outperforms both HMM strategies in **4/6** cells
  (AUDNZD-Sep24, AUDNZD-Aug25, GBPEUR-Sep24, GBPEUR-Aug25).
- **AR (hard kill)** outperforms Baseline only once: **GBPEUR-Aug24**.
- **MS-AR (soft gearbox)** outperforms Baseline only in
  AUDNZD-Aug24 — and "outperforms" means "loses 5.1 Sharpe instead of
  6.4 Sharpe", which is a marginal month for both strategies.

**The plain answer is no, the regime filter does not generally pay** in
calm cointegration months. The unfiltered z-score strategy is the
dominant choice when the spread is cleanly mean-reverting.

This is, in some sense, the *textbook* reason pairs trading is appealing
in the first place: a stable cointegration relationship doesn't *need*
a regime detector. The danger is when cointegration breaks down — and
that's when the filter could earn its keep, but only if it correctly
identifies the breakdown and only if the breakdown is persistent
enough to justify the cost of switching off.

---

## 5.2 When does the regime filter pay? The GBPEUR August 2024 case study

Look closely at the **GBPEUR August 2024** row from §4:

| Strategy | Sharpe | PnL bps | Trades |
|---|---:|---:|---:|
| Buy & Hold | (populate) | (populate) | — |
| Baseline   | −0.49 |  −5.9 |  93 |
| **AR**     | **+23.53** | **+44.0** | **19** |
| MS-AR      | +4.39 | +39.1 |  46 |

This is **the only month in our dataset where the regime filter
actually fires and helps**. Three facts together explain it:

1. **Baseline's PnL is essentially flat.** −0.49 Sharpe, −5.9 bps, 93
   trades — about three trades per trading day. The spread is *not*
   cleanly mean-reverting in this month. The un-filtered z-score gets
   whipsawed and barely scrapes even after costs.
2. **AR cuts 80 % of trades.** 19 trades instead of 93. The HMM is
   correctly identifying a persistent high-variance regime — most of
   the month — and the hard kill keeps the strategy in cash through
   it.
3. **The 19 trades AR keeps are high-quality.** 19 round trips earning
   +44.0 bps is ~2.3 bps per round trip, vs Baseline's near-zero
   average. The filter isn't just *avoiding* bad trades; it's keeping
   the actually-good ones.

In other words, GBPEUR-August-2024 is a **genuine regime-switching
month** for that pair. The MS-AR(1) detected a persistent shift in
spread dynamics, and the hard kill exploited it.

**Why does MS-AR fail in this same month?** Because the gearbox isn't
aggressive enough. Even at high Danger probability the entry threshold
widens only to $z_v = 2.5$ — apparently still narrow enough that 46
trades go through, and the average quality is much lower than AR's
19. The soft gearbox preserves trade frequency at the cost of trade
quality.

The sensitivity sweep confirms this isn't a parameter artefact: in
25 / 27 grid cells for this pair-month, AR > MS-AR. We tried widening
$z_v$ to 3.5 — AR still wins. We tried tightening $\delta$ to 0.15
(killing more aggressively) — AR still wins.

The lesson from this single cell: **for genuinely persistent regime
shifts, a hard kill outperforms a soft gearbox**. The cost-benefit
trade-off flips when the bad regime persists for weeks rather than
minutes.

---

## 5.3 Why doesn't the regime filter pay in the other 8 cells?

Because in those months, the spread is in a *single quasi-stationary
regime* for the entire month. The HMM's Danger regime never
persistently fires (or fires only transiently in single bars / brief
bursts). What happens then:

- **AR (hard kill):** every brief Danger flicker forces a liquidation
  at the worst possible moment (the moment the HMM thinks the trade is
  about to lose). A bar or two later, the HMM reverts to MR and the
  strategy wants to re-enter at a similar z-score. The result is an
  entry → kill → re-entry cycle that bleeds slippage on every cycle.
  Hence AR's Sharpe drops well below Baseline in calm months.
- **MS-AR (soft gearbox):** brief Danger flickers raise
  $z_q^{\text{eff}}$ from 1.3 to maybe 1.7 for a few bars, but
  existing positions are kept. New entries during the flicker are
  suppressed — *some* signal is lost but no slippage is paid. MS-AR's
  Sharpe stays close to Baseline rather than collapsing.

This is the conceptual picture:

| Regime structure of the month | Best strategy | Why |
|---|---|---|
| **Quasi-stationary** (one regime, calm) | Baseline | Filter cost > filter benefit |
| **Transient regime flickers** (HMM fires briefly) | MS-AR ≈ Baseline | Soft gating preserves value, hard kill destroys it |
| **Persistent regime switch** (HMM fires for days) | AR | Hard kill avoids a long bad regime |

The fundamental tension: we have **one filter** but **three regime
modes**. A perfect strategy would dynamically choose between Baseline
behaviour, soft gating, and hard killing based on the *persistence*
of the detected regime — but we don't have a clean persistence
signal in the current implementation.

---

## 5.4 The MS-AR vs AR tradeoff, structurally

MS-AR beats AR in **8 of 9** main-test cells and **47 of 81** sensitivity
cells (the only large pocket of AR wins is GBPEUR-Aug-24's 25/27).

Structural reasons MS-AR dominates AR in most contexts:

1. **No slippage on regime flickers.** AR liquidates and re-enters at
   the half-spread; MS-AR pays nothing extra to widen the threshold.
2. **Preservation of profitable in-progress trades.** A position that
   is half-way to mean reversion is closed by AR when Danger spikes
   transiently — the un-realised gain becomes a paid spread and a
   reset position. MS-AR holds and lets the trade complete.
3. **Continuous response to a continuous signal.** The HMM produces a
   continuous posterior $\text{Danger\_Prob} \in [0, 1]$. Binarising
   it with a single threshold $\delta$ throws away most of the
   information. MS-AR uses the posterior as a continuous weighting,
   which more efficiently encodes the model's uncertainty.

The single case where AR beats MS-AR (GBPEUR-Aug-24) is the case where
the regime is so *persistent* that the slippage cost of repeated kill
cycles never materialises — there's basically one big kill that lasts
most of the month.

**Practical recommendation:** if you're going to use the HMM at all,
MS-AR is the safer default. AR is the right choice only if you have
prior reason to believe the month contains a persistent regime
switch — which, by construction, you don't usually know in advance.

---

## 5.5 The NOKSEK story: a negative control

EURNOK/EURSEK loses money under every strategy in every month
(Baseline: −44 to −52 Sharpe; AR: −54 to −103 Sharpe; MS-AR: −31 to
−44 Sharpe). Why?

The pair is the textbook Scandi cross-rate trade — both currencies
nominally track the EUR via similar economic mechanisms, so the spread
"should" be cointegrated. Empirically in 2024–25, the Norwegian Krone
and Swedish Krona have moved out of sync, driven by their own central
bank reaction functions and divergent commodity exposures
(NOK ↔ Brent crude, SEK ↔ industrial metals). The cointegrating
relationship is unstable; the half-life of mean reversion, if it
exists at all on these months, is on the order of weeks.

When the underlying spread doesn't mean-revert, **no z-score-based
strategy can win**. The "mean" the z-score is normalising against is a
moving target.

The breakdown of strategies in NOKSEK:

- **Baseline** loses ~50 Sharpe/month — large, regular drawdowns from
  trading reversions that never happen. The rolling z-score happily
  signals deviations, but each deviation just continues to widen.
- **AR** loses *worse* (~50% bigger losses) — the kill switch
  correctly identifies the noise as "Danger" and sits out... but then
  the HMM briefly re-flags MR (false dawn), the strategy re-enters,
  gets re-killed, and pays slippage on the cycle. NOKSEK is the
  textbook example of *transient regime flickers* destroying value
  for the hard kill.
- **MS-AR** loses *least* — the soft gearbox suppresses the worst
  entries and doesn't compound losses from kill cycles. But it still
  loses 35–44 Sharpe. *Less bad* is not *good*.

**MS-AR's NOKSEK result is the most ambiguous evidence in our dataset.**
It's the only context where MS-AR > Baseline in 3 / 3 months — but the
entire pair is losing in absolute terms.

The honest interpretation: **NOKSEK is a negative control**. Including
it confirms the filters are doing something (the HMM clearly detects
the noise), but the takeaway is that **no filter rescues a broken
pair**. The filter is a refinement on top of a real edge, not a
substitute for one.

---

## 5.6 What we explicitly did NOT find

Things we explicitly looked for in the data and did not see:

1. **Consistent MS-AR > Baseline.** In 8 / 9 cells excluding NOKSEK,
   Baseline was at least as good. MS-AR's advantage over Baseline is
   not robust.
2. **A parameter regime where MS-AR > Baseline universally.** The
   sensitivity sweep covers 27 combinations × 3 pairs and includes
   many cells where the MS-AR > Baseline margin is small or where
   both are losing. There is no sweet-spot $(z_q, z_v, \delta)$ that
   makes the gearbox structurally beat the un-filtered baseline.
3. **A regime where AR > MS-AR universally.** Only GBPEUR-Aug-24 has
   that pattern (25/27 sweep cells favour AR). Outside that single
   month-pair, MS-AR is robustly better.
4. **A statistically distinguishable edge for any HMM-based strategy
   over Baseline at $n = 9$.** Even ignoring multiple-testing
   corrections, a paired test on the 6 non-NOKSEK Sharpes wouldn't
   reject the null at any reasonable level. With 9 observations and a
   high-variance metric like monthly Sharpe, we don't have the
   statistical power to claim a long-run mean difference.
5. **A "third regime" effect.** The default is $K = 2$. Briefly
   trying $K = 3$ (commented out in `MONTH.DEFAULT_CFG`) does not
   produce qualitatively different results — the EM splits the
   Danger regime into two micro-variants but the smoothed Danger
   posterior aggregates back to roughly the same series.

---

## 5.7 Why are the absolute Sharpe values so large?

The tick-clock annualisation factor $F = 252 \times 24 \times 60 \approx
3.6 \times 10^5$ scales any bar-clock Sharpe by $\sqrt{F} \approx 600$.
Equivalently, our tick-clock Sharpes are about $\sqrt{F / 252} \approx
38\times$ larger than the daily-returns Sharpe of the same strategy.

A daily Sharpe of 0.5 — perfectly respectable for a working strategy —
comes out as ≈ 19 in our convention. A daily Sharpe of 0.0 comes out
as 0. A daily Sharpe of −0.5 comes out as ≈ −19. So in interpretation:

- **Sharpe near 0** in our convention ≈ no edge.
- **|Sharpe| around 20** in our convention ≈ ±0.5 daily Sharpe, which
  is a real (positive or negative) edge.
- **|Sharpe| around 50** would be a daily Sharpe of ±1.3, exceptional.

The cross-strategy comparison within a single month and pair is
unaffected by $F$ (it cancels in the ratio). The cross-pair / cross-
month comparison is *approximately* preserved (the bar rate varies
slightly between pairs and months). The comparison to *published* daily
Sharpes is wildly inflated.

A more principled future implementation would correct for autocorrelation
in bar returns via Newey–West standard errors or a block bootstrap.
For this report the cross-strategy ranking is the headline; the
absolute number is presentational.

---

## 5.8 Specific vs general learnings

### Specific (about this dataset)

- **GBPEUR August 2024** is a single example where the hard-kill HMM
  filter dramatically outperformed Baseline. Sharpe +23.5 vs −0.5; AR
  cut 80 % of Baseline's trades and earned 44 bps where Baseline lost
  6 bps. This is the most empirically interesting single cell in the
  9-month grid and warrants a focused write-up (plot the
  per-fold Danger variance and Danger posterior; identify the calendar
  windows when the kill was active; check macro calendars for known
  events).
- **NOKSEK** is structurally broken on tick data in 2024–25. No
  trading filter rescues it. It serves as a useful negative control
  and a check that our HMM detects noisy spreads correctly (it does —
  Danger probability is persistently high).
- **AUDNZD** performs best with Baseline in calm months (Sep-24,
  Aug-25) and roughly tied with MS-AR in the marginally-losing month
  (Aug-24). It's the cleanest "no need for a filter" example.
- **3-day rolling MS-AR(1) training** is sufficient for the regime
  model to converge on every trading day across every pair × month
  combination we tested. No skipped folds. The 500-tick-bar resolution
  gives the EM enough observations to identify a clean two-regime
  structure within three days.

### General (about the methodology)

- **Soft beats hard, structurally.** MS-AR is uniformly preferable to
  AR as a way to use an HMM regime classifier, unless you have prior
  reason to believe the period contains a *persistent* regime switch.
  The slippage cost of hard kills around transient regime boundaries
  is greater than the avoided-loss benefit in most market conditions.
- **No filter beats both filters in calm cointegration months.** The
  regime filter is a refinement on top of a working edge, not an
  independent source of edge. If your underlying pair is cleanly
  mean-reverting, adding an HMM filter strictly destroys value (more
  parameters, more sensitivity to model misspecification, no upside).
- **The regime filter pays off rarely but dramatically.** 1 / 9 cells
  in our sample, with +24 Sharpe gain. The question of whether that
  rate of pay-off justifies the filter depends on how often the rare
  event occurs in live deployment — for which 9 observations is far
  too few to estimate.
- **End-of-day flattening is essential for clean experimentation.**
  Without it the gearbox can produce hundreds of overnight trade
  flips, washing out the entire edge. With it, days are independent
  experiments and the metric tables become interpretable.
- **Fixed-parameter testing reveals more than WFO-optimised testing.**
  The earlier two-year run with Optuna optimisation gave MS-AR a
  Sharpe of −3.x and 9 235 trades because Optuna picked extreme
  parameters that maximised in-sample Sharpe via churn. Fixed
  reasonable parameters expose the actual underlying dynamics.
- **Small, hand-picked experiments are more diagnostic than long
  aggregates.** The two-year aggregate hid the GBPEUR-Aug-24 finding
  by averaging it with the calm months. Splitting into 9 separate
  month-pairs surfaced the result.

---

## 5.9 What would change the conclusions?

A reasonable reader might ask: under what conditions would our
conclusions flip?

- **If we tested more months of stress.** Our 9-cell sample contains
  exactly one stress month. A wider sample — say, picking 20 months
  known *a priori* to contain macro events (Fed decisions, BOJ
  intervention, BOE surprises) — would increase the rate at which AR
  beats Baseline. The "regime filter has rare-event value" finding
  would become more robust.
- **If we modelled regime-dependent transaction costs.** Spreads in
  the Danger regime widen 2–5× in real markets. We use a fixed
  half-spread series from the data, but a more realistic cost model
  would penalise Baseline more in Danger regimes (where it's
  trading) and penalise AR / MS-AR less (where they're sitting out).
  This would tilt the results toward the HMM strategies.
- **If we used a different regime model.** Heavy-tailed innovations
  (Student-t MS-AR), more regimes ($K = 3$), or a Hidden Semi-Markov
  Model that explicitly models regime duration would give different
  Danger posteriors. The duration-aware variant is the most likely
  to improve on the current results — it could distinguish transient
  from persistent regimes and dynamically choose between MS-AR-like
  and AR-like behaviour.
- **If we removed `flatten_eod`.** The two-year runs showed this can
  add ~100 % trade count and turn marginal MS-AR wins into losses.
  But it also lets a strategy *carry* a profitable position
  overnight, which is sometimes the right call. The trade-off
  depends on whether the in-sample monthly noise is dominated by
  intra-day or overnight effects.

None of these would invalidate the basic methodological finding
(MS-AR ≥ AR almost always), but they would all shift the
"Baseline vs MS-AR" comparison.
