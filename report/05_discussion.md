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

Look closely at the **GBPEUR August 2024** row from §4, now with the
Buy & Hold column included:

| Strategy | Sharpe (MONTH) | Sharpe (TS) | PnL bps | Trades | Exposure |
|---|---:|---:|---:|---:|---:|
| **Buy & Hold** | — | **+4.83** | **+215.6** |   0 | 99.8 % |
| Baseline       | −0.49 | −0.16 |  −5.9 |  94 | 70.0 % |
| **AR**         | **+23.53** | +7.72 | +44.0 | **19** | **3.2 %** |
| MS-AR          | +4.39 | +1.44 | +39.1 |  46 | 37.1 % |

This is the most empirically interesting cell in the dataset, but the
BuyHold column changes the headline from "AR captured a regime switch"
to something more nuanced.

### 5.2.1 The real driver: the spread drifted, hard

BuyHold making **+215.6 bps** with zero trades means the spread
$\log A - \beta \log B$ moved roughly 2 percent in one direction over
the month. That is huge for a cointegrated pair on a 22-day window. It
implies the cointegration relationship itself shifted — either β moved
or the long-run mean repriced.

Pairs trading is built on the assumption that the spread oscillates
around a stable mean. When the mean walks away from you, the z-score
rises and rises, and the strategy keeps opening shorts that never come
back. **That's what wrecks the Baseline in this month.** 94 trades,
exposure 70 % of bars, almost all shorts against a rising spread,
ending −5.9 bps after costs.

### 5.2.2 What AR is actually doing

The HMM was not "predicting a Lehman-style flash regime". It was
detecting *high variance and persistence in the spread*, which is
exactly what a strongly-drifting spread looks like to a two-regime
classifier: the Danger regime fires because the AR(1) dynamics during
the drift have a larger innovation variance than during the calm
non-drift baseline of the previous training days.

When that gate fires and stays fired:

- AR's exposure collapses from Baseline's 70 % to **3.2 %**. The
  strategy sits in cash 97 % of the time.
- The 19 trades it does take are the rare windows where the HMM
  briefly flips back to MR — short reversion bursts during the drift.
- These average ~2.3 bps per round trip; small but consistently
  positive.

So AR doesn't *capture* the drift. It **avoids fighting it**. The
trade-cost savings from being out plus the small reversion-burst
profits net to +44 bps. Baseline, by contrast, fought the drift the
whole month.

### 5.2.3 The honest comparison

AR vs Baseline → AR wins by 50 bps. ✓ (the +24 Sharpe gap we headlined)
AR vs Buy & Hold → BuyHold wins by **172 bps**. AR captures roughly
20 % of the available drift.

There are two ways to read this:

- **Charitably**: AR is a *pairs-trading strategy*, not a directional
  one. Comparing it to directional Buy & Hold isn't fair — the user
  who runs AR isn't choosing between AR and BuyHold; they're choosing
  between AR and other pairs-trading variants. Among those, AR wins
  clearly.
- **Sceptically**: if BuyHold beats AR by 172 bps, the right question
  isn't "is the regime filter helpful?" but "why is the cointegration
  premise applicable here at all?" The strongest argument for pairs
  trading is when the spread is stationary; when the spread
  half-trends 200 bps in a month, you'd rather just be long it.

The sceptical reading suggests a richer experimental design: every
trading-day's decision could include an upstream "is this even a
cointegrated regime?" test, and abstain entirely (or switch to a
trend-following stance) when the answer is no. That's a different
project from what we ran.

### 5.2.4 Why MS-AR fails in this exact month

MS-AR's gearbox widens the entry band but the bot still trades 46
times (vs AR's 19, vs Baseline's 94). Each trade fights the drift to
some degree. Net: +39 bps — better than Baseline, worse than AR.

The sweep confirms it isn't a knob-tuning artefact: in **25 / 27**
grid cells for GBPEUR-Aug-24, AR > MS-AR. Even widening $z_v$ to 3.5
or tightening $\delta$ to 0.15 doesn't get MS-AR past AR. The hard
gate is structurally better here because the bad regime is the entire
month — partial-throttle is still too much throttle when the path
itself is the problem.

### 5.2.5 The lesson, restated

For **persistent drift regimes**, hard-kill (AR) beats soft-gating
(MS-AR), and both beat un-filtered Baseline, **but none of them beats
just being long the spread**. The HMM has detected the broken pairs
premise but the trading rule isn't capable of pivoting from "short
mean reversion" to "ride the drift". The filter is doing capital
preservation, not alpha capture.

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

## 5.5 The NOKSEK story: a negative control with a twist

EURNOK/EURSEK loses money under every strategy in every month
(Baseline: −44 to −52 Sharpe; AR: −54 to −103 Sharpe; MS-AR: −31 to
−44 Sharpe). The Buy & Hold benchmark tells the deeper story:

| Month | BuyHold bps | Baseline bps | AR bps | MS-AR bps |
|---|---:|---:|---:|---:|
| 202408 |   −267.8 | −2 287.7 | −2 117.5 | −1 583.0 |
| 202409 |    −57.0 | −2 202.4 | −2 074.4 | −1 531.9 |
| **202508** | **+110.1** | **−664.9** | **−255.5** | **−349.4** |

The pair is the textbook Scandi cross-rate trade — both currencies
nominally track the EUR via similar mechanisms, so the spread
"should" be cointegrated. Empirically in 2024–25, the Norwegian Krone
and Swedish Krona moved out of sync, driven by their own central
bank reaction functions and divergent commodity exposures
(NOK ↔ Brent crude, SEK ↔ industrial metals). The cointegrating
relationship is unstable.

### 5.5.1 Two failure modes side by side

**Aug & Sep 2024 — drift against pairs trading.** The spread drifts
*down* (BuyHold −268 and −57 bps respectively). Pairs trading
interpreted persistently negative z-scores as "buy the discount", went
long, and was carried down with the spread for the rest of the month.
Strategies amplified BuyHold's loss roughly **10×** via leveraged longs
and churn. The HMM did its job — Danger probability is high throughout
both months — but the kill gate just delays the long-entry; it doesn't
prevent it. So AR loses about the same as Baseline (slightly less
exposure, similar net damage).

**Aug 2025 — BuyHold *positive*, all strategies negative.** This is the
most damning evidence in the entire experiment. The spread did
something *favourable* over the month (+110 bps drift up), yet every
trading strategy lost money — Baseline by 665 bps, AR by 255, MS-AR by
349.

Why? The strategies were *shorting* the up-drift. The z-score was
high (spread had risen), classifier flagged Danger because of high
variance during the drift, AR's kill gate fired infrequently but the
strategy still managed 29 trades — and those trades almost all went
the wrong way. Baseline did the same thing 96 times. *MS-AR was the
only one that lost less than 50 % of Baseline's damage*, suggesting
the gearbox at least suppressed some of the worst entries.

### 5.5.2 What NOKSEK really shows

**NOKSEK is not failing because the pair is "noisy" or "uncointegrated".**
The traditional negative-control framing — "no filter rescues a broken
pair" — is true but incomplete. What NOKSEK demonstrates more
specifically is that:

> **The z-score trading rule is not symmetric in how it handles drift.**
> When the spread drifts in the direction the z-score wants to take you
> (calm months: long-bias by oscillation), the strategy makes money.
> When the spread drifts in the opposite direction (NOKSEK 2024),
> the strategy compounds losses by repeatedly opening shorts into the
> drift.

The HMM filter cannot fix this because both directions of drift look
the same to a variance-based classifier — both produce a high-variance
regime. The filter sees "Danger"; the trading rule still picks a
direction based on z-sign, and that direction is determined by the
drift itself, not by any persistent mean.

### 5.5.3 So is the filter doing anything on NOKSEK?

MS-AR consistently loses less than Baseline on NOKSEK (by 30–80 %
across the 3 months). So yes, the gearbox is suppressing the worst
entries. But going from "lose 2 200 bps" to "lose 1 530 bps" isn't an
edge — it's *damage control on a strategy that should not be running*.

**The honest takeaway**: NOKSEK shows the filter is *active* (the HMM
detects something real) but *insufficient* (the trading rule's
direction-sign is wrong). Fixing this requires changing the trading
rule, not the filter — e.g. adding a separate cointegration-stability
test that abstains entirely from drift months, or replacing the
symmetric z-score entry with an asymmetric rule that explicitly
prefers the direction of the recent drift.

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
4. **A strategy that beat Buy & Hold in absolute terms during
   GBPEUR-Aug-24.** BuyHold made +215.6 bps; the best strategy made
   +44 bps. AR's "win" is a win *vs Baseline*, not a win in absolute
   terms vs passive exposure.
5. **A "calm-cointegration" month where Baseline far outperformed
   passive.** Even in the best Baseline months (AUDNZD-Sep24 +312
   vs BuyHold +99; GBPEUR-Sep24 +322 vs BuyHold +151), the pairs-trade
   alpha on top of the drift was 2× the passive return — meaningful,
   but not the order-of-magnitude edge the inflated tick-clock Sharpe
   numbers in §4.1 might suggest. Most of the Baseline PnL is just
   "spread happened to drift the way the z-rule liked".
6. **A statistically distinguishable edge for any HMM-based strategy
   over Baseline at $n = 9$.** Even ignoring multiple-testing
   corrections, a paired test on the 6 non-NOKSEK Sharpes wouldn't
   reject the null at any reasonable level.
7. **A "third regime" effect.** The default is $K = 2$. Briefly
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

- **GBPEUR August 2024** is a strong-spread-drift month
  (BuyHold +215.6 bps). The hard-kill HMM filter cut 80 % of trades,
  achieved Sharpe +23.5 vs Baseline's −0.5, and captured ~20 % of the
  available drift. AR's value here is **damage control on a strategy
  fighting the drift**, not edge capture. Buy & Hold beat all three
  strategies by a wide margin. This is the most empirically
  interesting single cell in the 9-month grid and warrants a focused
  write-up (plot the per-fold Danger variance, the spread itself,
  identify when AR was in cash, cross-check against macro calendars).
- **NOKSEK** is structurally broken on tick data in 2024–25. **The
  most damning result is NOKSEK-Aug-25**: BuyHold made +110 bps but
  every traded strategy lost 250–665 bps. The pair isn't "untradeable
  because it's noisy" — it's "untradeable because our trading rule
  systematically shorts directional drift". The HMM correctly flags
  Danger; the trading rule's direction-sign is wrong; the filter can't
  fix that.
- **AUDNZD** performs best with Baseline in calm months (Sep-24,
  Aug-25). The pair is the cleanest example of a working cointegration
  trade in the dataset; Baseline beats BuyHold by 2–3× in profitable
  months, confirming there is mean-reversion alpha on top of the drift.
- **Every "profitable" month in the dataset is also a drift month.**
  We do not have a clean example of "the spread oscillated around a
  flat mean and Baseline harvested the oscillations". AUDNZD-Aug-24
  was the closest to flat (BuyHold +13 bps) and all strategies still
  lost there.
- **3-day rolling MS-AR(1) training** is sufficient for the regime
  model to converge on every trading day across every pair × month
  combination we tested. No skipped folds. The 500-tick-bar resolution
  gives the EM enough observations to identify a clean two-regime
  structure within three days.

### General (about the methodology)

- **The spread rarely sits still.** Every one of our 9 month-pairs
  exhibited a non-trivial directional drift in the spread over the
  month. The strongest pairs-trade alpha came from Baseline
  *partially capturing* the drift via its z-rule's directional bias;
  the worst losses came from Baseline *fighting* the drift the wrong
  way (GBPEUR-Aug-24, NOKSEK throughout). This suggests that **before
  applying any regime filter, an HF pairs-trade should first test
  whether the cointegration premise (stable mean) actually holds for
  the period being traded**.
- **Soft beats hard, structurally.** MS-AR is uniformly preferable to
  AR as a way to use an HMM regime classifier, unless you have prior
  reason to believe the period contains a *persistent* regime switch
  or directional drift the strategy is fighting. The slippage cost of
  hard kills around transient regime boundaries is greater than the
  avoided-loss benefit in most market conditions.
- **No filter beats both filters in calm cointegration months.** The
  regime filter is a refinement on top of a working edge, not an
  independent source of edge. If your underlying pair is cleanly
  mean-reverting, adding an HMM filter strictly destroys value.
- **The regime filter pays off rarely and modestly, never wins the
  absolute race.** 1 / 9 cells where AR beats Baseline; 0 / 9 cells
  where AR beats Buy & Hold. The filter is a *relative* refinement.
- **The HMM's "Danger" regime is a high-variance marker, not a
  direction marker.** It fires both when the spread is volatile
  *around* its mean and when the spread is *drifting away* from its
  mean. The trading rule, which only reads `Z_Score` (a sign and a
  magnitude), cannot tell which case it's in. Both kill (AR) and
  gearbox (MS-AR) reduce damage in drift months, but neither captures
  the drift.
- **End-of-day flattening is essential for clean experimentation.**
  Without it the gearbox can produce hundreds of overnight trade
  flips, washing out the entire edge.
- **Fixed-parameter testing reveals more than WFO-optimised testing.**
  The earlier two-year run with Optuna optimisation gave MS-AR a
  Sharpe of −3.x and 9 235 trades because Optuna picked extreme
  parameters that maximised in-sample Sharpe via churn.
- **Small, hand-picked experiments are more diagnostic than long
  aggregates.** The two-year aggregate hid the GBPEUR-Aug-24 finding
  by averaging it with the calm months. Splitting into 9 separate
  month-pairs surfaced both this and the NOKSEK-Aug-25 result.

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
