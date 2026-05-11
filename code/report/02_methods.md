# 2. Methods

This chapter is the mathematical core of the project. It defines, in order:

1. Bar construction from raw ticks  (§2.1)
2. Rolling cointegration  (§2.2)
3. Rolling z-score  (§2.3)
4. The Markov-switching AR(1) regime model  (§2.4)
5. The four trading strategies  (§2.5)  ← *the most important section*
6. PnL accounting  (§2.6)
7. Performance metrics  (§2.7)

Throughout, `code/scripts/...` references point at the file and class that
implements the corresponding maths.

---

## 2.1 Bar construction from raw ticks

The raw data is Dukascopy tick archives delivered as monthly parquet files
of `(timestamp, bid, ask, bid_volume, ask_volume)` rows, one file per
asset per month. The first step is to convert tick streams into uniform
bars.

For asset $X \in \{A, B\}$ we synthesise a mid price at every tick:

$$
M_t^X \;=\; \frac{B_t^X + A_t^X}{2}
$$

where $B_t^X$ is the latest bid quote at the timestamp of tick $t$ (joined
to the ask stream via `pandas.merge_asof(direction='backward')`).

We then aggregate ticks into bars of fixed **tick count** $\Delta$. A bar
closes when $\Delta$ ticks have accumulated; the bar's closing fields are
the values at the timestamp of the last tick in the bar. In all
multi-month experiments we use $\Delta = 500$. Bar-aggregation logic is
in `SPREAD._aggregate_bars`.

For each bar we keep:

- `Asset_A`, `Asset_B` — mid prices of the two legs
- `Log_A = log(Asset_A)`, `Log_B = log(Asset_B)`
- `Return_A = Log_A.diff()`, `Return_B = Log_B.diff()`
- `HalfSpread_A_bps = 0.5 · (Ask − Bid) / Mid · 10⁴` and similarly for B

The half-spread in basis points is later used as a per-trade slippage
estimate.

The choice of $\Delta$ is a bias-variance tradeoff:

- **Larger $\Delta$** ⇒ smoother bars, less microstructure noise, fewer
  observations per day.
- **Smaller $\Delta$** ⇒ more bars (sharper regime posteriors) but more
  bid-ask-bounce noise leaking into the spread.

$\Delta = 500$ gives roughly 200–500 bars per trading day on liquid
pairs in active hours, which is comfortably above the engine's minimum
requirement of $W_\beta + W_z + 10$ bars per training day.

---

## 2.2 Rolling cointegration

We model the cointegrating relationship as a slowly-varying linear
regression of $\log A$ on $\log B$:

$$
\log A_t \;=\; \beta_t \, \log B_t + \alpha_t + \varepsilon_t .
$$

We do **not** assume $\beta$ is constant. Instead we estimate $\beta_t$ via
**rolling OLS** with window size $W_\beta = 150$ bars:

$$
\beta_t \;=\;
\frac{ W_\beta \sum_{s=t-W_\beta+1}^{t} x_s y_s \;-\;
       \left(\sum_s x_s\right)\left(\sum_s y_s\right) }
     { W_\beta \sum_s x_s^2 \;-\; \left(\sum_s x_s\right)^2 }
$$

$$
\alpha_t \;=\; \frac{1}{W_\beta}\!\left( \sum_s y_s - \beta_t \sum_s x_s \right)
$$

with $x_s = \log B_s$, $y_s = \log A_s$, and all sums over the trailing
$W_\beta$ bars including the current one.

Implementation (`ENGINE._rolling_ols`) precomputes cumulative sums of
$x$, $y$, $x^2$, $xy$ so the whole series is computed in $O(N)$ rather
than the naïve $O(NW_\beta)$. This matters for live re-fitting and for
the walk-forward outer loop.

The cointegration **spread** at bar $t$ is then

$$
S_t \;=\; \log A_t \;-\; \beta_t \log B_t \;-\; \alpha_t .
$$

We also compute a no-look-ahead **spread return**:

$$
\Delta S_t \;=\; r_t^A \;-\; \beta_{t-1}\, r_t^B
$$

where $r_t^A = \log A_t - \log A_{t-1}$ etc. The $\beta_{t-1}$ (lagged
hedge ratio) is essential: a position held into bar $t$ was opened with
information available at bar $t-1$, so it earns the spread return
computed with the hedge ratio that was known at the time of opening.

---

## 2.3 Rolling z-score

For the **Baseline** strategy — which knows nothing about regimes — the
trading signal is the rolling z-score of the spread:

$$
Z_t \;=\; \frac{S_t - \mu_t}{\sigma_t},
\qquad
\mu_t \;=\; \frac{1}{W_z}\!\sum_{s=t-W_z+1}^{t} S_s,
\qquad
\sigma_t^2 \;=\; \frac{1}{W_z}\!\sum_{s=t-W_z+1}^{t} (S_s - \mu_t)^2
$$

with $W_z = 50$ bars in the multi-month runs.

Large positive $Z_t$ ⇒ spread is unusually wide relative to its rolling
mean ⇒ expect mean reversion downward ⇒ go short the spread.
Large negative $Z_t$ ⇒ spread is unusually tight ⇒ expect upward
reversion ⇒ go long the spread.

---

## 2.4 The Markov-switching AR(1) regime model

The key modelling step. We fit a **two-regime Markov-switching
autoregression of order 1** to the cointegration spread on each rolling
$T_{\text{train}} = 3$-day training window.

### 2.4.1 Generative model

Let $S_t$ denote the spread (`Spread_Level`) and $K_t \in \{1, 2\}$ a
latent discrete regime indicator. The joint generative model is:

$$
S_t \,\big|\, S_{t-1},\, K_t = k
\;\sim\; \mathcal N\!\left( c^{(k)} + \rho^{(k)} S_{t-1},\; (\sigma^{(k)})^2 \right)
$$

$$
P(K_t = j \mid K_{t-1} = i) \;=\; p_{ij},
\qquad
P \;=\; \begin{pmatrix} p_{11} & p_{12} \\ p_{21} & p_{22} \end{pmatrix},
\qquad p_{i1} + p_{i2} = 1.
$$

This is an AR(1) on the spread with **all three parameters switching**:
the constant $c^{(k)}$, the AR coefficient $\rho^{(k)}$, and the
innovation variance $(\sigma^{(k)})^2$.

The unconditional mean of regime $k$, when $|\rho^{(k)}| < 1$, is

$$
\mu^{(k)} \;=\; \frac{c^{(k)}}{1 - \rho^{(k)}} .
$$

### 2.4.2 Parameter estimation

We fit by maximum likelihood with the EM algorithm in
`statsmodels.tsa.MarkovAutoregression(order=1, k_regimes=2,
switching_ar=True, switching_trend=True, switching_variance=True)`.

The EM iterations alternate between:

1. **E-step.** Compute smoothed regime posteriors
   $\gamma_t^{(k)} = P(K_t = k \mid S_1, \dots, S_T)$ via the
   forward–backward algorithm.
2. **M-step.** Update the regime-specific parameters $(c^{(k)}, \rho^{(k)},
   \sigma^{(k)})$ and the transition matrix $P$ to maximise the expected
   complete-data log-likelihood given the posteriors.

EM convergence is fragile on heavy-tailed financial series. We apply two
preprocessing stabilisers (both are linear transforms, so the regime
structure — $\rho^{(k)}$, $P$, regime *probabilities* — is invariant,
and the estimated constants/variances are unscaled afterwards):

- **Winsorisation.** Clip $S_t$ at $\mathrm{median}(S) \pm 4 \cdot
  \mathrm{std}(S)$ before fitting. Suppresses outliers that would
  otherwise force the EM to allocate a regime to a single bar.
- **Scaling.** Multiply $S_t$ by $10^4$ before fitting. Typical spread
  magnitudes for FX log-prices are $10^{-4}$; rescaling to order one
  puts the EM solver in a numerically well-conditioned regime.

### 2.4.3 Regime classification (labelling)

The Markov chain itself does not assign a meaning to the labels $\{1,
2\}$ — that's done after the fit. We label the regime with the smallest
fitted innovation variance $\sigma^{(k)}$ as **Mean-Reverting (MR /
quiet)** and the regime with the largest variance as **Danger
(DR / volatile)**:

$$
\text{MR index} \;=\; \arg\min_k \sigma^{(k)},
\qquad
\text{DR index} \;=\; \arg\max_k \sigma^{(k)} .
$$

This is purely a convention for downstream code; the underlying chain
doesn't care. (Code: `ENGINE.fit_markov_regimes`, after the M-step.)

### 2.4.4 In-sample regime probabilities

For every bar in the training window we record the smoothed posterior of
being in the MR regime:

$$
\text{MR\_Prob}_t \;=\; \gamma_t^{(\text{MR})}
\;=\; P(K_t = \text{MR} \mid S_1, \dots, S_T)
$$

$$
\text{Danger\_Regime\_Prob}_t \;=\; 1 - \text{MR\_Prob}_t .
$$

These are used by the strategies during back-testing on the training
data and stored for diagnostic plotting.

### 2.4.5 Out-of-sample regime probabilities

For each trading day (the one day immediately after the training window)
we *do not* re-run the smoother. Re-running the smoother on the test
day would let the test-day data update its own posterior — a subtle
form of look-ahead.

Instead we keep the per-regime AR(1) parameters $(c^{(k)}, \rho^{(k)},
\sigma^{(k)})$ frozen at their training estimates and compute a
**one-step analytical posterior** at every test bar:

$$
P(K_t = k \mid S_t, S_{t-1})
\;=\;
\frac{ \mathcal N\!\left(S_t;\; c^{(k)} + \rho^{(k)} S_{t-1},\; (\sigma^{(k)})^2\right) }
     { \sum_{k'} \mathcal N\!\left(S_t;\; c^{(k')} + \rho^{(k')} S_{t-1},\; (\sigma^{(k')})^2\right) } .
$$

This is essentially the regime classifier viewed as a Gaussian mixture
with the prior replaced by the AR(1) one-step distribution. Code:
`ENGINE.predict_oos`. The output is `MR_Prob[t]` and
`Danger_Regime_Prob[t]` on the trading day, computed without ever
looking at $S_{t+1}, S_{t+2}, \dots$.

A small detail used inside the backtester: the OOS posteriors are
**smoothed with a 3-bar rolling median** before they enter the strategy
rules (`mr_probs.rolling(3).median().bfill()`). This filters single-bar
regime flips that would otherwise cause spurious entries / exits, at
the cost of one bar of lag. See `BACKTESTER.run` lines 56–57.

### 2.4.6 Walk-forward orchestration

The entire engine is wrapped in `ENGINE.walk_forward`, which loops over
calendar days and, for each day $i \geq T_{\text{train}}$:

1. Builds `train_df` from the previous $T_{\text{train}}$ days.
2. Builds `test_df` from day $i$.
3. Skips the day if `len(train_df) ≤ W_\beta + W_z + 10` (insufficient
   bars for both the OLS and the z-score window to fully populate).
4. Re-fits the rolling cointegration and the MS-AR(1).
5. Calls `predict_oos(test_df, ...)` to populate the trading day's
   columns.
6. Appends to the concatenated OOS frame and records per-fold parameters
   (β, α, regime variances, regime means, transition probabilities,
   AR coefficients) for diagnostic plotting.

Output: a single `oos_results` DataFrame with one row per *bar* across
all trading days, plus a `param_tracker` DataFrame with one row per
*trading day* holding the fold's model parameters.

---

## 2.5 The four trading strategies

This is the section the rest of the report refers back to most often.

All four strategies act on the same OOS DataFrame produced by
`ENGINE.walk_forward`. The columns they read are:

- `Z_Score` — rolling z-score of the spread (definition in §2.3)
- `MR_Prob`, `Danger_Regime_Prob` — HMM posteriors (definition in §2.4)
- `Spread_Return` — no-look-ahead spread return (definition in §2.2)
- `HalfSpread_A_bps`, `HalfSpread_B_bps` — slippage components

Each strategy is a function that, at every bar, decides a position
$P_t \in \{-1,\, 0,\, +1\}$ representing "short the spread", "flat" or
"long the spread". The four strategies are then run through the same
PnL machinery in §2.6.

### 2.5.1 Buy & Hold — the benchmark

The do-nothing strategy. The position is always long the spread:

$$
P_t^{\text{BH}} \;=\; +1 \quad\text{for every bar.}
$$

It earns $\Delta S_t$ every bar and pays *no transaction costs* (in our
model — the implicit assumption is that a single entry was opened at
$t = 0$ and the position is never traded). Code:
`BACKTESTER.run`, lines 103–108.

The purpose of Buy & Hold is diagnostic. If $\sum_t \Delta S_t > 0$ for
a given month, the spread *drifted in our favour* and a trading
strategy needs to do better than that passive drift to add value. If
$\sum_t \Delta S_t < 0$ the spread drifted against us — Baseline /
AR / MS-AR need to either avoid that drift (by being flat much of the
time) or capture mean-reversion opportunities to outperform the
passive number.

### 2.5.2 Baseline — pure z-score pairs trade

The textbook z-score strategy. Two parameters:

- $z_q$ = "quiet entry threshold" (we use $z_q = 1.3$)
- $z_x$ = "exit threshold" (we use $z_x = 0.0$ — exit on cross of zero)

Position rules, applied bar-by-bar:

$$
P_t \;=\;
\begin{cases}
+1 & \text{if } P_{t-1} = 0 \text{ and } Z_t < -z_q \\
-1 & \text{if } P_{t-1} = 0 \text{ and } Z_t > +z_q \\
0  & \text{if } P_{t-1} = +1 \text{ and } Z_t \geq -z_x \\
0  & \text{if } P_{t-1} = -1 \text{ and } Z_t \leq +z_x \\
P_{t-1} & \text{otherwise.}
\end{cases}
$$

Baseline **ignores** `MR_Prob` and `Danger_Regime_Prob` entirely. It
always trades on z-score deviation, no matter what the HMM is saying.

### 2.5.3 AR — z-score plus a hard kill switch

Same entry/exit *direction* rules as Baseline, with one extra binary
condition: the **gate**

$$
G_t \;=\; \mathbf{1}\!\left\{ \text{MR\_Prob}_t \;\geq\; 1 - \delta \right\}
$$

where $\delta$ is the `danger_threshold` parameter (we use $\delta = 0.30$,
so the gate opens whenever MR probability is at least 0.70 — equivalently,
the gate closes whenever Danger probability exceeds 0.30).

Modified rule:

$$
P_t \;=\;
\begin{cases}
0 & \text{if } G_t = 0 \quad\text{(panic liquidation \&\ no new entries)} \\
\text{(same as Baseline)} & \text{if } G_t = 1
\end{cases}
$$

Two things to notice:

1. The kill is **binary**: either trade normally or sit in cash. No middle
   ground.
2. The kill **applies to existing positions too**, not just to new
   entries. If you're long the spread and the HMM suddenly flags
   Danger, you close out and go flat at the next bar.

Code: `_generate_positions` in `code/scripts/backtester.py` lines 18–20
implement the panic kill (`if curr != 0.0 and not signals_allowed[i]:
curr = 0.0`). The `signals_allowed` array for the AR strategy is the
gate $G_t$ defined above.

### 2.5.4 MS-AR — z-score plus a soft gearbox

Same entry/exit *direction* rules as Baseline, but the entry threshold
itself is **dynamic** and **probability-weighted**:

$$
z_q^{\text{eff}}(t) \;=\; \text{MR\_Prob}_t \cdot z_q
                       \;+\; \text{Danger\_Regime\_Prob}_t \cdot z_v
$$

with $z_v$ ("volatile entry threshold") set to $z_v = 2.5$ in the
multi-month runs.

Position rule:

$$
P_t \;=\;
\begin{cases}
+1 & \text{if } P_{t-1} = 0 \text{ and } Z_t < -z_q^{\text{eff}}(t) \\
-1 & \text{if } P_{t-1} = 0 \text{ and } Z_t > +z_q^{\text{eff}}(t) \\
0  & \text{if } P_{t-1} \neq 0 \text{ and } |Z_t| \leq z_x \\
P_{t-1} & \text{otherwise.}
\end{cases}
$$

There is **no gate**, no panic liquidation. Already-open positions are
held until $Z_t$ crosses $z_x$ (which we set to zero), regardless of
what the HMM says. Only *new entries* respond to the regime, by way of
the entry threshold widening.

Intuition by limit:

- **All-quiet limit** ($\text{MR\_Prob} \to 1$): $z_q^{\text{eff}} \to
  z_q = 1.3$. Strategy entries reduce to Baseline.
- **All-danger limit** ($\text{Danger\_Prob} \to 1$): $z_q^{\text{eff}}
  \to z_v = 2.5$. Strategy still trades, but only on much more extreme
  z-deviations. Trade frequency drops.
- **Mixed regime**: linear blend. Soft gating.

This is the **gearbox**: the strategy shifts between aggressive (narrow
band) and cautious (wide band) without ever fully disengaging.

### 2.5.5 Side-by-side mathematical summary

| Strategy | Always long? | Entry threshold $E_t$ | Gate $G_t$ | Reads HMM? |
|----------|:---:|---|---|:---:|
| **Buy & Hold** | yes | — | — | no |
| **Baseline**   | no  | $z_q$ (constant) | always 1 | no |
| **AR**         | no  | $z_q$ (constant) | $\mathbf 1\{\text{MR\_Prob}_t \geq 1-\delta\}$ | yes (binary) |
| **MS-AR**      | no  | $\text{MR\_Prob}_t \cdot z_q + \text{Danger\_Prob}_t \cdot z_v$ | always 1 | yes (continuous) |

A single position-rule template covers Baseline / AR / MS-AR:

$$
P_t \;=\;
\begin{cases}
0 & \text{if } G_t = 0 \\[2pt]
\mathrm{sign}(-Z_t) & \text{if } P_{t-1} = 0,\; G_t = 1,\; |Z_t| > E_t \\[2pt]
0 & \text{if } P_{t-1} \neq 0,\; |Z_t| \leq z_x \\[2pt]
P_{t-1} & \text{otherwise.}
\end{cases}
$$

Code reference: `code/scripts/backtester.py` lines 59–78 build the
appropriate $(\text{entry\_z\_arr}, \text{signals\_allowed})$ pair per
strategy and dispatches to `_generate_positions` (or `_positions_daily`
when `flatten_eod=True`).

### 2.5.6 What's different at a glance

The simplest way to see the difference between AR and MS-AR is to
imagine the HMM has just emitted $\text{Danger\_Prob}_t = 0.8$ (so
$\text{MR\_Prob}_t = 0.2$):

- **Baseline** doesn't notice. Continues to trade against $z_q = 1.3$.
- **AR** sees $\text{MR\_Prob}_t = 0.2 < 1 - \delta = 0.7$, so $G_t = 0$.
  Any open position is closed immediately. No new entries allowed.
- **MS-AR** computes $z_q^{\text{eff}}(t) = 0.2 \cdot 1.3 + 0.8 \cdot 2.5
  = 2.26$. Any open position is held. New entries require $|Z_t| > 2.26$
  instead of the Baseline's $> 1.3$.

The two HMM strategies are using the *same* posterior in qualitatively
different ways: AR converts it to a binary decision (trade / don't
trade), MS-AR converts it to a continuous threshold modifier.

---

## 2.6 PnL accounting

The strategies above produce a position series $P_t$. The PnL machinery
turns that into realised returns.

### 2.6.1 No look-ahead

Position is **applied with a one-bar lag**:

$$
\mathrm{Target}_t \;=\; P_{t-1} .
$$

This is the standard fix to prevent a strategy from earning the same
bar's return it used to make the decision. In code:
`self.data['Target_Baseline'] = pd.Series(pos_base, ...).shift(1).fillna(0)`.

### 2.6.2 Gross return

$$
r_t^{\text{gross}} \;=\; \mathrm{Target}_t \cdot \Delta S_t
$$

with $\Delta S_t$ already computed using $\beta_{t-1}$ (so the position
held into bar $t$ was opened with information available at $t-1$, and
earns the spread return at the hedge ratio that was known at the time
of opening — fully causal).

### 2.6.3 Trade indicator and costs

A trade is initiated whenever the target position changes:

$$
T_t \;=\; \mathbf 1\!\left\{ \mathrm{Target}_t \neq \mathrm{Target}_{t-1} \right\} .
$$

Two cost terms per trade:

- **Fixed fee.** $c_f \cdot T_t / 10^4$, with $c_f = 0.5$ bps. Per-trade
  flat commission, modelled as a fraction of notional.
- **Half-spread slippage.** $T_t \cdot (\text{HS}^A_t + \text{HS}^B_t) /
  10^4$. Crossing the spread on both legs costs the sum of the two
  half-spreads.

### 2.6.4 Net return and cumulative return

$$
r_t \;=\; r_t^{\text{gross}} \;-\;
          \frac{c_f}{10^4} T_t \;-\;
          \frac{\text{HS}^A_t + \text{HS}^B_t}{10^4} T_t
$$

$$
R_T \;=\; \sum_{t=1}^{T} r_t .
$$

We store $r_t$ as `Return_{Strat}`, $r_t^{\text{gross}}$ as
`Return_{Strat}_Gross`, and $R_T$ as `CumReturn_{Strat}` for every strategy.

### 2.6.5 End-of-day flattening

In all multi-month notebooks we set `flatten_eod=True`. This rewrites
the position series so that

$$
P_t \;\leftarrow\; 0 \quad\text{if } t \text{ is the last bar of its day.}
$$

Implementation (`_positions_daily`) runs `_generate_positions`
independently on each day's slice and then forces `pos[-1] = 0`. The
target series therefore goes flat at the day boundary, which:

- Eliminates overnight risk.
- Isolates days as independent experiments (one of the goals of the
  multi-month design).
- Increases trade count: every held position now incurs at least one
  forced round-trip per day.

The earlier two-year notebooks in this repo did *not* use
`flatten_eod=True`; the resulting overnight carry meant MS-AR could
accumulate hundreds of overnight position-flips per month, with all the
slippage that entails. The `flatten_eod` version of the experiment is
strictly cleaner.

---

## 2.7 Performance metrics

All metrics are computed by `TEARSHEET._calc_metrics`
(`code/scripts/tearsheet.py`).

### 2.7.1 Annualisation convention

We use a **tick-clock annualisation factor**

$$
F \;=\; 252 \times 24 \times 60 \;=\; 362{,}880,
$$

interpreted as "minutes of trading per year if every bar is a minute".
For 500-tick bars on a liquid pair this is roughly the bar rate during
active hours, so $F$ is a reasonable scaling for ratio statistics.

**Important caveat.** Annualising a bar-clock Sharpe with $F = 362{,}880$
gives a number that is **$\sqrt{F/252} \approx 38\times$ larger** than
the same strategy's daily-returns Sharpe. The cross-strategy ordering
within a month-pair is preserved. The absolute level is not directly
comparable to Sharpe ratios in finance literature (which are typically
daily). We report relative ranking and treat the absolute number as
presentational.

### 2.7.2 Return metrics

- **Total Return** $= \sum_t r_t$ (fraction). Reported as bps
  ($\times 10^4$).
- **Annualised Return** $= \bar r \cdot F$ where $\bar r$ is the mean
  bar return.

### 2.7.3 Risk metrics

- **Annualised Volatility** $= \hat\sigma_r \cdot \sqrt{F}$.
- **Max Drawdown** $= \min_t \!\left( R_t - \max_{s \leq t} R_s \right)$.
- **Max Drawdown Duration** = longest stretch of bars below the previous
  peak.
- **Ulcer Index** $= \sqrt{\,\mathbb E[(R_t - \max_{s \leq t} R_s)^2]\,}$.
  RMS of the running drawdown — penalises both deep *and* long
  drawdowns.
- **Value at Risk 95%** = empirical 5th percentile of bar returns.
- **Conditional VaR 95%** = mean of $r_t$ below that percentile.

### 2.7.4 Risk-adjusted metrics

- **Sharpe** $= \dfrac{\bar r}{\hat\sigma_r} \sqrt F$.
- **Sortino** $= \dfrac{\bar r}{\hat\sigma_{r,\text{downside}}} \sqrt F$
  where the downside std uses only $r_t < 0$.
- **Calmar** $= \dfrac{\text{AnnReturn}}{|\text{MaxDD}|}$.
- **Profit Factor** $= \dfrac{\sum_{r > 0} r}{\big|\sum_{r < 0} r\big|}$.
- **Payoff Ratio** $= \dfrac{|\bar r_+|}{|\bar r_-|}$ where $\bar r_+$ is
  the mean of positive bar returns and $\bar r_-$ the mean of negatives.
- **Tail Ratio** $= \dfrac{|P_{95}(r)|}{|P_5(r)|}$ — symmetry of tails.

### 2.7.5 Trading metrics

- **Number of Trades** $= \dfrac{1}{2} \sum_t T_t$ (round trips).
- **Win Rate** = fraction of *active* bars (where $r_t \neq 0$) with
  $r_t > 0$.
- **Market Exposure** = fraction of bars where $\mathrm{Target}_t \neq 0$.

### 2.7.6 Distributional metrics

- **Skewness** of $\{r_t\}$.
- **Kurtosis** of $\{r_t\}$ (excess kurtosis; fat-tailed returns score
  much higher than 3).

---

## 2.8 Summary

Section 2.5 is the section the report keeps coming back to. The four
strategies — Buy & Hold, Baseline, AR, MS-AR — share the same
cointegration spread, the same rolling z-score, and the same HMM
regime posteriors. They differ only in how they map those signals to
a position. That difference is the entire empirical content of the
project.
