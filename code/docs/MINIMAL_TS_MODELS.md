# Minimal scripts: what the comments mean

This note unpacks the short comments at the top of `code/scripts/arimax.py`, `arch1.py`, and `garch.py`. Those files are **toys**: they simulate data from a known model (**DGP** = data-generating process), then fit the matching model in software so you can see that estimation recovers something sensible.

---

## Words used in the code comments

| Term | Meaning |
|------|--------|
| **DGP** | The rules we use to *simulate* \(y_t\). We pick numbers (`p`, `b`, `w`, `a`, …) on purpose, then check whether the fitted model’s parameters are close. |
| **Fit** | Running maximum likelihood (or equivalent) in **statsmodels** / **arch** on the simulated series. “Fit” is the mirror image of the DGP: same family of model, unknown parameters estimated from data. |
| **`e_t` vs `z_t`** | Often `z_t` is standard normal noise; `e_t` is the *innovation* that actually enters the model (sometimes scaled by volatility). |

---

## `arimax.py`

### Comment 1: DGP

```text
y_t = p*y_{t-1} + b*x_t + e_t,  e_t ~ N(0, s^2);  fit adds intercept (trend="c").
```

- **`y_t`**: series we care about.  
- **`p`**: AR(1) coefficient on yesterday’s \(y\).  
- **`x_t`**: **exogenous** regressor (here independent standard normal draws). In a real application this could be temperature, policy indicator, etc.  
- **`e_t`**: Gaussian noise; in code `e = randn * 0.5` so \(s = 0.5\).  
- The simulated loop **does not** add a separate constant \(c\). The comment warns that the **fit** still uses `trend="c"`, so statsmodels estimates an **intercept** as well. If the true DGP had \(c=0\), the estimated constant should be near zero (up to noise).

### Comment 2: Fit

```text
ARIMA(1,0,0) + exog x_t  ==  minimal ARIMAX here.
```

- **ARIMA(1,0,0)** means AR order 1, **no differencing**, MA order 0: an AR(1) for \(y_t\).  
- **`exog`**: passing \(x_t\) makes this an **ARIMAX** setup: AR(1) in \(y\) *plus* linear effect of \(x_t\).  
- “Minimal” = one lag, one exogenous column, no seasonality—just enough to show the idea.

### First value `y[0] = e[0]`

The loop starts at \(t=1\) using \(y_0\). Setting \(y_0 = e_0\) avoids needing \(y_{-1}\). That’s a simple start-up; a longer burn-in would make the early terms closer to the stationary distribution, but for a long series it barely matters.

---

## `arch1.py`

### Comment 1: DGP (ARCH(1))

```text
h_t^2 = w + a*e_{t-1}^2,  e_t = h_t*z_t,  z_t ~ N(0,1).  Fit: mean="Zero", vol ARCH(1).
```

Interpretation:

1. **`z_t`**: independent standard normals.  
2. **`h_t`**: *conditional* standard deviation of \(e_t\) given past information.  
3. **`h_t^2`** (variance) depends on **yesterday’s squared shock** \(e_{t-1}^2\): large \(|e_{t-1}|\) tends to make today’s variance larger—that’s **volatility clustering**.  
4. **`w > 0`**, **`a ≥ 0`** (and in stationary ARCH(1), **`a < 1`**): here `a = 0.88`.  
5. **Fit**: `arch_model(..., mean="Zero", vol="ARCH", p=1)` means “no mean equation (we model \(e_t\) directly), ARCH(1) variance”.

### Initial `h2 = w / (1 - a)` and `e[0]`

For a stationary ARCH(1), the **unconditional** variance of \(e_t\) is \(\mathbb{E}[e_t^2] = \omega / (1-\alpha)\) in the usual notation \(\omega,\alpha\). The code uses `w,a` for those. Starting with that variance and drawing `e[0]` is a standard way to initialise the recursion so early draws are not arbitrarily tiny or huge.

### Comment 2: why the file is `arch1.py`, not `arch.py`

Python imports modules by name. A script named `arch.py` in the same folder can **shadow** the third-party package **`arch`** (`from arch import arch_model`). Then Python loads *your* file instead of the library and the script breaks. **`arch1.py`** avoids that name clash.

### Comment on `y = e * 100`

The **arch** package estimates \(\omega\) on the **scale of the data**. If \(e_t\) is very small, \(\omega\) is tiny and optimisation can be finicky. Multiplying the whole series by 100 scales variances by \(100^2\); fitted \(\omega\) is on that scaled scale. The **ratios** (e.g. \(\alpha\) as persistence) are what you usually care about for interpretation; levels of \(\omega\) must be read on the \(y\)-scale you pass in.

---

## `garch.py`

### Comment 1: DGP (GARCH(1,1))

```text
h_t^2 = w + a*e_{t-1}^2 + b*h_{t-1}^2,  e_t = h_t*z_t.  Fit: mean="Zero", vol GARCH(1,1).
```

Compared to ARCH(1), variance now depends on:

- **`a * e_{t-1}^2`**: shock yesterday (same idea as ARCH), and  
- **`b * h_{t-1}^2`**: *yesterday’s variance itself* (persistence).

Typical financial series: **`a` small, `b` large**—volatility dies slowly. Stationarity needs **`a + b < 1`** (here `0.08 + 0.90`).

### Comment 2: the loop and `h2`

The variable **`h2`** stores **current** conditional variance \(h_t^2\).

At each step the code does:

```python
h2 = w + a * e[t - 1] ** 2 + b * h2
```

The **`h2` on the right-hand side** is still the value from the **previous** iteration—that is \(h_{t-1}^2\). Then the assignment overwrites `h2` with the new \(h_t^2\). So one variable carries “yesterday’s variance” into “today’s variance” without needing a separate `h2_prev`.

### Initial `h2 = w / (1 - a - b)`

That is the **unconditional** variance of \(e_t\) under stationary GARCH(1,1). Same role as in ARCH: reasonable starting variance before the time loop.

### `y = e * 100`

Same scaling idea as in `arch1.py`: keeps reported `omega` away from extremely small numbers during optimisation.

---

## How this connects to the rest of the project

- These three scripts are **building blocks**: mean dynamics + exogenous variable (ARIMAX), variance depends only on past shocks (ARCH), variance depends on shocks *and* past variance (GARCH).  
- A **Markov-switching** extension (your main line) lets **`p`, `b`, or \((w,a,b)\)` change with a hidden state**. The comments in these minimal files describe the **within-regime** model; switching adds “which regime are we in?” on top.

---

## Quick sanity checklist when you read the printed output

1. **ARIMAX**: Do `ar.L1` and `x1` sit near the simulated `p` and `b`? Is `sigma2` near \(0.5^2 = 0.25\)?  
2. **ARCH / GARCH**: Is `alpha[1]` in the ballpark of simulated `a`? For GARCH, is `beta[1]` near simulated `b`? Remember **`omega` is on the scale of `y`** (including the ×100 if you keep that line).

If something is wildly off, check seed, sample length, or whether the simulated DGP and the `fit()` call describe the same model (e.g. accidentally fitting a constant mean when the DGP assumed zero mean).
