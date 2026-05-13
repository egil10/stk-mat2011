
import warnings

import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from arch import arch_model

class ENGINE:
    def __init__(self, df):
        self.data = df.copy()
        self.beta = None
        self.alpha = None
        self.beta_series = None
        self.alpha_series = None
        self.danger_variance = None
        self.safe_variance = None
        self.ar_phi = None
        self.garch_params = None
        self.forecasted_vol = None
        # AR(1)-HMM per-regime parameters (MR/DR are short-hands for the
        # mean-revert and danger regime; full per-regime params live in the
        # _regime_* lists so OOS likelihoods can use every state).
        self.mr_mu = None
        self.mr_sigma = None
        self.mr_rho = None
        self.dr_mu = None
        self.dr_sigma = None
        self.dr_rho = None
        self.mr_const = None
        self.dr_const = None
        self.mr_idx = None
        self.k_regimes = None
        self._regime_consts = None
        self._regime_ars = None
        self._regime_sigmas = None

    @staticmethod
    def _rolling_ols(y, x, window, refit_every=1):
        n = len(y)
        beta, alpha = np.full(n, np.nan), np.full(n, np.nan)
        y_arr, x_arr = np.asarray(y, dtype=float), np.asarray(x, dtype=float)

        csum = lambda a: np.concatenate(([0.0], np.cumsum(a)))
        cx, cy = csum(x_arr), csum(y_arr)
        cxx, cxy = csum(x_arr * x_arr), csum(x_arr * y_arr)

        last_b, last_a = np.nan, np.nan
        for t in range(window - 1, n):
            if (t - (window - 1)) % refit_every != 0 and not np.isnan(last_b):
                beta[t], alpha[t] = last_b, last_a
                continue

            lo, hi, w = t - window + 1, t + 1, window
            sx, sy = cx[hi] - cx[lo], cy[hi] - cy[lo]
            sxx, sxy = cxx[hi] - cxx[lo], cxy[hi] - cxy[lo]

            denom = w * sxx - sx * sx
            if denom == 0 or not np.isfinite(denom): continue
            
            b = (w * sxy - sx * sy) / denom
            a = (sy - b * sx) / w
            beta[t], alpha[t] = b, a
            last_b, last_a = b, a

        return pd.Series(beta, index=y.index), pd.Series(alpha, index=y.index)

    def fit_cointegration(self, y_col='Log_A', x_col='Log_B', coint_window=2000, z_window=1000):
        Y, X = self.data[y_col], self.data[x_col]
        beta_s, alpha_s = self._rolling_ols(Y, X, coint_window)
        
        self.beta_series, self.alpha_series = beta_s, alpha_s
        self.data['Beta'], self.data['Alpha'] = beta_s, alpha_s

        beta_lag, alpha_lag = beta_s.shift(1), alpha_s.shift(1)
        self.data['Spread_Level'] = Y - (beta_s * X + alpha_s)

        # Rolling z-score kept for the Baseline strategy (no HMM)
        roll_mean = self.data['Spread_Level'].rolling(window=z_window).mean()
        roll_std = self.data['Spread_Level'].rolling(window=z_window).std()
        self.data['Z_Score'] = (self.data['Spread_Level'] - roll_mean) / roll_std
        self.data['Spread_Return'] = self.data['Return_A'] - beta_lag * self.data['Return_B']

        self.beta = float(beta_s.dropna().iloc[-1]) if beta_s.notna().any() else None
        self.alpha = float(alpha_s.dropna().iloc[-1]) if alpha_s.notna().any() else None
        return self.data.dropna(subset=['Spread_Level', 'Z_Score', 'Spread_Return'])

    def fit_markov_regimes(self, k_regimes=2, random_seed=42,
                           winsorize_std=None, scaling=None,
                           n_init=1, **kwargs):
        """
        AR(1)-HMM on Spread_Level per the pairs trading spec:
            z_t = mu^k + rho^k * (z_{t-1} - mu^k) + eps_t^k

        Regimes classified by sigma:
          - smallest sigma = mean-reverting / quiet (MR)
          - largest  sigma = danger / volatile        (DR)

        winsorize_std : clip the training spread to +/- winsorize_std * std before
            fitting the HMM. Stabilises the EM solver on fat-tailed series.
        scaling : multiply the training spread by `scaling` before fitting (a
            pure numerical-conditioning trick). Estimated parameters are
            unscaled before being stored, so downstream code sees them in the
            original Spread_Level units.
        """
        if random_seed is not None: np.random.seed(random_seed)

        spread_orig = self.data['Spread_Level'].dropna()

        # Pre-process for solver stability. These transforms are LINEAR, so the
        # regime structure (rho, transition matrix, regime probabilities) is
        # invariant; only the scale of const/sigma changes, which we undo below.
        spread_train = spread_orig.copy()
        if winsorize_std is not None and winsorize_std > 0:
            sigma_full = spread_train.std()
            if np.isfinite(sigma_full) and sigma_full > 0:
                hi = spread_train.median() + winsorize_std * sigma_full
                lo = spread_train.median() - winsorize_std * sigma_full
                spread_train = spread_train.clip(lower=lo, upper=hi)
        scale = float(scaling) if (scaling is not None and scaling > 0) else 1.0
        if scale != 1.0:
            spread_train = spread_train * scale

        # statsmodels emits a torrent of ValueWarning / ConvergenceWarning per
        # EM call.  They're not actionable here (we already winsorise + scale
        # for stability) and they swamp the notebook output when walk-forward
        # refits every trading day.  Suppress locally rather than globally.
        # n_init > 1 runs the EM from `n_init` random seeds and keeps the
        # highest-likelihood solution (paper §4.5 — EM converges to a local
        # max, multi-seed mitigates the worst seeds).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            spec = sm.tsa.MarkovAutoregression(
                spread_train,
                k_regimes=k_regimes,
                order=1,
                switching_ar=True,
                switching_trend=True,
                switching_variance=True,
            )
            if n_init <= 1:
                model = spec.fit(disp=False)
            else:
                rng = np.random.default_rng(random_seed or 0)
                base = np.asarray(spec.start_params, dtype=float)
                best = None
                for s in range(int(n_init)):
                    try:
                        if s == 0:
                            cand = spec.fit(disp=False)
                        else:
                            start = base + 0.1 * rng.standard_normal(base.size)
                            cand = spec.fit(disp=False, start_params=start)
                    except Exception:
                        continue
                    if best is None or cand.llf > best.llf:
                        best = cand
                if best is None:
                    best = spec.fit(disp=False)
                model = best

        # --- Extract per-regime parameters and undo the scale transform ---
        # If y_scaled = scale * y, then const_scaled = scale * const_orig and
        # sigma_scaled = scale * sigma_orig. AR coefficients are dimensionless.
        ar_coeffs = [float(model.params[f'ar.L1[{i}]']) for i in range(k_regimes)]
        variances = [float(model.params[f'sigma2[{i}]']) / (scale ** 2) for i in range(k_regimes)]
        consts    = [float(model.params[f'const[{i}]']) / scale for i in range(k_regimes)]

        # Unconditional mean: mu^k = const^k / (1 - rho^k)
        means  = [consts[i] / (1 - ar_coeffs[i]) if abs(ar_coeffs[i]) < 1 else np.nan
                   for i in range(k_regimes)]
        sigmas = [float(np.sqrt(v)) for v in variances]

        # --- Classify regimes by Volatility (sigma) ---
        # Quiet (MR) = lowest variance, Volatile (DR) = highest variance.
        mr_idx = int(np.argmin(sigmas))
        non_mr = [j for j in range(k_regimes) if j != mr_idx]
        dr_idx = int(non_mr[int(np.argmax([sigmas[j] for j in non_mr]))])

        # Store full per-regime parameter set so OOS likelihood can use every
        # state (matters for k_regimes >= 3, where the in-sample danger prob
        # is the SUM of all non-MR smoothed probabilities).
        self.k_regimes      = k_regimes
        self.mr_idx         = mr_idx
        self._regime_consts = consts
        self._regime_ars    = ar_coeffs
        self._regime_sigmas = sigmas

        # Short-hands for backward compatibility (param_tracker, tearsheet,
        # any downstream code that reads MR/DR fields directly).
        self.mr_mu, self.mr_sigma, self.mr_rho = means[mr_idx], sigmas[mr_idx], ar_coeffs[mr_idx]
        self.mr_const = consts[mr_idx]
        self.dr_mu, self.dr_sigma, self.dr_rho = means[dr_idx], sigmas[dr_idx], ar_coeffs[dr_idx]
        self.dr_const = consts[dr_idx]

        self.safe_variance   = variances[mr_idx]
        self.danger_variance = variances[dr_idx]
        self.safe_mean       = means[mr_idx]
        self.danger_mean     = means[dr_idx]
        self.ar_phi          = ar_coeffs[mr_idx]

        self.p_safe_safe     = model.params.get(f'p[{mr_idx}->{mr_idx}]', np.nan)
        self.p_danger_danger = model.params.get(f'p[{dr_idx}->{dr_idx}]', np.nan)

        # --- Regime probabilities (smoothed in-sample posteriors) ---
        mr_prob = model.smoothed_marginal_probabilities[mr_idx].reindex(self.data.index)

        if k_regimes > 2:
            non_mr_prob = sum(
                model.smoothed_marginal_probabilities[j].reindex(self.data.index)
                for j in range(k_regimes) if j != mr_idx
            )
            self.data['Danger_Regime_Prob'] = non_mr_prob
        else:
            self.data['Danger_Regime_Prob'] = 1.0 - mr_prob

        self.data['MR_Prob'] = mr_prob

        # Regime-conditional z-score uses the unconditional std of the MR AR(1):
        #   sigma_uncond = sigma / sqrt(1 - rho^2)
        self.mr_sigma_uncond = self.mr_sigma / np.sqrt(max(1 - self.mr_rho**2, 1e-6))
        self.data['Regime_Z'] = (self.data['Spread_Level'] - self.mr_mu) / self.mr_sigma_uncond

        return self.data

    def predict_oos(self, test_df, train_tail_df, z_window, coint_window, **kwargs):
        test_data = test_df.copy()
        
        combined_logA = pd.concat([train_tail_df['Log_A'], test_data['Log_A']])
        combined_logB = pd.concat([train_tail_df['Log_B'], test_data['Log_B']])
        combined_logA = combined_logA[~combined_logA.index.duplicated(keep='last')]
        combined_logB = combined_logB[~combined_logB.index.duplicated(keep='last')]

        beta_full, alpha_full = self._rolling_ols(combined_logA, combined_logB, coint_window)
        beta_test, alpha_test = beta_full.loc[test_data.index], alpha_full.loc[test_data.index]
        
        test_data['Beta'], test_data['Alpha'] = beta_test, alpha_test
        test_data['Spread_Level'] = test_data['Log_A'] - (beta_test * test_data['Log_B'] + alpha_test)

        beta_lag_test = beta_full.shift(1).loc[test_data.index]
        test_data['Spread_Return'] = test_data['Return_A'] - beta_lag_test * test_data['Return_B']

        # Rolling z-score for the Baseline strategy
        past_spread = train_tail_df['Spread_Level'].iloc[-z_window:]
        combined_spread = pd.concat([past_spread, test_data['Spread_Level']])
        combined_spread = combined_spread[~combined_spread.index.duplicated(keep='last')]
        roll_mean = combined_spread.rolling(window=z_window).mean()
        roll_std = combined_spread.rolling(window=z_window).std()
        test_data['Z_Score'] = ((combined_spread - roll_mean) / roll_std).loc[test_data.index]

        # --- Regime-conditional z-score (unconditional sigma) ---
        test_data['Regime_Z'] = (test_data['Spread_Level'] - self.mr_mu) / self.mr_sigma_uncond

        # --- OOS regime probability via per-regime AR(1) likelihoods ---
        # p(z_t | z_{t-1}, regime=k) = N(const^k + rho^k * z_{t-1}, sigma^k)
        # For k_regimes >= 3 we sum the likelihoods of all non-MR states so the
        # OOS Danger_Prob is computed the same way as the in-sample one.
        z_t   = test_data['Spread_Level'].values
        z_lag = np.roll(z_t, 1)
        z_lag[0] = train_tail_df['Spread_Level'].iloc[-1]

        if (self._regime_consts is not None
                and self._regime_ars is not None
                and self._regime_sigmas is not None
                and self.mr_idx is not None):
            consts_k = self._regime_consts
            ars_k    = self._regime_ars
            sigmas_k = self._regime_sigmas
            mr_idx   = self.mr_idx
        else:
            # Fallback for callers that bypassed fit_markov_regimes.
            consts_k = [self.mr_const, self.dr_const]
            ars_k    = [self.mr_rho,   self.dr_rho]
            sigmas_k = [self.mr_sigma, self.dr_sigma]
            mr_idx   = 0

        liks = []
        for c, rho, sig in zip(consts_k, ars_k, sigmas_k):
            mu_t = c + rho * z_lag
            liks.append(norm.pdf(z_t, loc=mu_t, scale=max(sig, 1e-12)))
        liks = np.vstack(liks)                 # shape (k, n)
        denom = liks.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            posterior = np.where(denom > 0, liks / denom, 1.0 / liks.shape[0])

        mr_prob = posterior[mr_idx]
        non_mr  = [j for j in range(liks.shape[0]) if j != mr_idx]
        danger_prob = posterior[non_mr].sum(axis=0) if non_mr else 1.0 - mr_prob

        test_data['MR_Prob'] = mr_prob
        test_data['Danger_Regime_Prob'] = danger_prob

        test_data['AR_Phi'] = self.ar_phi if self.ar_phi is not None else np.nan
        return test_data

    @classmethod
    def each_day(cls, df, coint_window, z_window, k_regimes=2,
                 winsorize_std=None, scaling=None, n_init=1,
                 min_bars=None, verbose=True):
        """
        Independent per-day fit. Each calendar day is taken in isolation:
        the rolling-OLS hedge ratio, the rolling z-score and the MS-AR(1)
        regime model are all fit on that day's bars only. Days that do
        not have enough bars are skipped.

        This is the "in-sample, day-by-day" mode used in code/strats/:
        the smoothed posteriors γ_t^MR returned by the HMM are kept as-is
        (no OOS forward filter) so the strategies operate on the model's
        best estimate of the day's regime structure.

        Returns the concatenated per-day DataFrame and a per-day param
        tracker (one row per fitted day).
        """
        df = df.copy()
        df['Date'] = df.index.date
        unique_days = df['Date'].unique()
        if min_bars is None:
            min_bars = coint_window + z_window + 10

        if verbose:
            winsor_str = "off" if winsorize_std in (None, 0) else f"{winsorize_std}σ"
            scale_str  = "off" if scaling       in (None, 0, 1) else f"x{scaling}"
            print(f"Each-day fit | coint_window={coint_window} | z_window={z_window} "
                  f"| k_regimes={k_regimes} | winsor={winsor_str} | scale={scale_str} "
                  f"| n_init={n_init} | days={len(unique_days)}")

        out_chunks, params = [], []
        for d in unique_days:
            day_df = df[df['Date'] == d].copy()
            if len(day_df) < min_bars:
                if verbose:
                    print(f"  {d}: skipped (only {len(day_df)} bars, need ≥{min_bars})")
                continue
            try:
                eng = cls(day_df)
                eng.fit_cointegration(coint_window=coint_window, z_window=z_window)
                eng.fit_markov_regimes(
                    k_regimes=k_regimes,
                    winsorize_std=winsorize_std,
                    scaling=scaling,
                    n_init=n_init,
                )
            except Exception as e:
                if verbose:
                    print(f"  {d}: HMM failed ({type(e).__name__}: {e})")
                continue
            out_chunks.append(eng.data)
            params.append({
                'Date': d,
                'Bars': len(day_df),
                'Beta': eng.beta, 'Alpha': eng.alpha,
                'Safe_Variance': eng.safe_variance, 'Danger_Variance': eng.danger_variance,
                'Safe_Mean': eng.safe_mean, 'Danger_Mean': eng.danger_mean,
                'P_Safe_Safe': eng.p_safe_safe, 'P_Danger_Danger': eng.p_danger_danger,
                'MR_Rho': eng.mr_rho, 'DR_Rho': eng.dr_rho,
            })
        if not out_chunks:
            raise RuntimeError("each_day: no day produced a valid fit.")
        return pd.concat(out_chunks), pd.DataFrame(params).set_index('Date')

    @classmethod
    def walk_forward(cls, df, train_days, coint_window, z_window, k_regimes=2,
                     winsorize_std=None, scaling=None, print_freq=10, verbose=True,
                     **kwargs):
        df = df.copy()
        df['Date'] = df.index.date
        unique_days = df['Date'].unique()

        if verbose:
            winsor_str = "off" if winsorize_std in (None, 0) else f"{winsorize_std}σ"
            scale_str  = "off" if scaling in (None, 0, 1) else f"x{scaling}"
            print(f"Running Engine | train_days={train_days} | coint_window={coint_window} "
                  f"| z_window={z_window} | k_regimes={k_regimes} | winsor={winsor_str} | scale={scale_str}")
        oos_results, param_tracker = [], []

        for i in range(train_days, len(unique_days)):
            train_df = df[df['Date'].isin(unique_days[i - train_days : i])].copy()
            test_df  = df[df['Date'] == unique_days[i]].copy()
            if len(train_df) <= coint_window + z_window + 10 or len(test_df) < 5: continue

            try:
                eng = cls(train_df)
                eng.fit_cointegration(coint_window=coint_window, z_window=z_window)
                eng.fit_markov_regimes(
                    k_regimes=k_regimes,
                    winsorize_std=winsorize_std,
                    scaling=scaling,
                )

                oos = eng.predict_oos(test_df, eng.data, z_window=z_window, coint_window=coint_window)
            except Exception as e:
                print(f"[{unique_days[i]}] skipped: {e}")
                continue

            oos_results.append(oos)
            param_tracker.append({
                'Date': unique_days[i],
                'Beta': eng.beta, 'Alpha': eng.alpha,
                'Safe_Variance': eng.safe_variance, 'Danger_Variance': eng.danger_variance,
                'Safe_Mean': eng.safe_mean, 'Danger_Mean': eng.danger_mean,
                'P_Safe_Safe': eng.p_safe_safe, 'P_Danger_Danger': eng.p_danger_danger,
                'AR_Phi': eng.ar_phi,
                'MR_Rho': eng.mr_rho, 'DR_Rho': eng.dr_rho,
            })

            if i % print_freq == 0:
                print(f"[{unique_days[i]}] Beta: {eng.beta:.4f} | MR_rho: {eng.mr_rho:.4f} | DR_rho: {eng.dr_rho:.4f}")

        assert oos_results, "No folds ran."
        return pd.concat(oos_results), pd.DataFrame(param_tracker).set_index('Date')