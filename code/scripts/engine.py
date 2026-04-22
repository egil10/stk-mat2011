
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from arch import arch_model # Need this for GARCH

class ENGINE:
    def __init__(self, df):
        self.data = df.copy()
        # Scalar (static) coint params — kept for backwards compat / reporting
        self.beta = None
        self.alpha = None
        # Rolling coint params (series, aligned with self.data.index)
        self.beta_series = None
        self.alpha_series = None
        self.coint_window = None
        # Other model outputs
        self.danger_variance = None
        self.safe_variance = None
        self.ar_phi = None
        self.garch_params = None
        self.forecasted_vol = None
 
    # ------------------------------------------------------------------
    # Static (old) fit — kept so you can compare rolling-vs-static
    # ------------------------------------------------------------------
    def fit_cointegration_static(self, y_col='Log_A', x_col='Log_B', z_window=1000):
        Y = self.data[y_col]
        X = sm.add_constant(self.data[x_col])
        ols_model = sm.OLS(Y, X).fit()
        self.beta = float(ols_model.params[x_col])
        self.alpha = float(ols_model.params['const'])
 
        self.data['Spread_Level'] = Y - (self.beta * self.data[x_col] + self.alpha)
        roll_mean = self.data['Spread_Level'].rolling(window=z_window).mean()
        roll_std = self.data['Spread_Level'].rolling(window=z_window).std()
        self.data['Z_Score'] = (self.data['Spread_Level'] - roll_mean) / roll_std
        self.data['Spread_Return'] = self.data['Return_A'] - self.beta * self.data['Return_B']
 
        print(f"[STATIC] Cointegration Fitted | Beta: {self.beta:.4f} | Alpha: {self.alpha:.4f}")
        return self.data.dropna()
 
    # ------------------------------------------------------------------
    # Rolling fit — the new default
    # ------------------------------------------------------------------
    @staticmethod
    def _rolling_ols(y, x, window, refit_every=1):
        """
        Returns (beta_series, alpha_series) from rolling OLS of y on x over
        `window` bars. Uses closed-form (X'X)^-1 X'y — vectorized, fast.
 
        For `refit_every > 1`, fits only every k-th bar and forward-fills
        between refits (useful if you want to save compute).
        """
        n = len(y)
        beta = np.full(n, np.nan)
        alpha = np.full(n, np.nan)
        y_arr = np.asarray(y, dtype=float)
        x_arr = np.asarray(x, dtype=float)
 
        # Precompute cumulative sums for O(1) rolling moments
        # We need E[x], E[y], E[x^2], E[xy] over the window.
        csum = lambda a: np.concatenate(([0.0], np.cumsum(a)))
        cx = csum(x_arr)
        cy = csum(y_arr)
        cxx = csum(x_arr * x_arr)
        cxy = csum(x_arr * y_arr)
 
        last_b, last_a = np.nan, np.nan
        for t in range(window - 1, n):
            if (t - (window - 1)) % refit_every != 0 and not np.isnan(last_b):
                beta[t] = last_b
                alpha[t] = last_a
                continue
 
            lo = t - window + 1
            hi = t + 1  # exclusive
            w = window
            sx = cx[hi] - cx[lo]
            sy = cy[hi] - cy[lo]
            sxx = cxx[hi] - cxx[lo]
            sxy = cxy[hi] - cxy[lo]
 
            denom = w * sxx - sx * sx
            if denom == 0 or not np.isfinite(denom):
                continue
            b = (w * sxy - sx * sy) / denom
            a = (sy - b * sx) / w
            beta[t] = b
            alpha[t] = a
            last_b, last_a = b, a
 
        return pd.Series(beta, index=y.index), pd.Series(alpha, index=y.index)
 
    def fit_cointegration(self, y_col='Log_A', x_col='Log_B',
                          coint_window=2000, z_window=1000, refit_every=1):
        """
        Rolling cointegration fit.
 
        Parameters
        ----------
        coint_window : int
            Lookback bars for the rolling OLS of log_A on log_B.
            Rule of thumb: several multiples of the spread's half-life.
        z_window : int
            Lookback for the rolling z-score of the spread.
        refit_every : int
            Refit frequency in bars. 1 = refit every bar (cheap, default).
 
        Notes
        -----
        Beta/alpha at bar t are estimated from data up to AND INCLUDING t.
        Before using them to compute `Spread_Return[t]` (which depends on
        Return_A[t], Return_B[t]), we shift by 1 bar to avoid look-ahead.
        The spread LEVEL at t, however, uses beta[t] applied to log prices
        at t — this is a contemporaneous residual, not a forecast, and is
        used only for z-score construction. The z-score is then shifted
        implicitly by the signal logic (BACKTESTER already does shift(1)).
        """
        self.coint_window = coint_window
        Y = self.data[y_col]
        X = self.data[x_col]
 
        beta_s, alpha_s = self._rolling_ols(Y, X, coint_window, refit_every=refit_every)
        self.beta_series = beta_s
        self.alpha_series = alpha_s
 
        # Expose contemporaneous beta/alpha on self.data for inspection/plots
        self.data['Beta'] = beta_s
        self.data['Alpha'] = alpha_s
 
        # Lagged versions for return calc (beta known BEFORE the return arrives)
        beta_lag = beta_s.shift(1)
        alpha_lag = alpha_s.shift(1)
 
        # Spread LEVEL uses contemporaneous beta — this is a residual, not a trade
        self.data['Spread_Level'] = Y - (beta_s * X + alpha_s)
 
        # Rolling z-score
        roll_mean = self.data['Spread_Level'].rolling(window=z_window).mean()
        roll_std = self.data['Spread_Level'].rolling(window=z_window).std()
        self.data['Z_Score'] = (self.data['Spread_Level'] - roll_mean) / roll_std
 
        # Spread RETURN uses lagged beta to avoid look-ahead
        self.data['Spread_Return'] = (
            self.data['Return_A'] - beta_lag * self.data['Return_B']
        )
 
        # Keep last rolling values in the scalar attributes too, so existing
        # code paths that read self.beta / self.alpha still work.
        self.beta = float(beta_s.dropna().iloc[-1]) if beta_s.notna().any() else None
        self.alpha = float(alpha_s.dropna().iloc[-1]) if alpha_s.notna().any() else None
 
        valid = self.data.dropna(subset=['Spread_Level', 'Z_Score', 'Spread_Return'])
        print(f"[ROLLING] Cointegration Fitted over {coint_window}-bar window | "
              f"Last Beta: {self.beta:.4f} | Last Alpha: {self.alpha:.4f} | "
              f"Valid rows: {len(valid)}")
        return valid
 
    # ------------------------------------------------------------------
    # AR, GARCH, Markov — unchanged logic, just work on the new spread
    # ------------------------------------------------------------------
    def fit_ar_reversion(self, lags=1):
        model = sm.tsa.AutoReg(self.data['Spread_Level'].dropna(), lags=lags).fit()
        self.ar_phi = float(model.params.iloc[1])
        return self.ar_phi
 
    def fit_garch_vol(self, scaling=10000):
        series = (self.data['Spread_Return'] * scaling).dropna()
        model = arch_model(series, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
        res = model.fit(disp='off')
        self.garch_params = res.params
        forecast = res.forecast(horizon=1)
        self.forecasted_vol = float(np.sqrt(forecast.variance.values[-1, 0]))
        return self.forecasted_vol
 
    def fit_markov_regimes(self, k_regimes=2, scaling=10000,
                           jitter_size=1e-5, random_seed=42):
        if random_seed is not None:
            np.random.seed(random_seed)
        jitter = np.random.normal(0, jitter_size, size=len(self.data))
        target = (self.data['Spread_Return'] * scaling) + jitter
        target = pd.Series(target, index=self.data.index).dropna()
 
        model = sm.tsa.MarkovRegression(
            target, k_regimes=k_regimes,
            switching_variance=True, trend='c',
        ).fit(disp=False)
 
        variances = [model.params[f'sigma2[{i}]'] for i in range(k_regimes)]
        danger_idx = int(np.argmax(variances))
        safe_idx = int(np.argmin(variances))
        self.danger_variance = variances[danger_idx]
        self.safe_variance = variances[safe_idx]
        self.data['Danger_Regime_Prob'] = (
            model.smoothed_marginal_probabilities[danger_idx].reindex(self.data.index)
        )
        print(f"Markov Fitted (k={k_regimes}) | Danger Var: {self.danger_variance:.2f} "
              f"| Safe Var: {self.safe_variance:.2f}")
        return self.data
 
    # ------------------------------------------------------------------
    # OOS projection with rolling cointegration
    # ------------------------------------------------------------------
    def predict_oos(self, test_df, train_tail_df,
                    z_window=1000, scaling=10000,
                    refit_every=500, garch_window=5000, ar_window=2000,
                    coint_window=None, coint_refit_every=1):
        """
        Rolling-cointegration OOS projection.
 
        The rolling OLS is continued across the train/test boundary using the
        tail of training data as seed, so the first OOS bar already has a
        fully-formed beta estimate. Beta at each OOS bar uses ONLY data up to
        (and including) that bar; the spread RETURN uses the 1-bar-lagged beta.
        """
        assert train_tail_df.index.max() < test_df.index.min(), \
            "train_tail_df must end strictly before test_df begins"
 
        if coint_window is None:
            coint_window = self.coint_window or 2000
 
        test_data = test_df.copy()
 
        # --- Combined log series across train tail + test, for rolling OLS ---
        combined_logA = pd.concat([train_tail_df['Log_A'], test_data['Log_A']])
        combined_logB = pd.concat([train_tail_df['Log_B'], test_data['Log_B']])
        # Drop any accidental duplicate indices
        combined_logA = combined_logA[~combined_logA.index.duplicated(keep='last')]
        combined_logB = combined_logB[~combined_logB.index.duplicated(keep='last')]
 
        beta_full, alpha_full = self._rolling_ols(
            combined_logA, combined_logB, coint_window, refit_every=coint_refit_every
        )
        beta_test = beta_full.loc[test_data.index]
        alpha_test = alpha_full.loc[test_data.index]
        test_data['Beta'] = beta_test
        test_data['Alpha'] = alpha_test
 
        # --- Spread level (contemporaneous) and returns (lagged beta) ---
        test_data['Spread_Level'] = test_data['Log_A'] - (beta_test * test_data['Log_B'] + alpha_test)
 
        # For the RETURN we need beta at t-1. To get a sensible value for the
        # very first OOS bar, pull beta from the last bar of the train tail.
        beta_lag_test = beta_full.shift(1).loc[test_data.index]
        test_data['Spread_Return'] = test_data['Return_A'] - beta_lag_test * test_data['Return_B']
 
        # --- Rolling z-score seeded with the train tail spread ---
        past_spread = train_tail_df['Spread_Level'].iloc[-z_window:]
        combined_spread = pd.concat([past_spread, test_data['Spread_Level']])
        combined_spread = combined_spread[~combined_spread.index.duplicated(keep='last')]
        roll_mean = combined_spread.rolling(window=z_window).mean()
        roll_std = combined_spread.rolling(window=z_window).std()
        test_data['Z_Score'] = ((combined_spread - roll_mean) / roll_std).loc[test_data.index]
 
        # --- OOS HMM classification (unchanged) ---
        scaled_returns = test_data['Spread_Return'] * scaling
        prob_safe = norm.pdf(scaled_returns, loc=0, scale=np.sqrt(self.safe_variance))
        prob_danger = norm.pdf(scaled_returns, loc=0, scale=np.sqrt(self.danger_variance))
        denom = prob_safe + prob_danger
        test_data['Danger_Regime_Prob'] = np.where(denom > 0, prob_danger / denom, 0.5)
 
        # --- Rolling GARCH & AR on spread returns / levels ---
        train_sr = train_tail_df['Spread_Return'].dropna()
        test_sr = test_data['Spread_Return'].dropna()
        full_sr = pd.concat([train_sr, test_sr])
        full_sr = full_sr[~full_sr.index.duplicated(keep='last')]
        full_sl = pd.concat([train_tail_df['Spread_Level'].dropna(),
                             test_data['Spread_Level'].dropna()])
        full_sl = full_sl[~full_sl.index.duplicated(keep='last')]
 
        garch_vol_out = np.full(len(test_data), np.nan)
        ar_phi_out = np.full(len(test_data), np.nan)
        last_garch_vol = self.forecasted_vol if self.forecasted_vol is not None else np.nan
        last_ar_phi = self.ar_phi if self.ar_phi is not None else np.nan
 
        for i, ts in enumerate(test_data.index):
            if i % refit_every == 0:
                sr_slice = full_sr.loc[full_sr.index < ts].iloc[-garch_window:]
                if len(sr_slice) > 100:
                    try:
                        m = arch_model(sr_slice * scaling, vol='Garch', p=1, q=1,
                                       dist='Normal', rescale=False).fit(disp='off')
                        fc = m.forecast(horizon=1)
                        last_garch_vol = float(np.sqrt(fc.variance.values[-1, 0]))
                    except Exception:
                        pass
 
                sl_slice = full_sl.loc[full_sl.index < ts].iloc[-ar_window:]
                if len(sl_slice) > 50:
                    try:
                        ar_m = sm.tsa.AutoReg(sl_slice, lags=1).fit()
                        last_ar_phi = float(ar_m.params.iloc[1])
                    except Exception:
                        pass
 
            garch_vol_out[i] = last_garch_vol
            ar_phi_out[i] = last_ar_phi
 
        test_data['GARCH_Vol'] = garch_vol_out
        test_data['AR_Phi'] = ar_phi_out
        return test_data


    @classmethod
    def walk_forward(cls, df, train_days=30, z_window=250, coint_window=None,
                    k_regimes=2, scaling=10000, print_freq=10, verbose=True):
        """
        Runs the full walk-forward loop: one engine per test day, fit on the
        preceding `train_days` days, project OOS onto the test day.

        Returns
        -------
        live_trading_data : pd.DataFrame   stitched OOS predictions
        df_params         : pd.DataFrame   per-fold scalar parameters
        """
        df = df.copy()
        df['Date'] = df.index.date
        unique_days = df['Date'].unique()

        # Auto-size coint_window if not given
        if coint_window is None:
            sample = df[df['Date'].isin(unique_days[:train_days])]
            coint_window = min(2000, int(0.4 * len(sample)))
        if verbose:
            print(f"train_days={train_days} | coint_window={coint_window} | z_window={z_window}")

        oos_results, param_tracker = [], []

        for i in range(train_days, len(unique_days)):
            train_df = df[df['Date'].isin(unique_days[i - train_days : i])].copy()
            test_df  = df[df['Date'] == unique_days[i]].copy()
            if len(train_df) <= coint_window + z_window + 10 or len(test_df) < 5:
                continue

            try:
                eng = cls(train_df)
                eng.fit_cointegration(coint_window=coint_window, z_window=z_window)
                eng.fit_ar_reversion(lags=1)
                eng.fit_garch_vol(scaling=scaling)
                eng.fit_markov_regimes(k_regimes=k_regimes, scaling=scaling)
                oos = eng.predict_oos(test_df, eng.data,
                                    z_window=z_window, coint_window=coint_window)
            except Exception as e:
                if verbose:
                    print(f"[{unique_days[i]}] skipped: {type(e).__name__}: {e}")
                continue

            oos_results.append(oos)
            param_tracker.append({
                'Date': unique_days[i],
                'Beta': eng.beta, 'Alpha': eng.alpha,
                'Safe_Variance': eng.safe_variance,
                'Danger_Variance': eng.danger_variance,
                'GARCH_Vol': eng.forecasted_vol,
                'AR_Phi': eng.ar_phi,
            })

            if verbose and i % print_freq == 0:
                print(f"[{unique_days[i]}] Beta: {eng.beta:.4f} | "
                    f"GARCH: {eng.forecasted_vol:.2f} | AR: {eng.ar_phi:.4f}")

        assert oos_results, "No folds ran — lower coint_window or train_days."
        live_trading_data = pd.concat(oos_results)
        df_params = pd.DataFrame(param_tracker).set_index('Date')
        if verbose:
            print(f"\nOOS rows: {len(live_trading_data)} | Folds: {len(oos_results)}")
        return live_trading_data, df_params