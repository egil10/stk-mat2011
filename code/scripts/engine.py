
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
        # AR(1)-HMM per-regime parameters
        self.mr_mu = None
        self.mr_sigma = None
        self.mr_rho = None
        self.dr_mu = None
        self.dr_sigma = None
        self.dr_rho = None
        self.mr_const = None
        self.dr_const = None

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

    def fit_markov_regimes(self, k_regimes=2, random_seed=42, **kwargs):
        """
        AR(1)-HMM on Spread_Level per the pairs trading spec:
            z_t = mu^k + rho^k * (z_{t-1} - mu^k) + eps_t^k

        Regimes classified by |rho^k|:
          - smallest |rho| = mean-reverting (MR)
          - largest  |rho| = drifting       (DR)
        """
        if random_seed is not None: np.random.seed(random_seed)

        spread = self.data['Spread_Level'].dropna()

        model = sm.tsa.MarkovAutoregression(
            spread,
            k_regimes=k_regimes,
            order=1,
            switching_ar=True,
            switching_trend=True,
            switching_variance=True,
        ).fit(disp=False)

        # --- Extract per-regime parameters ---
        ar_coeffs = [model.params[f'ar.L1[{i}]'] for i in range(k_regimes)]
        variances = [model.params[f'sigma2[{i}]'] for i in range(k_regimes)]
        consts    = [model.params[f'const[{i}]'] for i in range(k_regimes)]

        # Unconditional mean: mu^k = const^k / (1 - rho^k)
        means  = [consts[i] / (1 - ar_coeffs[i]) if abs(ar_coeffs[i]) < 1 else np.nan
                   for i in range(k_regimes)]
        sigmas = [np.sqrt(v) for v in variances]

        # --- Classify regimes by mean-reversion speed |rho^k| ---
        abs_rho = [abs(r) for r in ar_coeffs]
        mr_idx = int(np.argmin(abs_rho))  # fastest mean-reversion

        # For k=2 the other is drifting; for k=3 pick the slowest
        non_mr = [j for j in range(k_regimes) if j != mr_idx]
        dr_idx = int(non_mr[np.argmax([abs_rho[j] for j in non_mr])])

        # Store per-regime parameters
        self.mr_mu, self.mr_sigma, self.mr_rho = means[mr_idx], sigmas[mr_idx], ar_coeffs[mr_idx]
        self.mr_const = consts[mr_idx]
        self.dr_mu, self.dr_sigma, self.dr_rho = means[dr_idx], sigmas[dr_idx], ar_coeffs[dr_idx]
        self.dr_const = consts[dr_idx]

        # Backward-compat fields used by param_tracker / tearsheet
        self.safe_variance   = variances[mr_idx]
        self.danger_variance = variances[dr_idx]
        self.safe_mean       = means[mr_idx]
        self.danger_mean     = means[dr_idx]
        self.ar_phi          = ar_coeffs[mr_idx]  # rho of MR regime

        self.p_safe_safe     = model.params.get(f'p[{mr_idx}->{mr_idx}]', np.nan)
        self.p_danger_danger = model.params.get(f'p[{dr_idx}->{dr_idx}]', np.nan)

        # --- Regime probabilities ---
        mr_prob = model.smoothed_marginal_probabilities[mr_idx].reindex(self.data.index)

        # For k>=3, sum all non-MR regimes into "danger"
        if k_regimes > 2:
            non_mr_prob = sum(
                model.smoothed_marginal_probabilities[j].reindex(self.data.index)
                for j in range(k_regimes) if j != mr_idx
            )
            self.data['Danger_Regime_Prob'] = non_mr_prob
        else:
            self.data['Danger_Regime_Prob'] = 1.0 - mr_prob

        self.data['MR_Prob'] = mr_prob

        # --- Regime-conditional z-score: (z_t - mu^MR) / sigma^MR ---
        self.data['Regime_Z'] = (self.data['Spread_Level'] - self.mr_mu) / self.mr_sigma

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

        # --- Regime-conditional z-score ---
        test_data['Regime_Z'] = (test_data['Spread_Level'] - self.mr_mu) / self.mr_sigma

        # --- OOS regime probability via AR(1) likelihoods ---
        # p(z_t | z_{t-1}, regime=k) = N(const^k + rho^k * z_{t-1}, sigma^k)
        z_t   = test_data['Spread_Level'].values
        z_lag = np.roll(z_t, 1)
        z_lag[0] = train_tail_df['Spread_Level'].iloc[-1]

        cond_mean_mr = self.mr_const + self.mr_rho * z_lag
        cond_mean_dr = self.dr_const + self.dr_rho * z_lag

        ll_mr = norm.pdf(z_t, loc=cond_mean_mr, scale=self.mr_sigma)
        ll_dr = norm.pdf(z_t, loc=cond_mean_dr, scale=self.dr_sigma)
        denom = ll_mr + ll_dr

        mr_prob = np.where(denom > 0, ll_mr / denom, 0.5)
        test_data['MR_Prob'] = mr_prob
        test_data['Danger_Regime_Prob'] = 1.0 - mr_prob

        test_data['AR_Phi'] = self.ar_phi if self.ar_phi is not None else np.nan
        return test_data

    @classmethod
    def walk_forward(cls, df, train_days, coint_window, z_window, k_regimes=2, print_freq=10, **kwargs):
        df = df.copy()
        df['Date'] = df.index.date
        unique_days = df['Date'].unique()

        print(f"Running Engine | train_days={train_days} | coint_window={coint_window} | z_window={z_window}")
        oos_results, param_tracker = [], []

        for i in range(train_days, len(unique_days)):
            train_df = df[df['Date'].isin(unique_days[i - train_days : i])].copy()
            test_df  = df[df['Date'] == unique_days[i]].copy()
            if len(train_df) <= coint_window + z_window + 10 or len(test_df) < 5: continue

            try:
                eng = cls(train_df)
                eng.fit_cointegration(coint_window=coint_window, z_window=z_window)
                eng.fit_markov_regimes(k_regimes=k_regimes)
                
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