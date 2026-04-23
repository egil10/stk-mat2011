
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

        roll_mean = self.data['Spread_Level'].rolling(window=z_window).mean()
        roll_std = self.data['Spread_Level'].rolling(window=z_window).std()
        self.data['Z_Score'] = (self.data['Spread_Level'] - roll_mean) / roll_std
        self.data['Spread_Return'] = self.data['Return_A'] - beta_lag * self.data['Return_B']

        self.beta = float(beta_s.dropna().iloc[-1]) if beta_s.notna().any() else None
        self.alpha = float(alpha_s.dropna().iloc[-1]) if alpha_s.notna().any() else None
        return self.data.dropna(subset=['Spread_Level', 'Z_Score', 'Spread_Return'])

    def fit_ar_reversion(self, lags=1):
        model = sm.tsa.AutoReg(self.data['Spread_Level'].dropna(), lags=lags).fit()
        self.ar_phi = float(model.params.iloc[1])
        return self.ar_phi

    def fit_markov_regimes(self, k_regimes=2, scaling=10000, jitter_size=1e-5, random_seed=42):
        if random_seed is not None: np.random.seed(random_seed)
        jitter = np.random.normal(0, jitter_size, size=len(self.data))
        target = (self.data['Spread_Return'] * scaling) + jitter
        target = pd.Series(target, index=self.data.index).dropna()

        model = sm.tsa.MarkovRegression(target, k_regimes=k_regimes, switching_variance=True, trend='c').fit(disp=False)
        variances = [model.params[f'sigma2[{i}]'] for i in range(k_regimes)]
        
        danger_idx, safe_idx = int(np.argmax(variances)), int(np.argmin(variances))
        self.danger_variance, self.safe_variance = variances[danger_idx], variances[safe_idx]
        self.danger_mean = model.params.get(f'const[{danger_idx}]', 0.0)
        self.safe_mean = model.params.get(f'const[{safe_idx}]', 0.0)
        self.p_danger_danger = model.params.get(f'p[{danger_idx}->{danger_idx}]', np.nan)
        self.p_safe_safe = model.params.get(f'p[{safe_idx}->{safe_idx}]', np.nan)

        self.data['Danger_Regime_Prob'] = model.smoothed_marginal_probabilities[danger_idx].reindex(self.data.index)
        return self.data

    def predict_oos(self, test_df, train_tail_df, z_window, coint_window, scaling=10000):
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

        past_spread = train_tail_df['Spread_Level'].iloc[-z_window:]
        combined_spread = pd.concat([past_spread, test_data['Spread_Level']])
        combined_spread = combined_spread[~combined_spread.index.duplicated(keep='last')]
        roll_mean = combined_spread.rolling(window=z_window).mean()
        roll_std = combined_spread.rolling(window=z_window).std()
        test_data['Z_Score'] = ((combined_spread - roll_mean) / roll_std).loc[test_data.index]

        scaled_returns = test_data['Spread_Return'] * scaling
        prob_safe = norm.pdf(scaled_returns, loc=0, scale=np.sqrt(self.safe_variance))
        prob_danger = norm.pdf(scaled_returns, loc=0, scale=np.sqrt(self.danger_variance))
        denom = prob_safe + prob_danger
        test_data['Danger_Regime_Prob'] = np.where(denom > 0, prob_danger / denom, 0.5)

        test_data['AR_Phi'] = self.ar_phi if self.ar_phi is not None else np.nan
        return test_data

    @classmethod
    def walk_forward(cls, df, train_days, coint_window, z_window, scaling=10000, print_freq=10):
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
                # Parameters passed dynamically here
                eng.fit_cointegration(coint_window=coint_window, z_window=z_window)
                eng.fit_ar_reversion(lags=1)
                eng.fit_markov_regimes(k_regimes=2, scaling=scaling)
                
                # Parameters passed dynamically here
                oos = eng.predict_oos(test_df, eng.data, z_window=z_window, coint_window=coint_window, scaling=scaling)
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
            })

            if i % print_freq == 0:
                print(f"[{unique_days[i]}] Beta: {eng.beta:.4f} | AR: {eng.ar_phi:.4f}")

        assert oos_results, "No folds ran."
        return pd.concat(oos_results), pd.DataFrame(param_tracker).set_index('Date')