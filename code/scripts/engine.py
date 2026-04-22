
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm

class ENGINE:
    
    def __init__(self, df):

        self.data = df.copy()
        self.beta = None
        self.alpha = None

        self.danger_variance = None
        self.safe_variance = None

    def fit_cointegration(self, y_col='Log_A', x_col='Log_B', z_window=1000):

        # engle-granger ols
        Y = self.data[y_col]
        X = sm.add_constant(self.data[x_col])
        ols_model = sm.OLS(Y, X).fit()

        self.beta = ols_model.params[x_col]
        self.alpha = ols_model.params['const']

        # spread and z-score
        self.data['Spread_Level'] = Y - (self.beta * self.data[x_col] + self.alpha)

        roll_mean = self.data['Spread_Level'].rolling(window=z_window).mean()
        roll_std = self.data['Spread_Level'].rolling(window=z_window).std()
        self.data['Z_Score'] = (self.data['Spread_Level'] - roll_mean) / roll_std

        # calculate strategy returns 
        self.data['Spread_Return'] = self.data['Return_A'] - self.beta * self.data['Return_B']

        print(f"Cointegration Fitted | Beta: {self.beta:.4f} | Alpha: {self.alpha:.4f}")

        return self.data.dropna()

    def fit_markov_regimes(self, k_regimes=2, scaling=10000, jitter_size=1e-5, random_seed=42):

        # scale to basis points and add jitter
        if random_seed is not None:
            np.random.seed(random_seed)

        jitter = np.random.normal(0, jitter_size, size=len(self.data))
        target_series = (self.data['Spread_Return'] * scaling) + jitter

        model = sm.tsa.MarkovRegression(
            target_series,
            k_regimes=2,
            switching_variance=True,
            trend='c',
        ).fit(disp=False)

        variances = [model.params[f'sigma2[{i}]'] for i in range(k_regimes)]
        danger_idx = np.argmax(variances)

        self.data['Danger_Regime_Prob'] = model.smoothed_marginal_probabilities[danger_idx]

        print(f"Markov Fitted | Danger Variance: {np.max(variances):.2f} | Safe Variance: {np.min(variances):.2f}")
        
        return self.data 

    def predict_oss(self, test_df, train_tail_df, z_window=1000, scaling=10000:
        """
        Projects frozen train parameters onto strictly unseen test data
        """

        test_data = test_df.copy()

        # calculate OOS spread using frozen beta and alpha
        Y_test = test_data['Log_A']
        X_test = test_data['Log_B']
        test_spread = Y_test - (self.beta * X_test + self.alpha)
        test_data['Spread_Level'] = test_spread

        # OOS z-score 
        past_spread = train_tail_df['Spread_Level'].iloc[-z_window:]
        combined_spread = pd.concat([past_spread, test_spread])

        roll_mean = combined_spread.rolling(window=z_window).mean()
        roll_std = combined_spread.rolling(window=z_window).std()

        test_data['Z_Score'] = ((combined_spread - roll_mean) / roll_std).loc[test_data.index]
        test_data['Spread_Return'] = test_data['Return_A'] - self.beta * test_data['Return_B']

        # oos hmm classification (pdf evaluation)
        scaled_returns = test_data['Spread_Return'] * scaling

        prob_safe = norm.pdf(scaled_returns, loc=0, scale=np.sqrt(self.safe_variance))
        prob_danger = norm.pdf(scaled_returns, loc=0, scale=np.sqrt(self.danger_variance))

        # normalize to get a 0 to 1 probability
        test_data['Danger_Regime_Prob'] = prob_danger / (prob_safe + prob_danger)

        return test_data

