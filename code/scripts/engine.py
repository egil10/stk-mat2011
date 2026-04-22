
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import numpy as np

class ENGINE:
    
    def __init__(self, df):
        self.data = df.copy()
        self.beta = None
        self.alpha = None

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