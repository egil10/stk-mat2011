
import pandas as pd
import matplotlib.pyplot as plt

class TEARSHEET:

    def __init__(self, df_results):
        self.df = df_results
        self.returns = df_results['Strategy_Return']

    def calculate_financials(self):
        # Convert index to a pandas Series to ensure .dt and .unique() work
        idx_series = pd.Series(self.df.index)
        
        # Calculate how many unique days are in the sample
        unique_days_count = len(idx_series.dt.date.unique())
        
        # Annualization factor: (Total Bars / Years in Sample)
        ann_factor = len(self.df) / (unique_days_count / 252)

        total_ret = self.returns.sum() * 10000
        vol = self.returns.std() * np.sqrt(ann_factor) * 10000
        
        # Sharpe Ratio
        sharpe = (self.returns.mean() * ann_factor) / (self.returns.std() * np.sqrt(ann_factor))

        # Drawdown
        cum_ret = self.returns.cumsum()
        running_max = cum_ret.cummax()
        drawdown = cum_ret - running_max
        max_dd = drawdown.min() * 10000

        print(f"{'--- STRATEGY METRICS ---':^30}")
        print(f"Total Return: {total_ret:>10.2f} Bps")
        print(f"Annual Vol:   {vol:>10.2f} Bps")
        print(f"Sharpe Ratio: {sharpe:>10.2f}")
        print(f"Max Drawdown: {max_dd:>10.2f} Bps")
        print("-" * 30)

    def plot_performance(self):

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # equity curve
        axes[0].plot(self.df.index, self.df['Strategy_Return'].cumsum() * 10000, color='blue')
        axes[0].set_title("Cumulative Returns (Bps)")
        axes[0].grid(True, alpha=0.3)

        # drawdown
        cum_ret = self.df['Strategy_Return'].cumsum()
        drawdown = (cum_ret - cum_ret.cummax()) * 10000
        axes[1].fill_between(self.df.index, drawdown, color='red', alpha=0.3)
        axes[1].set_title("Underwater Plot (Drawdown in Bps)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        