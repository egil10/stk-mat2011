
import numpy as np
import pandas as pd

class BACKTESTER:
    """
    Simulates a mean-reversion pairs trading strategy using Z-Scores,
    regulated by a Markov-Switching volatility kill-switch.
    """

    def __init__(self, df):
        self.data = df.copy()

    def run(self, entry_z=2.0, exit_z=0.0, danger_threshold=0.5, fee_bps=0.5):

        # extract columns to numpy arrays for lightning fast iteration
        z_scores = self.data['Z_Score'].values
        danger_probs = self.data['Danger_Regime_Prob'].values
        spread_returns = self.data['Spread_Return'].values
        garch_vol = self.data['Forecasted_Vol'].values
        vol_scalar = garch_vol / np.nanmedian(garch_vol)
        ar_phi = self.data['AR_Phi'].values
        positions = np.zeros(len(self.data))
        current_pos = 0  

        for i in range(len(self.data)):
            
            # skip the warm up period 
            if np.isnan(z_scores[i]):
                positions[i] = 0
                continue

            in_danger = danger_probs[i] > danger_threshold

            # the HMM kill switch
            if in_danger:
                current_pos = 0  # exit to cash
            
            # normal trading logic
            else:
                if current_pos == 0:
                    if z_scores[i] < -entry_z:
                        current_pos = 1
                    elif z_scores[i] > entry_z:
                        current_pos = -1
                
                elif current_pos == 1 and z_scores[i] >= exit_z:
                    current_pos = 0
                
                elif current_pos == -1 and z_scores[i] <= -exit_z:
                    current_pos = 0

            positions[i] = current_pos

        self.data['Position'] = positions

        # shift positions by 1
        self.data['Target_Position'] = self.data['Position'].shift(1).fillna(0)

        # calculate transaction costs
        trades = self.data['Target_Position'].diff().abs().fillna(0)
        costs = trades * (fee_bps / 10000)

        # calculate final strategy returns
        self.data['Strategy_Return'] = (self.data['Target_Position'] * self.data['Spread_Return']) - costs
        self.data['Cum_Return'] = self.data['Strategy_Return'].cumsum()

        self._print_summary()

        return self.data

    def _print_summary(self):

        total_return_bps = self.data['Cum_Return'].iloc[-1] * 10000
        total_trades = self.data['Target_Position'].diff().abs().sum() / 2

        roll_max = self.data['Cum_Return'].cummax()
        drawdown = self.data['Cum_Return'] - roll_max
        max_dd_bps = drawdown.min() * 10000

        print("\n--- BACKTEST SUMMARY ---")
        print(f"Total Return:   {total_return_bps:.2f} Bps")
        print(f"Max Drawdown:   {max_dd_bps:.2f} Bps")
        print(f"Total Trades:   {int(total_trades)}")