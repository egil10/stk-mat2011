
import numpy as np
import pandas as pd

class BACKTESTER:
    def __init__(self, df):
        self.data = df.copy()

    def run(self, base_z=2.0, exit_z=0.0, danger_threshold=0.6, fee_bps=0.5, ar_limit=0.995):
        z_scores = self.data['Z_Score'].values
        danger_probs = self.data['Danger_Regime_Prob'].values
        garch_vol = self.data['GARCH_Vol'].values
        ar_phi = self.data['AR_Phi'].values
        
        vol_scalar = garch_vol / np.nanmedian(garch_vol)
        
        # Track both strategies
        pos_adaptive = np.zeros(len(self.data))
        pos_baseline = np.zeros(len(self.data))
        curr_adapt = 0  
        curr_base = 0

        for i in range(len(self.data)):
            if np.isnan(z_scores[i]) or np.isnan(vol_scalar[i]):
                continue

            # --- 1. STATIC BASELINE LOGIC ---
            if curr_base == 0:
                if z_scores[i] < -base_z: curr_base = 1
                elif z_scores[i] > base_z: curr_base = -1
            elif curr_base == 1 and z_scores[i] >= exit_z: curr_base = 0
            elif curr_base == -1 and z_scores[i] <= -exit_z: curr_base = 0
            
            pos_baseline[i] = curr_base

            # --- 2. ADAPTIVE LOGIC ---
            in_danger = danger_probs[i] > danger_threshold
            trending = ar_phi[i] > ar_limit 

            if in_danger or trending:
                curr_adapt = 0  
            else:
                dynamic_entry = base_z * vol_scalar[i]
                if curr_adapt == 0:
                    if z_scores[i] < -dynamic_entry: curr_adapt = 1
                    elif z_scores[i] > dynamic_entry: curr_adapt = -1
                elif curr_adapt == 1 and z_scores[i] >= exit_z: curr_adapt = 0
                elif curr_adapt == -1 and z_scores[i] <= -exit_z: curr_adapt = 0
            
            pos_adaptive[i] = curr_adapt

        # Save Target Positions
        self.data['Target_Adaptive'] = pd.Series(pos_adaptive, index=self.data.index).shift(1).fillna(0)
        self.data['Target_Baseline'] = pd.Series(pos_baseline, index=self.data.index).shift(1).fillna(0)

        # Calculate Transaction Costs & Returns
        for strat in ['Adaptive', 'Baseline']:
            trades = self.data[f'Target_{strat}'].diff().abs().fillna(0)
            costs = trades * (fee_bps / 10000)
            self.data[f'Return_{strat}'] = (self.data[f'Target_{strat}'] * self.data['Spread_Return']) - costs
            self.data[f'CumReturn_{strat}'] = self.data[f'Return_{strat}'].cumsum()

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