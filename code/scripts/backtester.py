
import numpy as np
import pandas as pd

class BACKTESTER:
    def __init__(self, df):
        self.data = df.copy()

    def run(self, base_z=1.25, exit_z=0.0, danger_threshold=1.1, ar_limit=0.995, fee_bps=0.5,
            stop_loss_bps=None, take_profit_bps=None, max_hold_bars=None):
        
        # Extract columns to numpy arrays for lightning fast iteration
        z_scores = self.data['Z_Score'].values
        danger_probs = self.data['Danger_Regime_Prob'].values
        garch_vol = self.data['GARCH_Vol'].values
        ar_phi = self.data['AR_Phi'].values
        spread_returns = self.data['Spread_Return'].values # Needed to track live PnL
        
        vol_scalar = garch_vol / np.nanmedian(garch_vol)
        
        # Track both strategies
        pos_adaptive = np.zeros(len(self.data))
        pos_baseline = np.zeros(len(self.data))
        
        curr_adapt = 0  
        curr_base = 0

        # --- NEW: Trade Trackers for Execution Rules ---
        bars_held = 0
        trade_pnl_bps = 0.0

        for i in range(len(self.data)):
            if np.isnan(z_scores[i]) or np.isnan(vol_scalar[i]):
                continue

            # --- 1. STATIC BASELINE LOGIC ---
            if curr_base == 0:
                if z_scores[i] < -2.0: curr_base = 1
                elif z_scores[i] > 2.0: curr_base = -1
            elif curr_base == 1 and z_scores[i] >= exit_z: curr_base = 0
            elif curr_base == -1 and z_scores[i] <= -exit_z: curr_base = 0
            
            pos_baseline[i] = curr_base

            # --- 2. ADAPTIVE LOGIC WITH RISK MANAGEMENT ---
            in_danger = danger_probs[i] > danger_threshold
            trending = ar_phi[i] > ar_limit 

            # If we are currently IN a trade...
            if curr_adapt != 0:
                # Accumulate live PnL (in basis points) and time for the active trade
                trade_pnl_bps += curr_adapt * spread_returns[i] * 10000
                bars_held += 1
                
                # Check Execution Kill-Switches
                hit_sl = (stop_loss_bps is not None) and (trade_pnl_bps <= -stop_loss_bps)
                hit_tp = (take_profit_bps is not None) and (trade_pnl_bps >= take_profit_bps)
                hit_time = (max_hold_bars is not None) and (bars_held >= max_hold_bars)
                
                # If ANY risk rule triggers, OR market enters danger/trend -> Bail out
                if hit_sl or hit_tp or hit_time or in_danger or trending:
                    curr_adapt = 0  # Force Exit
                elif curr_adapt == 1 and z_scores[i] >= exit_z:
                    curr_adapt = 0  # Standard Exit (Mean Reverted)
                elif curr_adapt == -1 and z_scores[i] <= -exit_z:
                    curr_adapt = 0  # Standard Exit (Mean Reverted)
                    
            # If we are currently FLAT (No position)...
            else:
                # Reset trackers
                bars_held = 0
                trade_pnl_bps = 0.0
                
                # Check Entry Logic (Only if market is safe)
                if not (in_danger or trending):
                    dynamic_entry = base_z * vol_scalar[i]
                    if z_scores[i] < -dynamic_entry: curr_adapt = 1
                    elif z_scores[i] > dynamic_entry: curr_adapt = -1
            
            pos_adaptive[i] = curr_adapt

        # Save Target Positions (shifted by 1 to prevent lookahead bias)
        self.data['Target_Adaptive'] = pd.Series(pos_adaptive, index=self.data.index).shift(1).fillna(0)
        self.data['Target_Baseline'] = pd.Series(pos_baseline, index=self.data.index).shift(1).fillna(0)

        # Calculate Transaction Costs & Returns
        for strat in ['Adaptive', 'Baseline']:
            trades = self.data[f'Target_{strat}'].diff().abs().fillna(0)
            costs = trades * (fee_bps / 10000)
            self.data[f'Return_{strat}'] = (self.data[f'Target_{strat}'] * self.data['Spread_Return']) - costs
            self.data[f'CumReturn_{strat}'] = self.data[f'Return_{strat}'].cumsum()

        return self.data