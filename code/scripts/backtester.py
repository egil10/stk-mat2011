
import numpy as np
import pandas as pd

class BACKTESTER:
    """
    Event-loop backtester with:
      * vol-scaled dynamic entry threshold (GARCH)
      * regime kill-switch (fixed: threshold now in [0,1])
      * trend kill-switch (AR phi)
      * stop-loss / take-profit / max-hold
      * flat fee_bps PLUS per-leg half-spread slippage on every trade
    """
 
    def __init__(self, df):
        self.data = df.copy()
 
    def run(self, base_z=1.25, exit_z=0.0, danger_threshold=0.7, ar_limit=0.995,
            fee_bps=0.5, stop_loss_bps=None, take_profit_bps=None, max_hold_bars=None,
            slippage_mode='half_spread'):
        """
        slippage_mode: 'half_spread' -> pay HalfSpread_A_bps + HalfSpread_B_bps
                       on each position change (entry or exit). 'none' -> skip.
        """
        z_scores = self.data['Z_Score'].values
        danger_probs = self.data['Danger_Regime_Prob'].values
        garch_vol = self.data['GARCH_Vol'].values
        ar_phi = self.data['AR_Phi'].values
        spread_returns = self.data['Spread_Return'].values
 
        # Guard against all-NaN or single-value GARCH (e.g. if OOS had one refit)
        med = np.nanmedian(garch_vol)
        if not np.isfinite(med) or med == 0:
            vol_scalar = np.ones_like(garch_vol)
        else:
            vol_scalar = garch_vol / med
 
        pos_adaptive = np.zeros(len(self.data))
        pos_baseline = np.zeros(len(self.data))
        curr_adapt, curr_base = 0, 0
        bars_held, trade_pnl_bps = 0, 0.0
 
        for i in range(len(self.data)):
            if np.isnan(z_scores[i]) or np.isnan(vol_scalar[i]):
                pos_adaptive[i] = curr_adapt
                pos_baseline[i] = curr_base
                continue
 
            # --- Static baseline ---
            if curr_base == 0:
                if z_scores[i] < -2.0:
                    curr_base = 1
                elif z_scores[i] > 2.0:
                    curr_base = -1
            elif curr_base == 1 and z_scores[i] >= exit_z:
                curr_base = 0
            elif curr_base == -1 and z_scores[i] <= -exit_z:
                curr_base = 0
            pos_baseline[i] = curr_base
 
            # --- Adaptive ---
            in_danger = danger_probs[i] > danger_threshold
            trending = ar_phi[i] > ar_limit if np.isfinite(ar_phi[i]) else False
 
            if curr_adapt != 0:
                trade_pnl_bps += curr_adapt * spread_returns[i] * 10000
                bars_held += 1
                hit_sl = (stop_loss_bps is not None) and (trade_pnl_bps <= -stop_loss_bps)
                hit_tp = (take_profit_bps is not None) and (trade_pnl_bps >= take_profit_bps)
                hit_time = (max_hold_bars is not None) and (bars_held >= max_hold_bars)
                if hit_sl or hit_tp or hit_time or in_danger or trending:
                    curr_adapt = 0
                elif curr_adapt == 1 and z_scores[i] >= exit_z:
                    curr_adapt = 0
                elif curr_adapt == -1 and z_scores[i] <= -exit_z:
                    curr_adapt = 0
            else:
                bars_held, trade_pnl_bps = 0, 0.0
                if not (in_danger or trending):
                    dynamic_entry = base_z * vol_scalar[i]
                    if z_scores[i] < -dynamic_entry:
                        curr_adapt = 1
                    elif z_scores[i] > dynamic_entry:
                        curr_adapt = -1
 
            pos_adaptive[i] = curr_adapt
 
        # Shift positions by 1 bar to avoid look-ahead
        self.data['Target_Adaptive'] = pd.Series(pos_adaptive, index=self.data.index).shift(1).fillna(0)
        self.data['Target_Baseline'] = pd.Series(pos_baseline, index=self.data.index).shift(1).fillna(0)
 
        # --- Transaction costs: flat fee + half-spread slippage on each leg ---
        half_spread_total_bps = (
            self.data.get('HalfSpread_A_bps', pd.Series(0.0, index=self.data.index)).fillna(0)
            + self.data.get('HalfSpread_B_bps', pd.Series(0.0, index=self.data.index)).fillna(0)
        )
 
        for strat in ['Adaptive', 'Baseline']:
            trades = self.data[f'Target_{strat}'].diff().abs().fillna(0)  # 0, 1, or 2
            flat_costs = trades * (fee_bps / 10000.0)
            if slippage_mode == 'half_spread':
                slip_costs = trades * (half_spread_total_bps / 10000.0)
            else:
                slip_costs = 0.0
            self.data[f'Return_{strat}'] = (
                self.data[f'Target_{strat}'] * self.data['Spread_Return']
                - flat_costs - slip_costs
            )
            self.data[f'CumReturn_{strat}'] = self.data[f'Return_{strat}'].cumsum()
 
        return self.data