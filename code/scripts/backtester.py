
import numpy as np
import pandas as pd
from numba import njit

@njit
def _generate_positions(z_scores, entry_z, exit_z, signals_allowed):
    n = len(z_scores)
    pos = np.zeros(n)
    curr = 0
    for i in range(n):
        if np.isnan(z_scores[i]):
            pos[i] = curr
            continue
        if curr == 0:
            if signals_allowed[i]:
                if z_scores[i] < -entry_z: curr = 1
                elif z_scores[i] > entry_z: curr = -1
        elif curr == 1 and z_scores[i] >= -exit_z: curr = 0
        elif curr == -1 and z_scores[i] <= exit_z: curr = 0
        pos[i] = curr
    return pos

class BACKTESTER:
    def __init__(self, df):
        self.data = df.copy()

    # All parameters exposed here for dynamic calling
    def run(self, base_z, exit_z, danger_threshold, ar_limit, fee_bps=0.5, slippage_mode='half_spread'):
        
        z_scores = self.data['Z_Score'].values
        ar_phi = self.data['AR_Phi'].values
        danger_probs = self.data['Danger_Regime_Prob'].values
        
        base_allowed = np.ones(len(self.data), dtype=np.bool_)
        ar_allowed = np.where(np.isfinite(ar_phi), ar_phi < ar_limit, False)
        ms_ar_allowed = ar_allowed & (danger_probs <= danger_threshold)

        pos_base = _generate_positions(z_scores, base_z, exit_z, base_allowed)
        pos_ar = _generate_positions(z_scores, base_z, exit_z, ar_allowed)
        pos_ms_ar = _generate_positions(z_scores, base_z, exit_z, ms_ar_allowed)

        self.data['Target_Baseline'] = pd.Series(pos_base, index=self.data.index).shift(1).fillna(0)
        self.data['Target_AR'] = pd.Series(pos_ar, index=self.data.index).shift(1).fillna(0)
        self.data['Target_MS_AR'] = pd.Series(pos_ms_ar, index=self.data.index).shift(1).fillna(0)

        half_spread_total_bps = (
            self.data.get('HalfSpread_A_bps', pd.Series(0.0, index=self.data.index)).fillna(0) +
            self.data.get('HalfSpread_B_bps', pd.Series(0.0, index=self.data.index)).fillna(0)
        )

        for strat in ['Baseline', 'AR', 'MS_AR']:
            trades = self.data[f'Target_{strat}'].diff().abs().fillna(0)
            flat_costs = trades * (fee_bps / 10000.0)
            slip_costs = trades * (half_spread_total_bps / 10000.0) if slippage_mode == 'half_spread' else 0.0
            
            self.data[f'Return_{strat}'] = (self.data[f'Target_{strat}'] * self.data['Spread_Return']) - flat_costs - slip_costs
            self.data[f'CumReturn_{strat}'] = self.data[f'Return_{strat}'].cumsum()

        self.data['Return_Cash'] = 0.0
        self.data['CumReturn_Cash'] = 0.0
        return self.data