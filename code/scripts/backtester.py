
from numba import njit
import numpy as np
import pandas as pd
import optuna
from tqdm.auto import tqdm

@njit
def _generate_positions(z_scores, entry_z, exit_z, signals_allowed):
    n = len(z_scores)
    pos = np.zeros(n)
    curr = 0.0  # Use float to support scaling later
    for i in range(n):
        if np.isnan(z_scores[i]):
            pos[i] = curr
            continue
            
        # FIX 1: THE PANIC BUTTON
        # If we are in a trade but the regime suddenly breaks, liquidate immediately
        if curr != 0.0 and not signals_allowed[i]:
            curr = 0.0
            
        if curr == 0.0:
            if signals_allowed[i]:
                if z_scores[i] < -entry_z: curr = 1.0
                elif z_scores[i] > entry_z: curr = -1.0
        elif curr == 1.0 and z_scores[i] >= -exit_z: curr = 0.0
        elif curr == -1.0 and z_scores[i] <= exit_z: curr = 0.0
        
        pos[i] = curr
    return pos

def _positions_daily(z_scores, entry_z, exit_z, signals_allowed, day_ids):
    """Run _generate_positions independently per day. Each day starts flat, ends flat."""
    pos = np.zeros(len(z_scores))
    for d in np.unique(day_ids):
        mask = day_ids == d
        idx = np.where(mask)[0]
        z_day = z_scores[idx]
        allow_day = signals_allowed[idx]
        pos_day = _generate_positions(z_day, entry_z, exit_z, allow_day)
        pos_day[-1] = 0  # force flatten at EOD
        pos[idx] = pos_day
    return pos


class BACKTESTER:
    def __init__(self, df):
        self.data = df.copy()

    def run(self, base_z, exit_z, danger_threshold, fee_bps=0.5,
            slippage_mode='half_spread', flatten_eod=False, **kwargs):

        z_scores  = self.data['Z_Score'].values
        mr_probs  = self.data['MR_Prob'].values

        base_allowed = np.ones(len(self.data), dtype=np.bool_)
        hard_allowed = np.where(np.isfinite(mr_probs), mr_probs >= (1.0 - danger_threshold), False)

        if flatten_eod:
            day_ids = pd.Categorical(self.data.index.date).codes.astype(np.int64)
            gen = lambda z, ez, xz, sa: _positions_daily(z, ez, xz, sa, day_ids)
        else:
            gen = _generate_positions

        # --- Baseline: rolling z-score, no regime filter ---
        pos_base = gen(z_scores, base_z, exit_z, base_allowed)

        # --- AR (Hard HMM): rolling z-score, trade only when P(MR) is high ---
        pos_ar = gen(z_scores, base_z, exit_z, hard_allowed)

        # --- MS_AR (Soft HMM): rolling z-score, positions scaled by P(MR) ---
        pos_ms_ar_raw = gen(z_scores, base_z, exit_z, base_allowed)
        
        # FIX 2: STEP-FUNCTION SCALING
        # Instead of multiplying by raw probabilities (which causes daily fee bleed),
        # we step-scale: 1.0 size if safe, 0.5 size if cautious, 0.0 if panic.
        pos_ms_ar = pos_ms_ar_raw.copy()
        
        # Panic zone: force to 0
        panic_mask = mr_probs < (1.0 - danger_threshold)
        pos_ms_ar[panic_mask] = 0.0
        
        # Caution zone: cut in half
        caution_mask = (mr_probs >= (1.0 - danger_threshold)) & (mr_probs < (1.0 - (danger_threshold / 2.0)))
        pos_ms_ar[caution_mask] = pos_ms_ar_raw[caution_mask] * 0.5

        # --- Target Assignments ---
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
            
            gross = self.data[f'Target_{strat}'] * self.data['Spread_Return']
            self.data[f'Return_{strat}_Gross'] = gross
            self.data[f'Return_{strat}'] = gross - flat_costs - slip_costs
            self.data[f'CumReturn_{strat}'] = self.data[f'Return_{strat}'].cumsum()
            self.data[f'CumReturn_{strat}_Gross'] = gross.cumsum()

        # --- Buy-and-Hold Spread ---
        self.data['Target_BuyHold'] = 1.0
        self.data['Return_BuyHold'] = self.data['Spread_Return']
        self.data['Return_BuyHold_Gross'] = self.data['Spread_Return']
        self.data['CumReturn_BuyHold'] = self.data['Return_BuyHold'].cumsum()
        self.data['CumReturn_BuyHold_Gross'] = self.data['Return_BuyHold'].cumsum()

        self.data['Return_Cash'] = 0.0
        self.data['CumReturn_Cash'] = 0.0
        return self.data