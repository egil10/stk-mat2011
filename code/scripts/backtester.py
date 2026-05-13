
from numba import njit
import numpy as np
import pandas as pd
import optuna
from tqdm.auto import tqdm

@njit
def _generate_positions(z_scores, entry_z_arr, exit_z, signals_allowed):
    n = len(z_scores)
    pos = np.zeros(n)
    curr = 0.0  
    for i in range(n):
        if np.isnan(z_scores[i]):
            pos[i] = curr
            continue
            
        # Panic Button (for the AR bot)
        if curr != 0.0 and not signals_allowed[i]:
            curr = 0.0
            
        if curr == 0.0:
            if signals_allowed[i]:
                # KEY CHANGE: Compare against the array index [i]
                if z_scores[i] < -entry_z_arr[i]: curr = 1.0
                elif z_scores[i] > entry_z_arr[i]: curr = -1.0
        elif curr == 1.0 and z_scores[i] >= -exit_z: curr = 0.0
        elif curr == -1.0 and z_scores[i] <= exit_z: curr = 0.0
        
        pos[i] = curr
    return pos

def _positions_daily(z_scores, entry_z_arr, exit_z, signals_allowed, day_ids):
    """Run _generate_positions independently per day."""
    pos = np.zeros(len(z_scores))
    for d in np.unique(day_ids):
        mask = day_ids == d
        idx = np.where(mask)[0]
        # Pass the sliced arrays to the Numba function
        pos_day = _generate_positions(z_scores[idx], entry_z_arr[idx], exit_z, signals_allowed[idx])
        pos_day[-1] = 0  
        pos[idx] = pos_day
    return pos


class BACKTESTER:
    def __init__(self, df):
        self.data = df.copy()

    def run(self, z_quiet, z_volatile, exit_z, danger_threshold, fee_bps=0.5,
            slippage_mode='half_spread', flatten_eod=False,
            prob_smoothing=0, **kwargs):
        """
        prob_smoothing : int, default 0
            Window of a centred-no-look-ahead rolling median applied to
            MR_Prob / Danger_Regime_Prob before they enter the strategy.
            0 (or 1) disables smoothing — this matches the paper. Set to
            ≥2 to suppress single-bar regime flips at the cost of one bar
            of lag.
        """
        z_scores  = self.data['Z_Score'].values

        # 1. Optional smoothing of regime probabilities (off by default;
        #    paper uses the raw posteriors). When enabled, bfill is used
        #    only to seed the very first (window-1) bars.
        if prob_smoothing and prob_smoothing > 1:
            mr_probs     = self.data['MR_Prob'].rolling(window=prob_smoothing).median().bfill().values
            danger_probs = self.data['Danger_Regime_Prob'].rolling(window=prob_smoothing).median().bfill().values
        else:
            mr_probs     = self.data['MR_Prob'].values
            danger_probs = self.data['Danger_Regime_Prob'].values

        base_allowed = np.ones(len(self.data), dtype=np.bool_)
        hard_allowed = np.where(np.isfinite(mr_probs), mr_probs >= (1.0 - danger_threshold), False)

        # 2. THE GEARBOX: Generate arrays for entry thresholds
        # Baseline and AR bots use the static "Quiet" Z
        static_entry_z = np.full(len(z_scores), z_quiet)
        
        # MS_AR blends them: If P(Danger) is high, the net widens to z_volatile
        dynamic_entry_z = (mr_probs * z_quiet) + (danger_probs * z_volatile)

        if flatten_eod:
            day_ids = pd.Categorical(self.data.index.date).codes.astype(np.int64)
            gen = lambda z, ez, xz, sa: _positions_daily(z, ez, xz, sa, day_ids)
        else:
            gen = _generate_positions

        # 3. Apply the dynamic array to MS_AR
        pos_base  = gen(z_scores, static_entry_z, exit_z, base_allowed)
        pos_ar    = gen(z_scores, static_entry_z, exit_z, hard_allowed)
        pos_ms_ar = gen(z_scores, dynamic_entry_z, exit_z, base_allowed)
        
        # ... rest of your PnL math remains the same

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