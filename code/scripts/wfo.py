
import pandas as pd
import numpy as np
import optuna
from tqdm.auto import tqdm
from backtester import BACKTESTER

# Silence Optuna's per-trial logging globally
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Bars-per-year heuristic used to annualise high-frequency Sharpe.
_ANN_FACTOR = 252 * 24 * 60


def _annualised_sharpe(returns):
    r = returns.fillna(0)
    s = r.std()
    if s == 0 or not np.isfinite(s):
        return 0.0
    return float(r.mean() / s * np.sqrt(_ANN_FACTOR))


class WFO:
    def __init__(self, engine_data, flatten_eod=False):
        """
        engine_data: The output from ENGINE.walk_forward()
        flatten_eod: If True, positions reset each calendar day (no overnight carry)
        """
        self.data = engine_data.copy()
        self.data['Date'] = self.data.index.date
        self.unique_days = sorted(self.data['Date'].unique())
        self.flatten_eod = flatten_eod

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------

    def _objective(self, trial, train_data, objective):
        # Hunt for two Z's instead of one
        z_quiet          = trial.suggest_float("z_quiet", 1.0, 2.0, step=0.1)
        z_volatile       = trial.suggest_float("z_volatile", 2.0, 4.5, step=0.1)
        exit_z           = trial.suggest_float("exit_z", -0.5, 0.5, step=0.1)
        danger_threshold = trial.suggest_float("danger_threshold", 0.05, 0.50, step=0.05)

        bt = BACKTESTER(train_data)
        results = bt.run(
            z_quiet=z_quiet, z_volatile=z_volatile, exit_z=exit_z,
            danger_threshold=danger_threshold,
            fee_bps=0.5, slippage_mode='half_spread',
            flatten_eod=self.flatten_eod,
        )

        ms_returns = results['Return_MS_AR'].fillna(0)
        ar_returns = results['Return_AR'].fillna(0)

        # Hard guard: if MS-AR barely trades, the Sharpe is meaningless.
        if (results['Target_MS_AR'] != 0).sum() < 5:
            return -99.0
        if ms_returns.std() == 0:
            return -99.0

        sharpe_ms = _annualised_sharpe(ms_returns)

        if objective == 'ms_only':
            return sharpe_ms

        # 'ms_minus_ar': directly proves the regime model adds value over the
        # hard-kill AR baseline. We do NOT short-circuit on AR no-trade --
        # if AR sits in cash, sharpe_ar = 0 and MS-AR's standalone Sharpe wins.
        sharpe_ar = _annualised_sharpe(ar_returns)
        return sharpe_ms - sharpe_ar

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_wfo(self, val_months=12, test_months=3, n_trials=100,
                objective='ms_minus_ar', verbose=True):
        """
        objective : 'ms_minus_ar' (default) or 'ms_only'.
            'ms_minus_ar' searches for params where the Markov gearbox actually
            beats the hard-kill AR baseline -- a fair test of the regime model.
            'ms_only' replicates the legacy behaviour (maximise MS-AR Sharpe
            standalone, AR gets dragged along with whatever danger_threshold
            wins for MS-AR).
        """
        if objective not in ('ms_minus_ar', 'ms_only'):
            raise ValueError(f"objective must be 'ms_minus_ar' or 'ms_only', got {objective!r}")

        val_step  = val_months  * 21
        test_step = test_months * 21

        # Pre-compute the windows so tqdm knows the total count
        windows = []
        for i in range(val_step, len(self.unique_days), test_step):
            train_days = self.unique_days[i - val_step : i]
            test_days  = self.unique_days[i : i + test_step]
            if not test_days:
                break
            windows.append((train_days, test_days))

        if not windows:
            raise ValueError(
                f"WFO has 0 windows: need at least {val_months + test_months} months of data "
                f"(got {len(self.unique_days)} unique days). "
                f"Lower val_months / test_months or pass more history."
            )

        if verbose:
            print(f"WFO setup: {val_months}mo train / {test_months}mo test  |  objective={objective}")
            print(f"  Windows:  {len(windows)}")
            print(f"  Trials:   {n_trials} per window")
            print(f"  Period:   {windows[0][1][0]} → {windows[-1][1][-1]}")
            print()
            header = (
                f"{'#':>3}  {'Test window':<25}  {'Best score':>11}  "
                f"{'Zq':>5}  {'Zv':>5}  {'ExitZ':>6}  {'DngThr':>7}  "
                f"{'OOS MS':>9}  {'OOS AR':>9}"
            )
            print(header)
            print("-" * len(header))

        all_oos_results = []
        iterator = enumerate(windows, start=1)
        if verbose:
            iterator = tqdm(list(iterator), desc="WFO windows", leave=False)

        for idx, (train_days, test_days) in iterator:
            train_data = self.data[self.data['Date'].isin(train_days)]
            test_data  = self.data[self.data['Date'].isin(test_days)]

            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._objective(trial, train_data, objective),
                n_trials=n_trials,
                show_progress_bar=False,
            )

            best = study.best_params

            # Apply OOS
            bt_oos = BACKTESTER(test_data)
            oos_results = bt_oos.run(
                z_quiet=best['z_quiet'],
                z_volatile=best['z_volatile'],
                exit_z=best['exit_z'],
                danger_threshold=best['danger_threshold'],
                fee_bps=0.5, slippage_mode='half_spread',
                flatten_eod=self.flatten_eod,
            )
            all_oos_results.append(oos_results)

            if verbose:
                ms_ret = oos_results['Return_MS_AR'].fillna(0).sum()
                ar_ret = oos_results['Return_AR'].fillna(0).sum()
                window_str = f"{test_days[0]} → {test_days[-1]}"
                row = (
                    f"{idx:>3}  {window_str:<25}  "
                    f"{study.best_value:>11.3f}  "
                    f"{best['z_quiet']:>5.2f}  {best['z_volatile']:>5.2f}  "
                    f"{best['exit_z']:>6.2f}  {best['danger_threshold']:>7.2f}  "
                    f"{ms_ret:>9.4f}  {ar_ret:>9.4f}"
                )
                tqdm.write(row) if hasattr(iterator, 'write') else print(row)

        if verbose:
            print()
            print(f"WFO complete: {len(all_oos_results)} OOS windows concatenated")

        return pd.concat(all_oos_results)
