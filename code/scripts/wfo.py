
import pandas as pd
import numpy as np
import optuna
from tqdm.auto import tqdm
from backtester import BACKTESTER

# Silence Optuna's per-trial logging globally
optuna.logging.set_verbosity(optuna.logging.WARNING)


class WFO:
    def __init__(self, engine_data):
        """
        engine_data: The output from ENGINE.walk_forward()
        """
        self.data = engine_data.copy()
        self.data['Date'] = self.data.index.date
        self.unique_days = sorted(self.data['Date'].unique())

    def _objective(self, trial, train_data):
        entry_z          = trial.suggest_float("entry_z", 1.0, 2.5, step=0.1)
        exit_z           = trial.suggest_float("exit_z", -0.5, 0.5, step=0.1)
        danger_threshold = trial.suggest_float("danger_threshold", 0.50, 0.95, step=0.05)
        ar_limit         = trial.suggest_float("ar_limit", 0.85, 0.99, step=0.01)

        bt = BACKTESTER(train_data)
        results = bt.run(
            base_z=entry_z, exit_z=exit_z,
            danger_threshold=danger_threshold, ar_limit=ar_limit,
            fee_bps=0.5, slippage_mode='half_spread'
        )

        returns = results['Return_MS_AR'].fillna(0)
        if returns.std() == 0 or (results['Target_MS_AR'] != 0).sum() < 5:
            return -99.0

        return (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)

    def run_wfo(self, val_months=3, test_months=1, n_trials=100, verbose=True):
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

        if verbose:
            print(f"WFO setup: {val_months}mo train / {test_months}mo test")
            print(f"  Windows:  {len(windows)}")
            print(f"  Trials:   {n_trials} per window")
            print(f"  Period:   {windows[0][1][0]} → {windows[-1][1][-1]}")
            print()
            header = f"{'#':>3}  {'Test window':<25}  {'Best Sharpe':>12}  {'Entry Z':>8}  {'Exit Z':>7}  {'Danger':>7}  {'AR lim':>7}  {'OOS ret':>10}"
            print(header)
            print("-" * len(header))

        all_oos_results = []
        iterator = enumerate(windows, start=1)
        if verbose:
            iterator = tqdm(list(iterator), desc="WFO windows", leave=False)

        for idx, (train_days, test_days) in iterator:
            train_data = self.data[self.data['Date'].isin(train_days)]
            test_data  = self.data[self.data['Date'].isin(test_days)]

            # Optimize silently
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._objective(trial, train_data),
                n_trials=n_trials,
                show_progress_bar=False,
            )

            best = study.best_params

            # Apply OOS
            bt_oos = BACKTESTER(test_data)
            oos_results = bt_oos.run(
                base_z=best['entry_z'],
                exit_z=best['exit_z'],
                danger_threshold=best['danger_threshold'],
                ar_limit=best['ar_limit'],
                fee_bps=0.5, slippage_mode='half_spread',
            )

            all_oos_results.append(oos_results)

            if verbose:
                oos_ret = oos_results['Return_MS_AR'].fillna(0).sum()
                window_str = f"{test_days[0]} → {test_days[-1]}"
                row = (
                    f"{idx:>3}  {window_str:<25}  "
                    f"{study.best_value:>12.3f}  "
                    f"{best['entry_z']:>8.2f}  {best['exit_z']:>7.2f}  "
                    f"{best['danger_threshold']:>7.2f}  {best['ar_limit']:>7.3f}  "
                    f"{oos_ret:>10.4f}"
                )
                tqdm.write(row) if hasattr(iterator, 'write') else print(row)

        if verbose:
            print()
            print(f"WFO complete: {len(all_oos_results)} OOS windows concatenated")

        return pd.concat(all_oos_results)