
import pandas as pd
import numpy as np
import optuna
from backtester import BACKTESTER

class WFO:
    def __init__(self, engine_data):
        """
        engine_data: The output from ENGINE.walk_forward()
        """
        self.data = engine_data.copy()
        self.data['Date'] = self.data.index.date
        self.unique_days = sorted(self.data['Date'].unique())

    def _objective(self, trial, train_data):
        # Define the search space for Optuna
        entry_z = trial.suggest_float("entry_z", 1.0, 2.5, step=0.1)
        exit_z = trial.suggest_float("exit_z", -0.5, 0.5, step=0.1)
        danger_threshold = trial.suggest_float("danger_threshold", 0.50, 0.95, step=0.05)
        ar_limit = trial.suggest_float("ar_limit", 0.85, 0.99, step=0.01)

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

    def run_wfo(self, val_months=3, test_months=1, n_trials=100):
        print(f"Starting WFO: {val_months}mo Validation / {test_months}mo Test...")
        
        all_oos_results = []
        
        # Approximate 21 trading days per month
        val_step = val_months * 21
        test_step = test_months * 21
        
        # Start at the end of the first validation window
        for i in range(val_step, len(self.unique_days), test_step):
            # 1. Define Windows
            train_days = self.unique_days[i - val_step : i]
            test_days = self.unique_days[i : i + test_step]
            
            if not test_days: break
            
            train_data = self.data[self.data['Date'].isin(train_days)]
            test_data = self.data[self.data['Date'].isin(test_days)]
            
            print(f"Tuning for {test_days[0]} to {test_days[-1]}...")

            # 2. Optimize on the Training Window
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self._objective(trial, train_data), n_trials=n_trials)
            
            best = study.best_params
            
            # 3. Apply best params to the Out-of-Sample (OOS) Window
            bt_oos = BACKTESTER(test_data)
            oos_results = bt_oos.run(
                base_z=best['entry_z'], 
                exit_z=best['exit_z'], 
                danger_threshold=best['danger_threshold'], 
                ar_limit=best['ar_limit'],
                fee_bps=0.5, 
                slippage_mode='half_spread'
            )
            
            all_oos_results.append(oos_results)
            
        return pd.concat(all_oos_results)