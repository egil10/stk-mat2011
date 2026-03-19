import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm
import warnings

# =============================================================================
# 1. Simulation of AR(1) HMM with k states
# =============================================================================
def simulate_ar1_hmm(T: int, P: np.ndarray, rho: np.ndarray, sigma: np.ndarray, seed: int = None):
    """
    Simulate an AR(1) process moving between k hidden states.
    
    P: Transition matrix (k x k)
    rho: AR(1) coefficients for each state (shape k,)
    sigma: Volatility for each state (shape k,)
    """
    if seed is not None:
        np.random.seed(seed)
        
    k = len(rho)
    states = np.zeros(T, dtype=int)
    y = np.zeros(T)
    
    # Assume stationary starting distribution for states (or just uniform)
    # Simple uniform start:
    states[0] = np.random.choice(k)
    y[0] = np.random.normal(0, sigma[states[0]] / np.sqrt(1 - min(rho[states[0]]**2, 0.99)))
    
    for t in range(1, T):
        states[t] = np.random.choice(k, p=P[states[t-1]])
        s = states[t]
        y[t] = rho[s] * y[t-1] + np.random.normal(scale=sigma[s])
        
    return y, states


# =============================================================================
# 2. Estimation and Filtering for K-State AR(1) HMM
# =============================================================================
class AR1_HMM:
    def __init__(self, k=2):
        self.k = k
        self.rho = None
        self.sigma = None
        self.P = None
        self.pi = None # Initial state distribution
        
    def _unpack_params(self, params):
        k = self.k
        # params:
        # [0 : k] -> rho (unconstrained b_i, transformed via tanh)
        # [k : 2k] -> sigma_raw (eta_i, transformed via exp)
        # [2k : 2k + k*(k-1)] -> P_raw (for softmax transition matrix)
        
        rho_raw = params[0:k]
        sigma_raw = params[k:2*k]
        P_raw = params[2*k : 2*k + k*(k-1)]
        
        rho = np.tanh(rho_raw)  # ensures in (-1, 1)
        sigma = np.exp(sigma_raw) # ensures > 0
        
        # Build Transition matrix using softmax to ensure rows sum to 1
        P = np.zeros((k, k))
        P_raw_reshaped = P_raw.reshape(k, k-1)
        for i in range(k):
            # We append a 0 for the last state before softmax for identifiability
            row_vals = np.append(P_raw_reshaped[i], 0.0)
            exp_vals = np.exp(row_vals - np.max(row_vals)) # max subtraction for numerical stability
            P[i] = exp_vals / np.sum(exp_vals)
            
        return rho, sigma, P

    def negative_log_likelihood(self, params, y):
        k = self.k
        T = len(y)
        rho, sigma, P = self._unpack_params(params)
        
        # Initialization
        alpha = np.ones(k) / k # uniform initial
        log_L = 0.0
        
        # Forward pass
        for t in range(1, T):
            # Emission probabilities f(y_t | s_t = j, y_{t-1})
            # To avoid underflow, we might compute directly, but if sigma is large, it's fine.
            # Normal PDF:
            y_diff = y[t] - rho * y[t-1]
            emit_probs = np.exp(-0.5 * (y_diff / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
            
            # Predict step: sum_i alpha_{t-1}(i) * P_{ij}
            alpha_pred = alpha @ P
            
            # Update step: alpha_t = alpha_pred * emit_probs
            alpha_new = alpha_pred * emit_probs
            
            c_t = np.sum(alpha_new)
            if c_t <= 0 or np.isnan(c_t):
                return 1e10 # Bad parameters penalty
                
            log_L += np.log(c_t)
            alpha = alpha_new / c_t
            
        return -log_L

    def fit(self, y, n_init=3):
        k = self.k
        best_nll = np.inf
        best_params = None
        
        for _ in range(n_init):
            # Random initialization
            rho_init = np.random.uniform(-0.5, 0.5, k)
            sigma_init = np.log(np.random.uniform(0.5, 2.0, k))
            P_init = np.random.normal(0, 0.5, k*(k-1))
            
            init_params = np.concatenate([rho_init, sigma_init, P_init])
            
            try:
                res = opt.minimize(
                    self.negative_log_likelihood, 
                    init_params, 
                    args=(y,),
                    method='L-BFGS-B',
                    options={'maxiter': 500}
                )
                if res.success and res.fun < best_nll:
                    best_nll = res.fun
                    best_params = res.x
            except Exception as e:
                pass
                
        if best_params is None:
            warnings.warn("HMM Optimization failed.")
            return False
            
        self.nll_ = best_nll
        self.rho, self.sigma, self.P = self._unpack_params(best_params)
        return True

    def filtered_probabilities(self, y):
        """Returns P(s_t = i | y_{1:t}) and the 1-step predicted state probs P(s_{t+1} | y_{1:t})"""
        k = self.k
        T = len(y)
        filter_probs = np.zeros((T, k))
        pred_probs = np.zeros((T, k))
        
        alpha = np.ones(k) / k
        filter_probs[0] = alpha
        pred_probs[0] = alpha @ self.P
        
        for t in range(1, T):
            y_diff = y[t] - self.rho * y[t-1]
            emit_probs = np.exp(-0.5 * (y_diff / self.sigma)**2) / (np.sqrt(2 * np.pi) * self.sigma)
            
            alpha_pred = alpha @ self.P
            alpha_new = alpha_pred * emit_probs
            
            c_t = np.sum(alpha_new)
            if c_t > 0:
                alpha = alpha_new / c_t
            
            filter_probs[t] = alpha
            pred_probs[t] = alpha @ self.P
            
        return filter_probs, pred_probs

    def bic(self, y):
        """Compute Bayesian Information Criterion (BIC)."""
        k = self.k
        n_params = 2 * k + k * (k - 1)
        # We need the log-likelihood evaluated at the best parameters. 
        # So we just re-evaluate negative_log_likelihood using the current parameters.
        # But we need to pack them first:
        rho_raw = np.arctanh(self.rho)
        sigma_raw = np.log(self.sigma)
        
        # Packing P is harder. Let's just calculate the log-likelihood directly using a custom run or 
        # store the log-likelihood from the fit.
        # Let's just store the best negative log likelihood in self.nll_ during fit().
        # If it's not stored, we return np.nan
        if not hasattr(self, 'nll_'):
            return np.nan
        
        return 2 * self.nll_ + n_params * np.log(len(y))
        
    def aic(self, y):
        """Compute Akaike Information Criterion (AIC)."""
        k = self.k
        n_params = 2 * k + k * (k - 1)
        if not hasattr(self, 'nll_'):
            return np.nan
        return 2 * self.nll_ + 2 * n_params

# =============================================================================
# 3. Model Comparisons & Predictions
# =============================================================================
def fit_single_ar1(y):
    """Fit a single AR(1) model: y_t = rho * y_{t-1} + e_t"""
    Y = y[1:]
    X = y[:-1]
    rho = np.sum(X * Y) / np.sum(X * X)
    sigma = np.sqrt(np.mean((Y - rho * X)**2))
    return rho, sigma

def run_prediction_experiment(y_train, y_test, model_hmm, single_rho, single_sigma):
    """
    Evaluate 1-step ahead predictions out-of-sample for all 3 methods:
    1. Single AR(1)
    2. HMM Hard Switch
    3. HMM Weighted Mixture
    """
    T_test = len(y_test)
    preds_single = np.zeros(T_test)
    preds_hard = np.zeros(T_test)
    preds_mixed = np.zeros(T_test)
    
    # Coverage tracking (95% interval)
    cov_single = np.zeros(T_test)
    cov_hard = np.zeros(T_test)
    cov_mixed = np.zeros(T_test)
    
    # We need to maintain the filter probability sequentially
    # Initialize with the final filter probability from the training set
    _, pred_probs_train = model_hmm.filtered_probabilities(y_train)
    p_next_state = pred_probs_train[-1] # P(s_{t+1} | y_{1:T_train})
    
    last_y = y_train[-1]
    
    for t in range(T_test):
        true_y = y_test[t]
        
        # --- 1. Single AR(1) ---
        preds_single[t] = single_rho * last_y
        ci_single = 1.96 * single_sigma
        cov_single[t] = (true_y >= preds_single[t] - ci_single) and (true_y <= preds_single[t] + ci_single)
        
        # --- 2. HMM Hard Switch ---
        best_state = np.argmax(p_next_state)
        preds_hard[t] = model_hmm.rho[best_state] * last_y
        ci_hard = 1.96 * model_hmm.sigma[best_state]
        cov_hard[t] = (true_y >= preds_hard[t] - ci_hard) and (true_y <= preds_hard[t] + ci_hard)
        
        # --- 3. HMM Mixture ---
        # hat{y}_t = sum_i p_i * rho_i * y_{t-1}
        mu_components = model_hmm.rho * last_y
        preds_mixed[t] = np.sum(p_next_state * mu_components)
        
        # Mixture variance: Var(Y) = E[Var(Y|S)] + Var(E[Y|S])
        sigma2_mix = np.sum(p_next_state * model_hmm.sigma**2) + \
                     np.sum(p_next_state * (mu_components - preds_mixed[t])**2)
                     
        ci_mixed = 1.96 * np.sqrt(sigma2_mix) # using Gaussian approximation for interval
        cov_mixed[t] = (true_y >= preds_mixed[t] - ci_mixed) and (true_y <= preds_mixed[t] + ci_mixed)
        
        # --- Step Forward ---
        # Update filter probability with the new observation true_y
        y_diff = true_y - model_hmm.rho * last_y
        emit_probs = np.exp(-0.5 * (y_diff / model_hmm.sigma)**2) / (np.sqrt(2 * np.pi) * model_hmm.sigma)
        
        alpha_new = p_next_state * emit_probs
        c_t = np.sum(alpha_new)
        if c_t > 0:
            alpha = alpha_new / c_t
        else:
            alpha = np.ones(model_hmm.k) / model_hmm.k
            
        p_next_state = alpha @ model_hmm.P
        last_y = true_y
        
    rmse_single = np.sqrt(np.mean((preds_single - y_test)**2))
    rmse_hard = np.sqrt(np.mean((preds_hard - y_test)**2))
    rmse_mixed = np.sqrt(np.mean((preds_mixed - y_test)**2))

    return {
        "RMSE Single": rmse_single,
        "RMSE Hard": rmse_hard,
        "RMSE Mixed": rmse_mixed,
        "Coverage Single": np.mean(cov_single),
        "Coverage Hard": np.mean(cov_hard),
        "Coverage Mixed": np.mean(cov_mixed)
    }

# =============================================================================
# 4. Pipeline Execution
# =============================================================================
if __name__ == "__main__":
    # Settings
    T_train = 1000
    T_test = 500
    np.random.seed(42)
    
    # Ground Truth Parameters (2 States)
    # State 0: Low variance, low autoregression
    # State 1: High variance, high autoregression (trend-following)
    true_rho = np.array([0.1, 0.8])
    true_sigma = np.array([0.5, 1.5])
    
    # "Stickiness": High diagonal values mean states persist longer
    true_P = np.array([
        [0.98, 0.02],
        [0.05, 0.95]
    ])
    
    print("Simulating Data...")
    y_full, state_full = simulate_ar1_hmm(T_train + T_test, true_P, true_rho, true_sigma)
    
    y_train = y_full[:T_train]
    y_test = y_full[T_train:]
    
    print("Fitting Single AR(1)...")
    single_rho, single_sigma = fit_single_ar1(y_train)
    print(f"Single AR(1) -> rho: {single_rho:.3f}, sigma: {single_sigma:.3f}")
    
    print("\nFitting AR(1) HMM (2 states)...")
    hmm = AR1_HMM(k=2)
    success = hmm.fit(y_train)
    
    if success:
        print("HMM Fitted Parameters:")
        print(f"  rho: {hmm.rho}")
        print(f"  sigma: {hmm.sigma}")
        print(f"  Transition Matrix:\n{hmm.P}")
        print(f"  AIC: {hmm.aic(y_train):.2f}")
        print(f"  BIC: {hmm.bic(y_train):.2f}")
        
        print("\nEvaluating Out-of-Sample Predictions...")
        results = run_prediction_experiment(y_train, y_test, hmm, single_rho, single_sigma)
        
        print("\n--- EXPERIMENT RESULTS ---")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
            
    else:
        print("HMM Fitting failed. Try altering initialization or bounds.")
