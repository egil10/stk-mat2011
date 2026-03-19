# HMM AR(1): Project Status & Next Steps

## 1. Current Status (From Last Week)

### What We Have
- **Simulations:** We have a clear mathematical note and a small Python snippet for a 2-state Hidden Markov Model (HMM) where the observations follow an Autoregressive (AR(1)) process defined by $\rho$ and $\sigma$.
- **Theory:** The notes derive the conditional density, the likelihood using forward probabilities ($\alpha_t(i)$) to prevent exponential complexity, and techniques for numerical stability (rescaling tricks) or parameter transformations (for variances, bounded transitions, and stationary roots).

### Codebase Investigation
- **What is implemented:** Looking at `code/scripts/hmm.py`, we see an implementation fitting a pure `GaussianHMM` (via `hmmlearn`) directly onto empirical tick-data features (log returns).
- **What is NOT yet implemented:** The project does **not yet contain** any code implementing the AR(1)-HMM model defined in the note. Specifically, the custom forward-probability log-likelihood function, numerical optimization loops, filtered probabilities, and AR(1) simulations are still missing from the active codebase. 
- **Recommendation:** We will need to write a script that generates this simulated AR(1) data, optimizes the custom parameters using `scipy.optimize.minimize`, and tests the output. *(Alternatively, `statsmodels` provides `MarkovAutoregression` which can be used to verify our custom implementation!)*

---

## 2. Next Steps (This Week's Plan)

### A. Generalization
1. **$k$-States:** Generalize the mathematical note outlining the filter probabilities and likelihood from 2 hidden states over to $k$ hidden states. Write this out nicely for inclusion in the project thesis.
2. **Other Models:** Consider how this framework generalizes beyond AR(1), to models like ARIMAX, ARCH, and GARCH. 
3. **Number of States:** Review the literature on how to determine a reasonable number of states (e.g., AIC, BIC, cross-validation). Investigate this empirically on our simulated data by tweaking the sample size ($n$) and the separation/difference between the true state parameters.

### B. Prediction and Calibration Experiments
*The following experiments will rely on simulating an AR(1) process with 2 sub-states, and varying three main ingredients: (a) Number of observations, (b) Difference in parameters between states, and (c) "Stickiness" (time spent in each state before transitioning).*

4. **HMM AR(1) vs. Single AR(1) (Hard Switch):** Investigate when an HMM AR(1) starts performing sufficiently better than simply fitting one global AR(1) to all data.
   - Use out-of-sample 1-step-ahead prediction with RMSE to evaluate performance.
   - For the HMM prediction, switch to the model given by the state with the highest filtered probability ($p > 0.5$).
5. **Prediction Intervals (Hard Switch):** Develop a prediction interval for the next step incorporating the specific $\sigma$ state parameters. Check calibration: does an estimated 95% interval actually capture the next simulated observations roughly 95% of the time?
6. **Hard Switch vs. Weighted Mixture:** Compare the prediction RMSE using three different methodologies:
   - Single AR(1) estimated on all data.
   - HMM AR(1) with a "Hard Switch" ($p > 0.5$).
   - HMM AR(1) with a "Weighted Mixture": $\hat{y}_t = p_1\hat{y}_t^{(1)} + p_2\hat{y}_t^{(2)}$.
7. **Mixture Prediction Interval:** Derive a formula for the variance/prediction interval when $\hat{y}_t$ is estimated using the weighted mixture rather than purely the argmax state. Contrast the calibration/coverage of this interval against both the global AR(1) and the hard switch prediction.
