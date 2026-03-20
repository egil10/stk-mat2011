import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# DGP: y_t = p*y_{t-1} + b*x_t + e_t,  e_t ~ N(0, s^2);  fit adds intercept (trend="c").
# Fit: ARIMA(1,0,0) + exog x_t  ==  minimal ARIMAX here.
np.random.seed(0)
T = 2000
x = np.random.randn(T)
e = np.random.randn(T) * 0.5
p, b = 0.35, 0.45
y = np.zeros(T)
y[0] = e[0]
for t in range(1, T):
    y[t] = p * y[t - 1] + b * x[t] + e[t]

m = ARIMA(y, order=(1, 0, 0), exog=x.reshape(-1, 1), trend="c").fit()
print(m.summary().tables[1])
