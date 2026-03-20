import numpy as np
from arch import arch_model

# DGP: h_t^2 = w + a*e_{t-1}^2 + b*h_{t-1}^2,  e_t = h_t*z_t.  Fit: mean="Zero", vol GARCH(1,1).
# Loop: h2 is h_t^2; RHS uses previous h2 as h_{t-1}^2 before overwrite.
np.random.seed(2)
T = 2000
w, a, b = 0.02, 0.08, 0.90
z = np.random.randn(T)
e = np.zeros(T)
h2 = w / (1 - a - b)
e[0] = np.sqrt(h2) * z[0]
for t in range(1, T):
    h2 = w + a * e[t - 1] ** 2 + b * h2
    e[t] = np.sqrt(h2) * z[t]
y = e * 100  # same scaling idea as arch1.py

r = arch_model(y, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
print(r.summary())
