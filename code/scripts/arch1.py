import numpy as np
from arch import arch_model

# DGP: h_t^2 = w + a*e_{t-1}^2,  e_t = h_t*z_t,  z_t ~ N(0,1).  Fit: mean="Zero", vol ARCH(1).
# Script name arch1.py (not arch.py) so "import arch" is the package, not this file.
np.random.seed(1)
T = 2000
w, a = 0.05, 0.88
z = np.random.randn(T)
e = np.zeros(T)
h2 = w / (1 - a)
e[0] = np.sqrt(h2) * z[0]
for t in range(1, T):
    h2 = w + a * e[t - 1] ** 2
    e[t] = np.sqrt(h2) * z[t]
y = e * 100  # scale so omega is not tiny in the optimizer

r = arch_model(y, mean="Zero", vol="ARCH", p=1, rescale=False).fit(disp="off")
print(r.summary())
