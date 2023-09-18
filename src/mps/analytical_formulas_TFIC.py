"""
We reproduce the analytical formulas of the magnetization and
two-point correlation function for the 1D Transverse Field Ising 
Chain.

"""
# %%
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

# %%
h, k = symbols("h k")
eps_k = sqrt(h**2 - 2 * h * cos(k) + 1)
der_eps_k = eps_k.diff(k)
# %%
h_0 = symbols("h_0")
cos_k = (h * h_0 - (h + h_0) * cos(k) + 1) / (eps_k.subs(h, h_0) * eps_k)
# %%
# I = der_eps_k*ln(abs(cos_k))/pi
I = ln(abs(cos(k))) / pi
integral = Integral(I.subs([(h, 0.3), (h_0, 0)]), (k, 0, pi))
# %%
# example
k, a = symbols("k a")
sin_func = a * sin(k)
cos_func = a * cos(k)
der_cos = diff(cos_func, k)
prod = der_cos * sin_func
I = Integral(prod.subs(a, pi), (k, 0, pi))
# %%
# integration by parts
prod = cos_func.subs(a, pi) * sin_func.subs(a, pi)
first = prod.subs(x, pi) - prod.subs(x, 0)
der_sin = diff(sin_func, x)
second = Integral(der_sin * cos_func, (x, 0, pi)).subs(a, pi)
# %%
f_der = der_eps_k
f = eps_k
g = ln(abs(cos_k))
g_der = diff(g, k)
# %%
prod = f.subs(h, 0.3) * g.subs([(h, 0.3), (h_0, 0)])
first = prod.subs(k, pi) - prod.subs(k, 0)
second = Integral(f * g_der, (k, 0, pi))
# %%
