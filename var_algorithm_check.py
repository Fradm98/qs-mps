#%%
# import packages
from mps_class_v9 import MPS
from utils import *
import matplotlib.pyplot as plt
from ncon import ncon
import scipy
from scipy.sparse import csr_array
import time

# %%
# fixed parameters
trotter_steps = 10
t = 1
delta = t / trotter_steps
h_t = 0
h_ev = 0.3
L = 9
chi = 16
n_sweeps = 4

# %%
# exact
psi_ev, mag_loc, mag_tot = exact_evolution(
    L=L, h_t=h_t, h_ev=h_ev, delta=delta, trotter_steps=trotter_steps
)
psi_init = exact_initial_state(L=L, h_t=h_t, h_l=1e-7).reshape(2**L, 1)
first_overlap_exact = psi_init.T.conjugate() @ psi_ev
print(
    f"Exact overlap between initial state and 1 trotter step evoleved state:\n{first_overlap_exact}"
)
# %%
# mps
chain = MPS(L=L, d=2, model="Ising", chi=chi, h=h_t, eps=0, J=1)
chain._random_state(seed=3, chi=chi)
chain.canonical_form(trunc_chi=True, trunc_tol=False)
chain.sweeping(trunc=True, n_sweeps=2)
chain.flipping_mps()
# I have in sites the initial ground state, now we initialize the ancilla_sites and evolve it
mag_mps_tot, mag_mps_loc, overlap, errors = chain.variational_mps_evolution(
    trotter_steps=trotter_steps, delta=delta, h_ev=h_ev, fidelity=True, conv_tol=1e-7, n_sweeps=n_sweeps
)

# %%
plt.plot(mag_mps_tot, 'o')
plt.plot(mag_tot)
plt.show()
plt.plot(overlap)
plt.show()
plt.plot(errors[-1])
plt.ylabel("$\mathcal{A* N A}$ - 2$\Re(\mathcal{A*M})$ + $\langle \psi| \psi\\rangle$")
plt.xlabel("iterations")
plt.yscale('log')
plt.show()
plt.plot(np.asarray(errors)[:,-1])
plt.ylabel("$\mathcal{A* N A}$ - 2$\Re(\mathcal{A*M})$ + $\langle \psi| \psi\\rangle$")
plt.xlabel("Trotter Steps (T)")
plt.yscale('log')
plt.show()
# %%
