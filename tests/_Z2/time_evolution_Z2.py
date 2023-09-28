#%%
# import packages
from mps_class import MPS
from utils import *
import matplotlib.pyplot as plt
from ncon import ncon
import scipy
from scipy.sparse import csr_array
import time

# %%
# fixed parameters
L = 8
d = 4
chi = 64 # this is interpreted as d**(int(log2(chi))) --> e.g. chi=8 == 4**3=64
array = np.linspace(0.4,1.5,20)
hs = [h for h in array]
model = "Z2_two_ladder"
charges = [1,1,1,1,1,1]
trotter_steps = 1
t = 0.1
delta = t / trotter_steps
h_t = 0.4
h_ev = 0.5
n_sweeps = 2

# %%
# # exact
# psi_ev, mag_loc, mag_tot = exact_evolution(
#     L=L, h_t=h_t, h_ev=h_ev, delta=delta, trotter_steps=trotter_steps
# )
# psi_init = exact_initial_state(L=L, h_t=h_t, h_l=1e-7).reshape(2**L, 1)
# first_overlap_exact = psi_init.T.conjugate() @ psi_ev
# print(
#     f"Exact overlap between initial state and 1 trotter step evoleved state:\n{first_overlap_exact}"
# )
# %%
# mps
ladder = MPS(L=L, d=4, model=model, chi=chi, charges=charges, h=h_t)
ladder.load_sites("/Users/fradm98/Desktop/mps/tests/results/tensor_data")
# new_mps = chain.enlarge_chi()
# chain.sites = new_mps.copy()
# I have in sites the initial ground state, now we initialize the ancilla_sites and evolve it
mag_mps_tot, mag_mps_loc, overlap, errors = ladder.variational_mps_evolution(
    trotter_steps=trotter_steps, delta=delta, h_ev=h_ev, fidelity=True, conv_tol=1e-15, n_sweeps=n_sweeps
)
# %%
# plt.plot(mag_mps_tot, 'o')
# plt.plot(mag_tot)
# plt.show()
# plt.plot(overlap)
# plt.show()
# rand_trott = np.random.randint(0,trotter_steps)
# plt.title(f"Truncation error at a random trotter step: {rand_trott}")
# plt.plot(errors[rand_trott], label="distance error $\left|\left| |\phi\\rangle - |\psi\\rangle \\right|\\right|^2$")
# if len(errors[rand_trott]) > L:
#     xs = np.linspace(0,len(errors[rand_trott]),len(errors[rand_trott])//L)
# else:
#     xs = np.linspace(0,L,1) + L
# plt.vlines(xs, ymin=min(errors[rand_trott]), ymax=max(errors[rand_trott]), linestyle='--', colors='indianred', label="sweeps")
# plt.xlim(0,L+0.2)
# plt.ylabel("$\mathcal{A* N A}$ - 2$\Re(\mathcal{A*M})$ + $\langle \psi| \psi\\rangle$")
# plt.xlabel("iterations")
# plt.yscale('log')
# plt.legend()
# plt.show()
# plt.title(f"Truncation error $vs$ trotter steps")
# last_errors = [sublist[-1] for sublist in errors]
# plt.plot(last_errors, '*', label="distance error $\left|\left| |\phi\\rangle - |\psi\\rangle \\right|\\right|^2$")
# plt.ylabel("$\mathcal{A* N A}$ - 2$\Re(\mathcal{A*M})$ + $\langle \psi| \psi\\rangle$")
# plt.xlabel("Trotter Steps (T)")
# plt.yscale('log')
# plt.legend()
# plt.show()
# # we can check in the fidelity the sum of the trotter error and the truncation error
# plt.plot(overlap, label="fidelity")
# plt.plot([1-delta**2-error[-1] for _, error in zip(range(trotter_steps+1), errors)], '--', color='indianred', label="trotter + trunc error")
# plt.legend()
# plt.show()