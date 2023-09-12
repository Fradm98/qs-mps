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
# variational compression changing bond dimension
chi = 16
trotter_step_list = [5,10,40,50]
t = 5
deltas = t/np.asarray(trotter_step_list)
h_t = 0
h_ev = 0.3
L = 9
n_sweeps = 2

# exact
psi_exact = exact_initial_state(L=L, h_t=h_t, h_l=1e-5).reshape(2**L,1)
Z = np.array([[1,0],[0,-1]])
# local
mag_loc_op = [single_site_op(op=Z, site=i, L=L) for i in range(1,L+1)]
# total
mag_tot_op = H_loc(L=L, op=Z)

mag_exact_loc = []
mag_exact_tot = []

# local
mag_exact = []
for i in range(L):
    mag_exact.append((psi_exact.T.conjugate() @ mag_loc_op[i] @ psi_exact).data[0].real)
mag_exact_loc.append(mag_exact)

# total
mag = (psi_exact.T.conjugate() @ mag_tot_op @ psi_exact).data
mag_exact_tot.append(mag.real)

trotter_steps = 50
delta = 0.1
for trott in range(trotter_steps):
    # exact
    psi_new = exact_evolution(L=L, psi_init=psi_exact, trotter_step=trott+1, delta=delta, h_t=h_ev)

    # local
    mag_exact = []
    for i in range(L):
        mag_exact.append((psi_new.T.conjugate() @ mag_loc_op[i] @ psi_new).data[0].real)
    mag_exact_loc.append(mag_exact)

    # total
    mag = (psi_new.T.conjugate() @ mag_tot_op @ psi_new).data
    mag_exact_tot.append(mag.real)

np.savetxt(f"results/mag_data/mag_exact_tot_L_{L}_delta_{delta}_trott_{trotter_steps}", mag_exact_tot)
np.savetxt(f"results/mag_data/mag_exact_loc_L_{L}_delta_{delta}_trott_{trotter_steps}", mag_exact_loc)

for delta, trotter_steps in zip(deltas,trotter_step_list):
    chain = MPS(L=L, d=2, model='Ising', chi=chi, h=h_t, eps=1e-5, J=1)
    chain._random_state(seed=3, chi=chi)
    chain.canonical_form()
    chain.sweeping(trunc=True, n_sweeps=2)
    chain.flipping_mps()
    mag_mps_tot, mag_mps_loc, overlap, errors = chain.variational_mps_evolution(trotter_steps=trotter_steps, delta=delta, h_ev=h_ev, fidelity=True)

    np.savetxt(f"results/mag_data/mag_mps_tot_L_{L}_delta_{delta}_chi_{chi}", mag_mps_tot)
    np.savetxt(f"results/mag_data/mag_mps_loc_L_{L}_delta_{delta}_chi_{chi}", mag_mps_loc)
    np.savetxt(f"results/fidelity_data/fidelity_L_{L}_delta_{delta}_chi_{chi}", overlap)


# %%
# visualization

# total
plt.title(f"Total Magnetization for $\chi = {chi}$ ;" + " $h_{ev} =$" + f"{h_ev}")
colors = create_sequential_colors(num_colors=len(trotter_step_list), colormap_name='viridis')
i = 0
for delta,trotter_steps in zip(deltas,trotter_step_list):
    mag_mps_tot = np.loadtxt(f"results/mag_data/mag_mps_tot_L_{L}_delta_{delta}_chi_{chi}")
    plt.scatter(delta*np.arange(trotter_steps+1), mag_mps_tot, s=25, marker='o', alpha=0.8, facecolors='none', edgecolors=colors[i], label=f"mps: $\delta={delta}$")
    i += 1
plt.plot(delta*np.arange(trotter_steps+1), mag_exact_tot, color='indianred', label=f"exact: $L={L}$")
plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()

# Local data
data1 = mag_exact_loc
data2 = mag_mps_loc
title1 = "Exact quench (local mag)"
title2 = f"MPS quench (local mag) $\chi={chi}$"
plot_side_by_side(data1=data1, data2=data2, cmap='seismic', title1=title1, title2=title2)

# fidelity
plt.title("Fidelity $\left<\psi_{MPS}(t)|\psi_{exact}(t)\\right>$: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
i = 0
for delta,trotter_steps in zip(deltas,trotter_step_list):
    fidelity = np.loadtxt(f"results/fidelity_data/fidelity_L_{L}_delta_{delta}_chi_{chi}")
    plt.scatter(delta*np.arange(trotter_steps+1), fidelity, s=20, marker='o', alpha=0.7, facecolors='none', edgecolors=colors[i], label=f"mps: $\delta={delta}$")
    i += 1
plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()

# entropy
plt.title("Middle Chain Entanglement Entropy: " + f"$\chi = {chi}$; $h_{{t-ev}} = {h_ev}$")
i = 0
for delta,trotter_steps in zip(deltas,trotter_step_list):
    entropy_chi = [0]
    for trott in range(trotter_steps):
        schmidt_vals = np.loadtxt(f"results/bonds_data/schmidt_values_middle_chain_Ising_L_{L}_chi_{chi}_trotter_step_{trott}_delta_{delta}")
        entropy = von_neumann_entropy(schmidt_vals)
        entropy_chi.append(entropy)
    plt.scatter(delta*np.arange(trotter_steps+1), entropy_chi, s=20, marker='o', alpha=0.7, facecolors='none', edgecolors=colors[i], label=f"mps: $\delta={delta}$")
    i += 1
plt.ylabel("entanglement von neumann entropy $(S_{\chi})$")
plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()
# %%
