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
trotter_steps = 100
t = 10
delta = t/trotter_steps
h_t = 0
h_ev = 0.3
L = 9
n_sweeps = 4

# exact
mag_exact_loc, mag_exact_tot = exact_evolution(L=L, h_t=h_t, h_ev=h_ev, delta=delta, trotter_steps=trotter_steps)

np.savetxt(f"results/mag_data/mag_exact_tot_L_{L}_delta_{delta}_trott_{trotter_steps}", mag_exact_tot)
np.savetxt(f"results/mag_data/mag_exact_loc_L_{L}_delta_{delta}_trott_{trotter_steps}", mag_exact_loc)

# variational truncation mps
for i in range(1,L//2+1):
    chi = 2**i
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
plt.title(f"Total Magnetization for $\delta = {delta}$ ;" + " $h_{ev} =$" + f"{h_ev}")
colors = create_sequential_colors(num_colors=len(range(2,L//2+2)), colormap_name='viridis')
for i in range(1,L//2+1):
    chi = 2**i
    mag_mps_tot = np.loadtxt(f"results/mag_data/mag_mps_tot_L_{L}_delta_{delta}_chi_{chi}")
    plt.scatter(delta*np.arange(trotter_steps+1), mag_mps_tot, s=25, marker='o', alpha=0.8, facecolors='none', edgecolors=colors[i-1], label=f"mps: $\chi={chi}$")

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
for i in range(1,L//2+1):
    chi = 2**i
    fidelity = np.loadtxt(f"results/fidelity_data/fidelity_L_{L}_delta_{delta}_chi_{chi}")
    plt.scatter(delta*np.arange(trotter_steps+1), fidelity, s=20, marker='o', alpha=0.7, facecolors='none', edgecolors=colors[i-1], label=f"mps: $\chi={chi}$")

plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()

# entropy
plt.title("Middle Chain Entanglement Entropy: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
for i in range(1,L//2+1):
    chi = 2**i
    entropy_chi = [0]
    for trott in range(trotter_steps):
        schmidt_vals = np.loadtxt(f"results/bonds_data/schmidt_values_middle_chain_Ising_L_{L}_chi_{chi}_trotter_step_{trott}_delta_{delta}")
        entropy = von_neumann_entropy(schmidt_vals)
        entropy_chi.append(entropy)
    plt.scatter(delta*np.arange(trotter_steps+1), entropy_chi, s=20, marker='o', alpha=0.7, facecolors='none', edgecolors=colors[i-1], label=f"mps: $\chi={chi}$")

plt.ylabel("entanglement von neumann entropy $(S_{\chi})$")
plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()
# %%
