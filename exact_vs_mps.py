#%%
from mps_class_v9 import MPS
import numpy as np
from utils import *
from ncon import ncon
import scipy
from scipy.linalg import expm
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
# ---------------------------------------------------------
# mps: h_t = 0 --> h_t = h_ev
# ---------------------------------------------------------
# time evolution to h_t = h_ev
L = 9
trotter_steps = 10
delta = 0.2
h_ev = 0.5
fidelity = True
chis = [2, 8, 16]
mag_chis = []
overlap_chis = []
for chi in chis:
    # initialize the chain in h_t = 0
    spin = MPS(L=L, d=2, model="Ising", chi=chi, J=1, h=0, eps=0)
    spin._random_state(seed=3, chi=chi)
    spin.canonical_form()
    energies = spin.sweeping(trunc=True)
    spin.flipping_mps()
    # time evolution and expectation values computation
    mag_mpo_tot, overlap = spin.direct_mpo_evolution(trotter_steps=trotter_steps, delta=delta, h_ev=h_ev, J_ev=1, fidelity=fidelity)
    mag_chis.append(mag_mpo_tot)
    overlap_chis.append(overlap)

# %%
L = 9
trotter_steps = 10
Z = np.array([[1,0],[0,-1]])
delta = 0.2
h_ev = 0.5

psi_0 = exact_initial_state(L=L, h_t=0)
mag_tot = H_loc(L=L, op=Z)
magnetization = [single_site_op(op=Z, site=i, L=L) for i in range(1,L+1)]
mag_exact_tot = []
mag_exact_loc = []
for T in range(trotter_steps):
    U_ev = exact_evolution_operator(L=L, h_t=h_ev, delta=delta, trotter_step=T)
    psi = U_ev @ psi_0
    mag = psi.T.conjugate() @ mag_tot @ psi
    mag_exact_tot.append(mag)
    local_mag = []
    for i in range(L):
        local_mag.append((psi.T.conjugate() @ magnetization[i] @ psi).real)
    mag_exact_loc.append(local_mag)
# %%
# visualization of total magnetizations exact vs mps
for mag_mpo, chi in zip(mag_chis, chis):
    plt.title("MPS total magnetization: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
    plt.ylabel("magnetization")
    plt.xlabel("time (s)")
    plt.plot(delta*np.arange(trotter_steps), mag_mpo, linestyle="--", label=f"$\chi = {chi}$")

plt.plot(delta*np.arange(trotter_steps), mag_exact_tot, label=f"exact: $L={L}$")
plt.legend()
plt.show()
# %%
# fidelity visualization with mps of largest chi
if fidelity:
    
    for overlap, chi in zip(overlap_chis, chis):
        plt.plot(np.abs(overlap), label=f'$\chi = {chi}')
    plt.title("Fidelity $\left<\psi_{MPS}(t)|\psi_{exact}(t)\\right>$: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
    plt.show()

# visualization of exact local magnetization
plt.title(f"Exact: $\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
plt.imshow(mag_exact_loc, cmap="seismic", vmin=-1, vmax=1, aspect=0.1)
plt.show()
# %%
# visualization of mps local magnetization
for mag_loc, chi in zip(mag_chis, chis):
    plt.title(f"Exact: $\delta = {delta}$; $h_{{t-ev}} = {h_ev}, \chi = {chi}$")
    plt.imshow(mag_loc, cmap="seismic", vmin=-1, vmax=1, aspect=0.1)
    plt.show()
# %%
# check the Schmidt values in the middle of the time evolved chain
site = spin.L//2
middle_site = spin.sites[site]
middle_site_new = middle_site.reshape((2*2**trotter_steps,2**trotter_steps))
u,s,v = np.linalg.svd(middle_site_new, full_matrices=False)
e_tol = 1e-15
condition = s >= e_tol
s_trunc = np.extract(condition, s)
print(f"Number of non-zero Schmidt values: {len(s_trunc)}")
print(f"Smallest Schmidt value:           {s[-1]}")
print(f"Smallest Schmidt value truncated: {s_trunc[-1]}")

# %%
# ---------------------------------------------------------
# exact: h_t = 0 --> h_t = h_ev
# ---------------------------------------------------------
L = 11
delta = 0.6
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
H = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=0)
e, v = np.linalg.eig(H)
psi = v[:,0]
flip = single_site_op(op=X, site=L // 2 + 1, L=L)
psi = flip @ psi
h_t = 0.1
H_ev = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=h_t)
U = expm(-1j*delta*H_ev)
U_new = truncation(array=U, threshold=1e-16)
U_new = csr_matrix(U_new)
psi_new = psi
mag_exact_tot = []
mag_mpo_tot = []
trotter_steps = 6
overlap = []
L = 11
spin = MPS(L=L, d=2, model="Ising", chi=32, J=1, h=0, eps=0)
spin._random_state(seed=3, chi=32)
spin.canonical_form()
energies = spin.sweeping(trunc=True)
spin.flipping_mps()
magnetization = [single_site_op(op=Z, site=i, L=L) for i in range(1,L+1)]
for T in range(trotter_steps):
    psi_new = U_new @ psi_new
    mag_exact = []
    print(f"Bond dim: {spin.sites[spin.L//2].shape[0]}")
    spin.mpo_Ising_time_ev(delta=delta, h_ev=h_ev, J_ev=1)
    spin.mpo_to_mps()
    for i in range(L):
        mag_exact.append((psi_new.T.conjugate() @ magnetization[i] @ psi_new).real)
    print(f"----- trotter step {T} --------")
    mag_exact_tot.append(mag_exact)
    # psi_new_mpo = mps_to_vector(spin.sites)
    # overlap.append(psi_new.T.conjugate() @ psi_new_mpo)

# %%
plt.title(f"Exact: $\delta = {delta}$; $h_{{t-ev}} = {h_t}$")
plt.imshow(mag_exact_tot, cmap="seismic", vmin=-1, vmax=1, aspect=1)
plt.show()

plt.title("Fidelity $\left<\psi_{exact}(t)|\psi_{MPS}(t)\\right>$: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_t}$")
plt.plot(np.abs(overlap))
plt.hlines(y=1-delta**2, xmin=0, xmax=trotter_steps-1)
plt.ylim((1-delta**2-0.005,1))
plt.show()

#######################
#######################
#######################
#######################
#######################

# %%
# mps: h_t = 0 --> h_t = 0.3
L = 80
spin = MPS(L=L, d=2, model="Ising", chi=64, J=1, h=0, eps=0)
spin._random_state(seed=3, chi=64)
spin.canonical_form()
energies = spin.sweeping(trunc=True)
spin.flipping_mps()

# %%
# time evolution
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
trotter_steps = 8
delta = 0.6
h_ev = 0.25
fidelity = False
if fidelity:
    psi = mps_to_vector(spin.sites)
mag_mpo_tot = []
entropy = []
for T in range(trotter_steps):
    print(f"Bond dim: {spin.sites[spin.L//2].shape[0]}")
    spin.mpo_Ising_time_ev(delta=delta, h_ev=h_ev, J_ev=1)
    spin.mpo_to_mps()
    # spin.save_sites()
    mag_mpo_tot.append(np.real(spin.mps_local_exp_val(op=Z)))
    site = spin.L//2
    middle_site = spin.sites[site]
    middle_site_new = middle_site.reshape((2*middle_site.shape[0],middle_site.shape[2]))
    u,s,v = np.linalg.svd(middle_site_new, full_matrices=False)
    e_tol = 1e-15
    condition = s >= e_tol
    s_trunc = np.extract(condition, s)
    entropy.append(von_neumann_entropy(s))
    if fidelity:
        psi_new_mpo = mps_to_vector(spin.sites)
        overlap.append(psi_new_mpo.T.conjugate() @ psi)
       
# %%
# visualization
plt.title(f"MPS: $\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
plt.imshow(mag_mpo_tot, cmap="seismic", vmin=-1, vmax=1, aspect=3)
plt.show()
if fidelity:
    plt.title("Fidelity $\left<\psi_{exact}(t)|\psi_{exact}(t=0)\\right>$: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
    plt.plot(overlap)
    plt.show()
plt.title("Entropy MPS: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
plt.plot(np.log(np.abs(entropy)))
plt.show()

# %%
# total mag

# %%
