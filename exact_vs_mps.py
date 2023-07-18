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
# exact: h_t = 0 --> h_t =0.3
L = 11
delta = 0.6
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
H = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=0)
e, v = np.linalg.eig(H)
psi = v[:,0]
flip = single_site_op(op=X, site=L // 2 + 1, L=L)
psi = flip @ psi
H_ev = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=0.5)
U = expm(-1j*delta*H_ev)
U_new = truncation(array=U, threshold=1e-16)
U_new=csr_matrix(U_new)
# %% 
# exact magnetization
magnetization = [single_site_op(op=Z, site=i, L=L) for i in range(1,L+1)]
mag_exact = []
for i in range(L):
    mag_exact.append(psi.T.conjugate() @ magnetization[i] @ psi)
print("magnetization exact:\n", mag_exact)
# %%
# mps: h_t = 0 --> h_t = 0.3
spin = MPS(L=L, d=2, model="Ising", chi=32, J=1, h=0, eps=0)
spin._random_state(seed=3, chi=32)
spin.canonical_form()
energies = spin.sweeping(trunc=True)
spin.flipping_mps()
# %%
# mps magnetization
mag_init = spin.mps_local_exp_val(op=Z)
print(mag_init)
# %%
# time evolution exact vs mps
psi_new = psi
mag_exact_tot = []
mag_mpo_tot = []
trotter_steps = 8
overlap = []
fidelity = True
overlap_init = []
for T in range(trotter_steps):
    psi_new = U_new @ psi_new
    mag_exact = []
    for i in range(L):
        mag_exact.append((psi_new.T.conjugate() @ magnetization[i] @ psi_new).real)
    print(f"----- trotter step {T} --------")
    mag_exact_tot.append(mag_exact)
    print(f"Bond dim: {spin.sites[spin.L//2].shape[0]}")
    spin.mpo_Ising_time_ev(delta=delta, h_ev=0.3, J_ev=1)
    spin.mpo_to_mps()
    # spin.save_sites()
    mag_mpo_tot.append(spin.mps_local_exp_val(op=Z))
    if fidelity:
        psi_new_mpo = mps_to_vector(spin.sites)
        overlap.append(psi_new.T.conjugate() @ psi_new_mpo)
        overlap_init.append(psi_new.T.conjugate() @ psi)
# %%
# visualization
mag_mpo_tot = [np.real(mag_chain) for mag_chain in mag_mpo_tot]
plt.title(f"Exact: $\delta = {delta}$")
plt.imshow(mag_exact_tot, cmap="seismic", vmin=-1, vmax=1, aspect=1)
plt.show()
plt.title(f"MPS: $\delta = {delta}$")
plt.imshow(mag_mpo_tot, cmap="seismic", vmin=-1, vmax=1, aspect=1)
plt.show()
if fidelity:
    error = [1-delta**2*ovrlp for ovrlp in overlap_init]
    plt.title(f"Fidelity $\left<\psi_{{exact}}(t)|\psi_{{MPS}}(t)\\right>$ for $\delta = {delta}$")
    plt.plot(np.abs(overlap), 'o')
    plt.plot(error, '--')
    # plt.hlines(y=error, xmin=0, xmax=trotter_steps-1)
    plt.show()
# %%
# overlaps varying delta
deltas = [0.001, 0.01, 0.1, 0.6]
overlap_deltas = []
trotter_steps = 8
for delta in deltas:
    print(f"----- delta {delta} --------")
    psi_new = psi
    U = expm(-1j*delta*H_ev)
    U_new = truncation(array=U, threshold=1e-16)
    U_new=csr_matrix(U_new)
    spin = MPS(L=L, d=2, model="Ising", chi=32, J=1, h=0, eps=0)
    spin._random_state(seed=3, chi=32)
    spin.canonical_form(trunc=True)
    energies = spin.sweeping(trunc=True)
    spin.flipping_mps()
    overlap = []
    for T in range(trotter_steps):
        print(f"----- trotter step {T} --------")
        psi_new = U_new @ psi_new
        spin.mpo_Ising_time_ev(delta=delta, h_ev=0.3, J_ev=1)
        spin.mpo_to_mps()
        psi_new_mpo = mps_to_vector(spin.sites)
        overlap.append(psi_new.T.conjugate() @ psi_new_mpo)
    overlap_deltas.append(overlap)
# %%
plt.title("Fidelity $\left<\psi_{exact}(t)|\psi_{MPS}(t)\\right>$")
num_colors = len(deltas)
colormap_name = 'viridis'
colors = create_sequential_colors(num_colors, colormap_name)
i = 0
for overlap, delta in zip(overlap_deltas, deltas):
    plt.plot(np.abs(overlap), color=colors[i], label=f"delta: {delta}")
    plt.yscale('log')
    plt.legend()
    plt.hlines(y=1-delta**3, linestyles="--", xmin=0, xmax=7, colors=colors[i])
    i += 1
plt.show()
# %%
# compare U exact with U mpo
L = 4
delta = 0.01
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
H_ev = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=0.3)
U = expm(-1j*delta*H_ev)
U_new = truncation(array=U, threshold=1e-16)
U_new = csr_matrix(U_new)
spin = MPS(L=L, d=2, model="Ising", chi=32, J=1, h=0, eps=0)
spin.mpo_Ising_time_ev(delta=delta, h_ev=0.3, J_ev=1)
U_mpo_list = spin.w
U_mpo_matrix = mpo_to_matrix(U_mpo_list)
U_mpo_new = truncation(array=U_mpo_matrix, threshold=1e-16)
U_mpo_new = csr_matrix(U_mpo_new)

# %%
# only exact time ev
L = 11
delta = 0.06
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
H = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=0)
e, v = np.linalg.eig(H)
psi = v[:,0]
flip = single_site_op(op=X, site=L // 2 + 1, L=L)
psi = flip @ psi
h_t = 2
H_ev = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=h_t)
U = expm(-1j*delta*H_ev)
U_new = truncation(array=U, threshold=1e-16)
U_new=csr_matrix(U_new)
psi_new = psi
mag_exact_tot = []
mag_mpo_tot = []
trotter_steps = 100
overlap = []
fidelity = True
overlap_init = []
for T in range(trotter_steps):
    psi_new = U_new @ psi_new
    mag_exact = []
    for i in range(L):
        mag_exact.append((psi_new.T.conjugate() @ magnetization[i] @ psi_new).real)
    print(f"----- trotter step {T} --------")
    mag_exact_tot.append(mag_exact)
    overlap_init.append(psi_new.T.conjugate() @ psi)
# %%
plt.title(f"Exact: $\delta = {delta}$; $h_{{t-ev}} = {h_t}$")
plt.imshow(mag_exact_tot, cmap="seismic", vmin=-1, vmax=1, aspect=0.1)
plt.show()

plt.title("Fidelity $\left<\psi_{exact}(t)|\psi_{exact}(t=0)\\right>$: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_t}$")
plt.plot(overlap_init)
plt.show()
# %%
