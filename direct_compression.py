# %%
from mps_class_v9 import MPS
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# %%
def mixed_canonical(mps):
    s_init = np.array([1])
    psi = np.diag(s_init)
    
    sites = mps.sites
    for i in range(mps.L - 1, mps.L//2, -1):
        new_site = ncon(
            [sites[i], psi],
            [
                [-1,-2,1],
                [1,-3],
            ],
        )
        u, s, v = np.linalg.svd(
            new_site.reshape(new_site.shape[0], mps.d * new_site.shape[2]),
            full_matrices=False,
        )
        bond_r = v.shape[1] // mps.d
        v = v.reshape((v.shape[0], mps.d, bond_r))
        sites[i] = v
        psi = ncon(
            [u, np.diag(s)],
            [
                [-1,1],
                [1,-2],
            ],
        )
    s_init = np.array([1])
    psi = np.diag(s_init)

    for i in range(mps.L//2+1):
        new_site = ncon(
            [psi, sites[i]],
            [
                [-1,1],
                [1,-2,-3],
            ],
        )
        u, s, v = np.linalg.svd(
            new_site.reshape(new_site.shape[0] * mps.d, new_site.shape[2]),
            full_matrices=False,
        )
        bond_l = u.shape[0] // mps.d
        u = u.reshape(bond_l, mps.d, u.shape[1])
        sites[i] = u
        psi = ncon(
            [np.diag(s), v],
            [
                [-1,1],
                [1,-2],
            ],
        )
    return mps

def trunc_mps(mps, chi):
    mps.canonical_form(trunc=True, svd_direction="left")
    # mps.canonical_form(trunc=True, svd_direction="right")
    mps._compute_norm(1)
    return mps

def exact_magnetization_loc(psi, magnetization):
    local_mag = []
    for i in range(L):
        local_mag.append((psi.T.conjugate() @ magnetization[i] @ psi).real)
    return local_mag

def exact_magnetization_tot(psi, magnetization):
    total_mag = psi.T.conjugate() @ magnetization @ psi
    return total_mag

def exact_evolution(L, T, psi_0): # , mag_tot, mag_loc
    print(f"Trotter step: {T}")
    U_ev = exact_evolution_operator(L=L, h_t=h_ev, delta=delta, trotter_step=T)
    psi = U_ev @ psi_0
    return psi

def direct_TEBD(trotter_steps, delta, h_ev, J_ev, chi_max, mag_mps_loc, mag_mps_tot, trunc, fidelity):
    overlap = []
    ent_ent = []
    for T in range(trotter_steps):
        print(f"Trotter step: {T}")
        chain.mpo_Ising_time_ev(delta=delta, h_ev=h_ev, J_ev=J_ev)
        new_chain = chain.mpo_to_mps()
        if trunc:
            new_chain = trunc_mps(chain, chi=chi_max)
        mag_mps_loc.append(np.real(new_chain.mps_local_exp_val(op=Z)))
        new_chain.order_param_Ising(op=Z)
        mag_mps_tot.append(new_chain.mpo_first_moment().real)
        u,s,v = np.linalg.svd(new_chain.sites[new_chain.L//2].reshape((new_chain.sites[new_chain.L//2].shape[0]*new_chain.d,new_chain.sites[new_chain.L//2].shape[2])))
        ent_ent.append(von_neumann_entropy(s))
        if fidelity:
            psi_mps = mps_to_vector(new_chain.sites)
            psi = exact_evolution(chain.L, T, psi_0)
            overlap.append(psi_mps.T.conjugate() @ psi)
    return mag_mps_loc, mag_mps_tot, overlap, ent_ent

def plot_three_colormaps(arr_1, arr_2, arr_3, cmap, aspect):
    # Create a sample data for demonstration
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the main figure with a grid layout
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # Plot data with the first colormap in the first row, first column
    ax1 = plt.subplot(gs[0, 0])
    im1 = ax1.imshow(arr_1, cmap=cmap, aspect=aspect)
    ax1.set_title('Exact')

    # Plot data with the second colormap in the first row, second column
    ax2 = plt.subplot(gs[0, 1])
    im2 = ax2.imshow(arr_2, cmap=cmap, aspect=aspect)
    ax2.set_title('MPS: L compression')

    # Plot data with the third colormap in the second row, centered
    ax3 = plt.subplot(gs[1, :])
    im3 = ax3.imshow(arr_3, cmap=cmap, aspect=aspect, vmin=-0.5, vmax=0.5)
    ax3.set_title('Difference')

    # Add colorbars to each subplot
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)

    # Adjust layout and display the plot
    # plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.6)
    plt.show()
# %%
L = 9
d = 2
h_0 = 0
J = 1
chis = [2]
Z = np.array([[1,0],[0,-1]])
# exact initial state and observables
psi_0 = exact_initial_state(L=L, h_t=0)
mag_tot = H_loc(L=L, op=Z)
magnetization = [single_site_op(op=Z, site=i, L=L) for i in range(1,L+1)]
# attempt of loop
trotter_steps = [5,6,7,8]
deltas = [1/T for T in trotter_steps]
chi = 2
h_ev = 0.5
trunc = False
mag_mps_tot_chi = []
mag_mps_loc_chi = []
overlap_chi = []
entr_chi = []
for trott, delta in zip(trotter_steps, deltas):
    # initializing the chain
    chain = MPS(L=L, d=d, model="Ising", chi=2, h=h_0, J=J, eps=0)
    chain._random_state(seed=7)
    chain.canonical_form()
    chain.sweeping(trunc=True)
    print(np.real(chain.mps_local_exp_val(op=Z)))
    chain.flipping_mps()
    chain.flip_all_mps()
    psi_0_mps = mps_to_vector(chain.sites)
    # computing expectation values before trotter
    mag_mps_loc = []
    mag_mps_tot = []
    print(np.real(chain.mps_local_exp_val(op=Z)))
    mag_mps_loc.append(np.real(chain.mps_local_exp_val(op=Z)))
    chain.order_param_Ising(op=Z)
    mag_mps_tot.append(chain.mpo_first_moment().real)
    chi_max = chi
    chain.chi = chi
    # trotter evolution for a specific chi
    mag_mps_loc, mag_mps_tot, overlap, ent_ent = direct_TEBD(trott, delta, h_ev, J, chi_max, mag_mps_loc, mag_mps_tot, trunc, fidelity=True)
    mag_mps_tot_chi.append(mag_mps_tot)
    mag_mps_loc_chi.append(mag_mps_loc)
    overlap_chi.append(overlap)
    entr_chi.append(ent_ent)
# %%
# ============================
total_mag_tr = []
local_mag_tr = []
for delta, trott in zip(deltas, trotter_steps):
    total_mag = []
    local_mag = []
    total_mag.append(exact_magnetization_tot(psi=psi_0, magnetization=mag_tot))
    local_mag.append(exact_magnetization_loc(psi=psi_0, magnetization=magnetization))
    for T in range(trott):
        print(f"Trotter step: {T}")
        U_ev = exact_evolution_operator(L=L, h_t=h_ev, delta=delta, trotter_step=T)
        psi = U_ev @ psi_0
        total_mag.append(exact_magnetization_tot(psi=psi, magnetization=mag_tot))
        local_mag.append(exact_magnetization_loc(psi=psi, magnetization=magnetization))
    total_mag_tr.append(total_mag)
    local_mag_tr.append(local_mag)
overlap_init = psi_0_mps.T.conjugate() @ psi_0
# %%
# ============================
# visualization
# ============================
# Total magnetization compression
plt.title("Magnetization order parameter")
plt.xlabel("time $(t = \delta · N)$", fontsize=14)
plt.ylabel("Expectation value $\quad$ $\left<\sum_i \sigma_i^z\\right>$")
plt.plot(delta*np.arange(trotter_steps+1), total_mag, label=f"exact: $L = {L}$")
for mag_mps_tot, chi in zip(mag_mps_tot_chi, chis):
    plt.plot(delta*np.arange(trotter_steps+1), mag_mps_tot, 'o', label=f"mps: $\chi = {chi}$")
plt.legend()
plt.show()
# %%
# ============================
# visualization
# ============================
# Total magnetization Error compression
plt.title("Magnetization order parameter Error")
plt.xlabel("time $(t = \delta · N)$", fontsize=14)
plt.ylabel("Error")
for mag_mps_tot, chi in zip(mag_mps_tot_chi, chis):
    plt.plot(delta*np.arange(trotter_steps+1), np.abs(np.asarray(total_mag)-np.asarray(mag_mps_tot))/np.asarray(total_mag), 'o', label=f"mps: $\chi = {chi}$")
plt.legend()
plt.show()
# %%
# Total magnetization no compression different deltas
plt.title("Magnetization order parameter")
plt.xlabel("time $(t = \delta · N)$", fontsize=14)
plt.ylabel("Expectation value $\quad$ $\left<\sum_i \sigma_i^z\\right>$")
for mag_mps_tot, total_mag, delta, trott in zip(mag_mps_tot_chi, total_mag_tr, deltas, trotter_steps):
    plt.plot(delta*np.arange(trott+1), total_mag, label=f"exact: $trotter = {trott}$")
    plt.plot(delta*np.arange(trott+1), mag_mps_tot, 'o', label=f"mps: $trotter = {trott}$")
plt.legend()
plt.show()
# %%
# Total magnetization NO compression
chi = chain.sites[L//2].shape[0]
plt.title("Magnetization order parameter")
plt.xlabel("time $(t = \delta · N)$", fontsize=14)
plt.ylabel("Expectation value $\quad$ $\left<\sum_i \sigma_i^z\\right>$")
plt.plot(delta*np.arange(trotter_steps+1), total_mag, label=f"exact: $L = {L}$")
plt.plot(delta*np.arange(trotter_steps+1), mag_mps_tot_chi[-1], 'o', label=f"mps: $\chi = {chi}$")
plt.legend()
plt.show()
# %%
# Total magnetization error NO compression
chi = chain.sites[L//2].shape[0]
plt.title("Magnetization order parameter")
plt.xlabel("time $(t = \delta · N)$", fontsize=14)
plt.ylabel("relative error $\quad$ $(\left<M_{exact}\\right> - \left<M_{MPS}\\right>)/\left<M_{exact}\\right>$")
plt.plot(delta*np.arange(trotter_steps+1), (np.asarray(total_mag)-np.asarray(mag_mps_tot_chi[-1]))/np.asarray(total_mag), label=f"difference exact - $\chi={chi}$:  $L = {L}$")
plt.legend()
plt.show()
# %%
# Local magnetization compression
cmap = 'seismic'
arr_1 = local_mag
arr_2 = mag_mps_loc_chi[-1]
arr_3 = np.asarray(arr_1) - np.asarray(arr_2)
aspect = 0.1
plot_three_colormaps(arr_1, arr_2, arr_3, cmap, aspect)

# %%
# Fidelity compression
from checks import commutator
from scipy.sparse.linalg import norm
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
A = csr_matrix(H_loc(L=L, op=X))
B = csr_matrix(H_int(L=L, op=Z))
delta_opt = 0.1
error_2 = [1 - T*delta_opt**3 for T in range(trotter_steps+1)]
plt.title("Fidelities with changing bond dimension")
for overlap, chi in zip(overlap_chi, chis):
    overlap = [overlap_init] + overlap
    plt.plot(np.linspace(0,t,trotter_steps+1), np.abs(overlap), 'o', label=f"$\chi = {chi}, \delta = {delta:.2f}$")
# plt.plot(np.linspace(0,1,trotter_steps+1), error_2, color='red', linestyle=':', label=f"trotter error for $\delta = {delta_opt}$")
plt.xlabel("time $(t = \delta · N)$", fontsize=14)
plt.ylabel("$\left<\psi_{MPS} (t)| \psi_{exact} (t)\\right>$", fontsize=14)
# plt.yscale('log')
# plt.ylim((0.9,1.01))
# plt.savefig('figures/fidelity_L_compression', transparent=True)
plt.legend()
plt.show()

# %%
# Fidelity NO compression
overlap = overlap_chi[-1]
overlap = [overlap_init] + overlap
# %%
overlap_symm = [1.0000000000000002,
 (0.8214598035063064-0.5306764685238107j),  
 (0.8219851579008668-0.5295464666971555j),
 (0.8228594558968224-0.5270196114890093j),
 (0.8239929856458802-0.5234175350211067j),
 (0.825260689130872-0.5192251638384628j),
 (0.8265316771641906-0.5149710319426287j),
 (0.8276934782218862-0.5111089764747592j)]
from scipy.sparse.linalg import norm
from checks import commutator
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
A = csr_matrix(H_loc(L=L, op=X))
B = csr_matrix(H_int(L=L, op=Z))
comm = commutator(A,B)
error_2 = [1 - T*delta**3 for T in range(trotter_steps+1)]
error = [1 - T*delta**2 for T in range(trotter_steps+1)]
plt.title("Fidelity: No compression")
plt.plot(np.linspace(0,1,trotter_steps+1), np.abs(overlap), 'o', label=f"$\chi = {chi}$")
plt.plot(np.linspace(0,1,trotter_steps+1), np.abs(overlap_symm), 'o', label=f"local op divided")
plt.plot(np.linspace(0,1,trotter_steps+1), error_2, color='red', linestyle=':', label=f"trotter error 2º order at $t={delta*trotter_steps}$")
# plt.plot(np.linspace(0,1,trotter_steps+1), error, color='red', linestyle='--', label=f"trotter error 1º order at $t={delta*trotter_steps}$")
plt.xlabel("time $(t = \delta · N)$", fontsize=14)
plt.ylabel("$\left<\psi_{MPS} (t)| \psi_{exact} (t)\\right>$", fontsize=14)
# plt.yscale('log')
# plt.ylim(0.9995, 1.000)
plt.legend(loc='best', fontsize=14)
plt.show()

# %%
# entanglement compression
plt.title("Entanglement entropy")
plt.xlabel("time $(t = \delta · N)$", fontsize=14)
plt.ylabel("$\sum_i \\alpha_i^2 log_2(\\alpha_i^2)$")
for entr, chi in zip(entr_chi, chis):
    plt.plot(delta*np.arange(trotter_steps), np.abs(entr), marker='o', linestyle='-', label=f"mps: $\chi = {chi}$")
plt.legend()
plt.show()
# %%
# entanglement NO compression
plt.title("Entanglement entropy")
plt.xlabel("time $(t = \delta · N)$", fontsize=14)
plt.ylabel("$\sum_i \\alpha_i^2 log_2(\\alpha_i^2)$")
plt.plot(delta*np.arange(trotter_steps), np.abs(entr_chi[-1]), 'o', label=f"mps: $\chi = {chi}$")
plt.legend()
plt.show()
# %%
from checks import commutator
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
A = csr_matrix(H_loc(L=L, op=X))
B = csr_matrix(H_int(L=L, op=Z))
print(1-delta**2*norm(commutator(A,B)))

# %%
errors = [(t**3)/12*norm(commutator(B,commutator(B,A)))+(t**3)/24*norm(commutator(A,commutator(A,B))) for t in delta*np.arange(trotter_steps)]
# %%
plt.plot(delta*np.arange(trotter_steps), errors)
# %%



# %%
L = 9
d = 2
h_0 = 0
J = 1
chis = [16, 32]
Z = np.array([[1,0],[0,-1]])
# exact initial state and observables
psi_0 = exact_initial_state(L=L, h_t=0)
mag_tot = H_loc(L=L, op=Z)
magnetization = [single_site_op(op=Z, site=i, L=L) for i in range(1,L+1)]
total_mag = []
local_mag = []
total_mag.append(exact_magnetization_tot(psi=psi_0, magnetization=mag_tot))
local_mag.append(exact_magnetization_loc(psi=psi_0, magnetization=magnetization))
# attempt of loop
t = 10
delta = 0.01
trotter_steps = int(t/delta)
chi = 2
h_ev = 0.5
trunc = True
mag_mps_tot_chi = []
mag_mps_loc_chi = []
overlap_chi = []
entr_chi = []
for chi in chis:
    # initializing the chain
    chain = MPS(L=L, d=d, model="Ising", chi=2, h=h_0, J=J, eps=0)
    chain._random_state(seed=7)
    chain.canonical_form()
    chain.sweeping(trunc=True)
    print(np.real(chain.mps_local_exp_val(op=Z)))
    chain.flipping_mps()
    chain.flip_all_mps()
    psi_0_mps = mps_to_vector(chain.sites)
    # computing expectation values before trotter
    mag_mps_loc = []
    mag_mps_tot = []
    print(np.real(chain.mps_local_exp_val(op=Z)))
    mag_mps_loc.append(np.real(chain.mps_local_exp_val(op=Z)))
    chain.order_param_Ising(op=Z)
    mag_mps_tot.append(chain.mpo_first_moment().real)
    chi_max = chi
    chain.chi = chi
    print("here")
    # trotter evolution for a specific chi
    mag_mps_loc, mag_mps_tot, overlap, ent_ent = direct_TEBD(trotter_steps, delta, h_ev, J, chi_max, mag_mps_loc, mag_mps_tot, trunc, fidelity=True)
    mag_mps_tot_chi.append(mag_mps_tot)
    mag_mps_loc_chi.append(mag_mps_loc)
    overlap_chi.append(overlap)
    entr_chi.append(ent_ent)
# %%
# ============================
for T in range(trotter_steps):
    print(f"Trotter step: {T}")
    U_ev = exact_evolution_operator(L=L, h_t=h_ev, delta=delta, trotter_step=T)
    psi = U_ev @ psi_0
    total_mag.append(exact_magnetization_tot(psi=psi, magnetization=mag_tot))
    local_mag.append(exact_magnetization_loc(psi=psi, magnetization=magnetization))
overlap_init = psi_0_mps.T.conjugate() @ psi_0
# %%
