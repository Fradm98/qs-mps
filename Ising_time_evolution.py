#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import packages
from mps_class_v9 import MPS
from utils import *
import matplotlib.pyplot as plt
from ncon import ncon
import scipy
from scipy.sparse import csr_array
import time


# In[2]:

# exact state and evolution
L = 9
h_t = 0
h_ev = 0.3
t = 5
trotter_steps = 50
delta = t/trotter_steps
psi_exact = exact_initial_state(L=L, h_t=h_t, h_l=1e-7).reshape(2**L,1)
# we save the exact states to compute the fidelity
exact_states = []
exact_states.append(psi_exact)
# define the local and total magnetization operators
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
    mag_exact.append((psi_exact.T.conjugate() @ mag_loc_op[i] @ psi_exact).data.real)
mag_exact_loc.append(mag_exact)

# total
mag = (psi_exact.T.conjugate() @ mag_tot_op @ psi_exact).data
mag_exact_tot.append(mag.real)

states = []
for trott in range(1,trotter_steps+1):
    # compute the U in the time we are interested in, that is, delta*trott
    psi_new = exact_evolution(L, psi_init=psi_exact, delta=delta, trotter_step=trott, h_t=h_ev)
    # final state after U time evolution
    # psi_new = U @ psi_exact
    states.append(psi_new)
    # exact_states.append(psi_new)
    # compute the total and local magnetization at that time
    
    # local
    mag_exact = []
    for i in range(L):
        mag_exact.append((psi_new.T.conjugate() @ mag_loc_op[i] @ psi_new).data.real)
    print(f"----- trotter step {trott} --------")
    mag_exact_loc.append(mag_exact)

    # total
    mag = (psi_new.T.conjugate() @ mag_tot_op @ psi_new).data
    mag_exact_tot.append(mag.real)


# In[3]:


# visualization of exact evolution
# local
plt.imshow(mag_exact_loc, cmap='seismic', vmin=-1, vmax=1, aspect=0.1)
plt.show()
# total
plt.title(f"Total Magnetization for $\delta = {delta}$ ;" + " $h_{ev} =$" + f"{h_ev}")
plt.plot(delta*np.arange(trotter_steps+1), mag_exact_tot, label=f"exact: $L={L}$")
plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()


# Now we can try to do the same with MPS.
# We need to implement the time evolution and compress the bond dimesion to allow larger chains.
# This procedure works as follows:
# - we apply the time ev mpo (bond dim = w) to an initial state (bond dim = m). To do that we variationally find an mps with bond dimension m' < w*m
# - we minimize a distance measure between the mpo applied on the initial state and the variational compressed state.
# - we take track of the error

# In[4]:

# compression functions
def lin_sys(classe, M, N_eff, site, l_shape, r_shape):
    M_new = M.flatten()
    new_site = scipy.sparse.linalg.spsolve(N_eff, M_new)
    new_site = new_site.reshape((l_shape[0], classe.d, r_shape[0]))
    # print("Comparison with previous state:\n")
    # print(f"Old site:\n{classe.sites[site - 1]}")
    # print(f"New site:\n{new_site}")
    classe.sites[site - 1] = new_site
    return classe

def _braket(ket, bra, w):
    sandwich = ncon([ket,w,bra.conjugate()],[[-1,1,-4],[-2,-5,1,2],[-3,2,-6]])
    return sandwich

def environments(classe, site):
    a = np.array([1])
    E_l = ncon([a,a],[[-1],[-2]])
    E_r = E_l
    env_right = []
    env_left = []

    env_right.append(E_r)
    env_left.append(E_l)
    array = classe.sites
    ancilla_array = classe.ancilla_sites

    for i in range(1, site):
        E_l = ncon(
            [E_l, ancilla_array[i - 1], array[i - 1].conjugate()],
            [
                [1,3],
                [1,2,-1],
                [3,2,-2],
            ],
            )
        env_left.append(E_l)

    for j in range(classe.L, site, -1):
        E_r = ncon(
            [E_r, ancilla_array[j - 1], array[j - 1].conjugate()],
            [
                [1,3],
                [-1,2,1],
                [-3,2,3],
            ],
            )
        env_right.append(E_r)

    classe.env_right = env_right
    classe.env_left = env_left
    return classe

def sandwich_site_ancilla_site(classe=MPS, site=int):
    """
    _compute_norm

    This function computes the norm of our quantum state which is represented in mps.
    It takes the attributes .sites and .bonds of the class which gives us
    mps in canonical form (Vidal notation).

    svd_direction: string - the direction the svd was performed. From left to right
                    is "right" and from right to left is "left"

    """
    array_2 = classe.sites
    array_1 = classe.ancilla_sites
    a = np.array([1])
    env = ncon([a,a,a,a],[[-1],[-2],[-3],[-4]])
    left = env

    for i in range(site-1):
        ten = classe.overlap_sites(array_1=array_1[i], array_2=array_2[i])
        env = ncon([env,ten],[[-1,-2,1,2],[1,2,-3,-4]])
    left = env

    env = ncon([a,a,a,a],[[-1],[-2],[-3],[-4]])
    right = env
    for i in range(classe.L-1, site-1, -1):
        ten = classe.overlap_sites(array_1=array_1[i], array_2=array_2[i])
        env = ncon([ten,env],[[-1,-2,1,2],[1,2,-3,-4]])
    right = env

    ten_site = classe.overlap_sites(array_1=array_1[site - 1], array_2=array_2[site - 1])
    N = ncon([left,ten_site,right],[[-1,-2,1,2],[1,2,3,4],[3,4,-3,-4]])
    N = N[0,0,0,0].real
    print(f"-=-=-= Norm: {N}\n")
    return N

def _N_eff(classe, site):   
    array = classe.sites
    a = np.array([1])
    env = ncon([a,a,a,a],[[-1],[-2],[-3],[-4]])

    for i in range(site-1):
        ten = classe.overlap_sites(array_1=array[i])
        env = ncon([env,ten],[[-1,-2,1,2],[1,2,-3,-4]])
    left = env
    left = ncon([a,a,left],[[1],[2],[1,2,-1,-2]])
    # print(csr_matrix(truncation(left, threshold=1e-15)))
    # plt.title("left")
    # plt.imshow(left.real, vmin=0, vmax=1)
    # plt.show()
    env = ncon([a,a,a,a],[[-1],[-2],[-3],[-4]])
    for i in range(classe.L-1, site-1, -1):
        ten = classe.overlap_sites(array_1=array[i])
        env = ncon([ten,env],[[-1,-2,1,2],[1,2,-3,-4]])
    right = env
    right = ncon([right,a,a],[[-1,-2,1,2],[1],[2]])
    # print(csr_matrix(truncation(right, threshold=1e-15)))
    # plt.title("right")
    # plt.imshow(right.real, vmin=0, vmax=1)
    # plt.show()
    kron = np.eye(2)
    # N = ncon([left,kron,right],[[-1,-4],[-2,-5],[-3,-6]]).reshape((classe.env_left[-1].shape[2]*classe.d*classe.env_right[-1].shape[2],classe.env_left[-1].shape[2]*classe.d*classe.env_right[-1].shape[2]))
    N = ncon([left,kron,right],[[-1,-4],[-2,-5],[-3,-6]]).reshape((left.shape[0]*classe.d*right.shape[0],left.shape[1]*classe.d*right.shape[1]))
    return N, left.shape, right.shape

def compute_M(classe, site, rev=False):
        """
        _compute_M

        This function computes the rank-3 tensor, in a specific site,
        given by the contraction of our variational state (phi) saved in classe.sites,
        and the uncompressed state (psi) saved in classe.ancilla_sites.

        site: int - site where to execute the tensor contraction

        """
        # array_1 = classe.ancilla_sites
        # array_2 = classe.sites
        # w = classe.w
        # if rev:
        #     array_1 = classe.sites
        #     array_2 = classe.ancilla_sites
        # a = np.array([1])
        # env = ncon([a,a,a],[[-1],[-2],[-3]])
        # classe.env_left.append(env)

        # for i in range(site-1):
        #     ten = _braket(ket=array_1[i], bra=array_2[i], w=w[i])
        #     env = ncon([env,ten],[[1,2,3],[1,2,3,-1,-2,-3]])
        # left = env
        # # print("The left overlap of the state:")
        # # print(left)
        # env = ncon([a,a,a],[[-1],[-2],[-3]])
        # right = env
        # for i in range(classe.L-1, site-1, -1):
        #     ten = _braket(ket=array_1[i], bra=array_2[i], w=w[i])
        #     # print(f"braket shape: {ten.shape}")
        #     # print(f"env shape: {env.shape}")
        #     env = ncon([ten,env],[[-1,-2,-3,1,2,3],[1,2,3]])
        # right = env
        # # print("The right overlap of the state:")

        # M = ncon([left,array_1[site - 1],w[site - 1],right],[[1,4,-1],[1,2,3],[4,5,2,-2],[3,5,-3]])
        M = ncon([classe.env_left[-1],classe.ancilla_sites[site-1],classe.w[site-1],classe.env_right[-1]],[[1,4,-1],[1,2,3],[4,5,2,-2],[3,5,-3]])
        return M

def compute_M_no_mpo(classe, site, rev=False):
    """
    _compute_M

    This function computes the rank-3 tensor, in a specific site,
    given by the contraction of our variational state (phi) saved in classe.sites,
    and the uncompressed state (psi) saved in classe.ancilla_sites.

    site: int - site where to execute the tensor contraction

    """
    # array_1 = classe.ancilla_sites
    # array_2 = classe.sites
    # if rev:
    #     array_1 = classe.sites
    #     array_2 = classe.ancilla_sites
    # a = np.array([1])
    # env = ncon([a,a],[[-1],[-3]])

    # for i in range(site-1):
    #     ten = ncon([array_1[i],array_2[i].conjugate()],[[-1,1,-3],[-2,1,-4]])
    #     env = ncon([env,ten],[[1,2],[1,2,-1,-2]])
    # left = env
    # # print("The left overlap of the state:")
    # # print(left)
    # env = ncon([a,a],[[-1],[-2]])
    # for i in range(classe.L-1, site-1, -1):
    #     ten = ncon([array_1[i],array_2[i].conjugate()],[[-1,1,-3],[-2,1,-4]])
    #     # print(f"braket shape: {ten.shape}")
    #     # print(f"env shape: {env.shape}")
    #     env = ncon([ten,env],[[-1,-2,1,2],[1,2]])
    # right = env
    # # print("The right overlap of the state:")

    # M = ncon([left,array_1[site - 1],right],[[1,-1],[1,-2,2],[2,-3]])
    M = ncon([classe.env_left[-1],classe.ancilla_sites[site-1],classe.env_right[-1]],[[1,-1],[1,-2,2],[2,-3]])
    return M

def error(classe, site, M, N_eff):
    AM = ncon([M,classe.sites[site-1].conjugate()],[[1,2,3],[1,2,3]])
    A = classe.sites[site-1].flatten()
    AN_effA = ncon([A,N_eff,A.conjugate()],[[1],[1,2],[2]])
    error = AN_effA - 2*AM.real
    return error

def update_state(classe, sweep, site, trunc_tol, trunc_chi, e_tol=10 ** (-15), precision=2):   
    """
    update_state

    This function updates the classe.a and classe.b lists of tensors composing
    the mps. The update depends on the sweep direction. We take the classe.m
    extracted from the eigensolver and we decomposed via svd.

    sweep: string - direction of the sweeping. Could be "left" or "right"
    site: int - indicates which site the DMRG is optimizing
    trunc: bool - if True will truncate the the Schmidt values and save the
            state accordingly.
    e_tol: float - the tolerance accepted to truncate the Schmidt values
    precision: int - indicates the precision of the parameter h

    """
    if sweep == "right":
        # we want to write M (left,d,right) in LFC -> (left*d,right)
        m = classe.sites[site - 1].reshape(
            classe.sites[site - 1].shape[0] * classe.d, classe.sites[site - 1].shape[2]
        )
        # m = truncation(m, threshold=1e-16)
        # m = csr_matrix(m)
        # print(m)
        # u, s, v = scipy.sparse.linalg.svds(m, k=min(m.shape)-1)
        u, s, v = np.linalg.svd(m, full_matrices=False)
        
        
        if trunc_tol:
            condition = s >= e_tol
            s_trunc = np.extract(condition, s)
            s = s_trunc / np.linalg.norm(s_trunc)
            bond_l = u.shape[0] // classe.d
            u = u.reshape(bond_l, classe.d, u.shape[1])
            u = u[:, :, : len(s)]
            v = v[: len(s), :]
        elif trunc_chi:
            s_trunc = s[:classe.chi]
            s = s / np.linalg.norm(s_trunc)
            # print(f"Schmidt Values:\n{s}")
            bond_l = u.shape[0] // classe.d
            u = u.reshape(bond_l, classe.d, u.shape[1])
            u = u[:, :, : len(s)]
            v = v[: len(s), :]
        else:
            u = u.reshape(
                classe.sites[site - 1].shape[0], classe.d, classe.sites[site - 1].shape[2]
            )
        
        # print(f"Schmidt sum: {sum(s**2)}")
        next_site = ncon(
            [np.diag(s), v, classe.sites[site]],
            [
                [-1,1],
                [1,2],
                [2,-2,-3],
            ],
        )
        classe.sites[site - 1] = u
        classe.sites[site] = next_site

    elif sweep == "left":
        # we want to write M (left,d,right) in RFC -> (left,d*right)
        m = classe.sites[site - 1].reshape(
            classe.sites[site - 1].shape[0], classe.d * classe.sites[site - 1].shape[2]
        )
        u, s, v = np.linalg.svd(m, full_matrices=False)
        
        if trunc_tol:
            condition = s >= e_tol
            s_trunc = np.extract(condition, s)
            s = s_trunc / np.linalg.norm(s_trunc)
            bond_r = v.shape[1] // classe.d
            v = v.reshape(v.shape[0], classe.d, bond_r)
            v = v[: len(s), :, :]
            u = u[:, : len(s)]
        elif trunc_chi:
            s_trunc = s[:classe.chi]
            s = s / np.linalg.norm(s_trunc)
            # print(f"Schmidt Values:\n{s}")
            bond_r = v.shape[1] // classe.d
            v = v.reshape(v.shape[0], classe.d, bond_r)
            v = v[: len(s), :, :]
            u = u[:, : len(s)]
        else:
            v = v.reshape(
                classe.sites[site - 1].shape[0], classe.d, classe.sites[site - 1].shape[2]
            )
        # print(f"Schmidt sum: {sum(s**2)}")
        next_site = ncon(
            [classe.sites[site - 2], u, np.diag(s)], 
            [
                [-1,-2,1],
                [1,2],
                [2,-3],
            ],
        )
        classe.sites[site - 1] = v
        classe.sites[site - 2] = next_site

    return classe

def update_envs(classe, sweep, site):
    """
    update_envs

    This function updates the left and right environments for the next
    site optimization performed by the eigensolver. After the update of the mps
    in LCF and RCF we can compute the new environment and throw the one we do not need.

    sweep: string - direction of the sweeping. Could be "left" or "right"
    site: int - site we are optimizing

    """
    if sweep == "right":
        array = classe.sites[site - 1]
        ancilla_array = classe.ancilla_sites[site - 1]
        E_l = classe.env_left[-1]
        E_l = ncon(
            [E_l,ancilla_array,array.conjugate()],
            [
                [1,3],
                [1,2,-1],
                [3,2,-3],
            ],
        )
        classe.env_left.append(E_l)
        classe.env_right.pop(-1)

    if sweep == "left":
        array = classe.sites[site - 1]
        ancilla_array = classe.ancilla_sites[site - 1]
        E_r = classe.env_right[-1]
        E_r = ncon(
            [E_r,ancilla_array,array.conjugate()],
            [
                [1,3],
                [-1,2,1],
                [-3,2,3],
            ],
        )
        classe.env_right.append(E_r)
        classe.env_left.pop(-1)

    return classe

def compression(classe, trunc_tol, trunc_chi, e_tol=10 ** (-15), n_sweeps=2, precision=2):
    sweeps = ["right", "left"]
    sites = np.arange(1, classe.L + 1).tolist()
    errors = []

    environments(classe, site=1)
    iter = 1
    for n in range(n_sweeps):
        # print(f"Sweep n: {n}\n")
        for i in range(classe.L - 1):
            # print(f"\n============= Site: {sites[i]} ===================\n")
            # N_eff, l_shape, r_shape = _N_eff(classe, site=sites[i])
            # N_eff = truncation(array=N_eff, threshold=1e-15)
            # N_eff_sp = csr_matrix(N_eff)
            # print("After N_eff")
            # classe._compute_norm(site=1)
            
            M = compute_M_no_mpo(classe, sites[i])
            # t_plus_dt = sandwich_site_ancilla_site(classe, sites[i])
            # print(f"The overlap of states phi and Opsi is: {t_plus_dt}")
            # print("After M")
            # classe._compute_norm(site=1)

            # lin_sys(classe, M, N_eff_sp, sites[i], l_shape, r_shape)
            classe.sites[sites[i]-1] = M
            # t_plus_dt = ncon([classe.sites[sites[i]-1].conjugate(),M],[[1,2,3],[1,2,3]])
            # print(f"The overlap of states phi (updated) and Opsi is: {t_plus_dt}")
            # print("After linear system")
            # classe._compute_norm(site=1)
            
            # err = error(classe,  site=sites[i], N_eff=N_eff, M=M)
            # print("After err")
            # classe._compute_norm(site=1)

            # print(f"error per site {sites[i]}: {err:.5f}")
            # errors.append(err)
            update_state(classe, sweeps[0], sites[i], trunc_tol, trunc_chi, e_tol, precision)
            update_envs(classe, sweeps[0], sites[i])
            # classe.update_envs(sweeps[0], sites[i], mixed=True)
            # print("After update state")
            # classe.canonical_form()
            # classe._compute_norm(site=1)

            iter += 1

        sweeps.reverse()
        sites.reverse()
    
    return errors


# Before compressing, we evolve the state at its maximum bond dimension since the chain is small.
# 
# The maximum bond dimension is $2^{L/2}$ so in our case $\chi_{max} = 2^{4} = 16 $. This $\chi_{max}$ is reached in $4$ trotter steps, hence we perform a direct application of the mpo anc compare with the exact time evolution

# In[5]:
L = 9
n_points = 100
ks = 20
spectrum = compute_ising_spectrum(L=L, h_l=1e-7, n_points=n_points, k=ks)
# %%
deltas = [0.1,0.2,0.4,1.0]
colors = create_sequential_colors(num_colors=len(deltas), colormap_name='viridis')
energy = []
plt.title(f"Energy spectrum $vs$ $h$ - (L={L})")
idx = 0
for k in range(ks):
    energy_k = np.asarray(spectrum)[:,k]
    energy.append(energy_k)
    
    if (k == 2**1 - 1) or (k == 2**4 + 2 - 1) or (k == ks - 1):
        plt.plot(np.linspace(0,2,n_points), energy_k, alpha=0.5, color=colors[idx], label=f"Energy k={idx}")
        idx += 1
    else:
        plt.plot(np.linspace(0,2,n_points), energy_k, alpha=0.5, color=colors[idx])
plt.ylabel("Energy levels")
plt.xlabel("trasverse field (h)")
plt.legend()
plt.show()
gap = np.abs(np.asarray(energy)[0]) - np.abs(np.asarray(energy)[2])
plt.title(f"Energy gap $vs$ $\delta$ - (L={L})")
plt.plot(np.linspace(0,2,n_points), gap)
for i, delta in enumerate(deltas):
    plt.hlines(y=delta, xmin=0, xmax=2, colors=colors[i], label=f"$\delta = {delta}$")
plt.ylabel("Energy gap ($E_{k=0} - E_{k=2}$ non degenerate)")
plt.xlabel("trasverse field (h)")
plt.legend()
plt.show()
# %%
# main compression algorithm

# ground state
L = 9
chi = 2**(L//2-1)
Z = np.array([[1,0],[0,-1]])
chain = MPS(L=L, d=2, model='Ising', chi=chi, h=0, eps=0, J=1)
chain._random_state(seed=3, chi=chi)
chain.canonical_form()
chain.sweeping(trunc=True, n_sweeps=2)
chain.flipping_mps()

# local
mag_loc = []
for i in range(chain.L):
    chain.single_operator_Ising(site=i+1, op=Z)
    mag_loc.append(chain.mpo_first_moment())
print(f"Magnetization: {mag_loc}")

# total
mag_tot = []
chain.order_param_Ising(op=Z)
mag_tot.append(np.real(chain.mpo_first_moment()))

# fidelity
# psi_mps = mps_to_vector(chain.sites)
# fidelity = np.abs((exact_states[0].T.conjugate() @ psi_mps).real)
# print(f'fidelity before evolution: {fidelity}')


# In[6]:

# direct mpo evolution
trotter_steps = 50
t = 5
delta = t/trotter_steps
h_ev = 0.3
Ls = [7,9,11,13,15,17,19,21]
comp_time = []
for L in Ls:
    chi = 2**(L//2-1)
    Z = np.array([[1,0],[0,-1]])
    chain = MPS(L=L, d=2, model='Ising', chi=chi, h=0, eps=0, J=1)
    chain._random_state(seed=3, chi=chi)
    chain.canonical_form()
    chain.sweeping(trunc=True, n_sweeps=2)
    chain.flipping_mps()
    tot_time = time.perf_counter()
    mag_tot_ev, mag_loc_ev, overlap = chain.direct_mpo_evolution(trotter_steps=trotter_steps, delta=delta, h_ev=h_ev, J_ev=1, fidelity=False, trunc=True)
    comp_time.append(time.perf_counter()-tot_time)
    print(f"bond dimesion in the middle of the chain: {chain.sites[L//2].shape[0]}")
    # print(f"Trotter time for direct (SVD) compression: {time.perf_counter()-tot_time}")
np.savetxt(f"results/times_data/svd_truncation_different_Ls_trotter_{trotter_steps}", comp_time)


# # total
# mag_mps_tot = mag_tot + mag_tot_ev

# # local
# mag_loc_ev.reverse()
# mag_mps_loc = mag_loc_ev
# mag_mps_loc.append(mag_loc)
# mag_mps_loc.reverse()

# fidelity
# overlap.reverse()
# overlap.append(fidelity)
# overlap.reverse()


# In[7]:


# visualization 

# Local data
data1 = mag_exact_loc
data2 = mag_mps_loc
# data1 = data2
title1 = "Exact quench (local mag)"
title2 = "MPS quench (local mag)"
# title1 = title2
plot_side_by_side_(data1=data1, data2=data2, cmap='seismic', title1=title1, title2=title2)

# total
# chi = chain.sites[L//2].shape[0]
plt.title(f"Total Magnetization for MPS $vs$ exact " + "$h_{ev} =$" + f"{h_ev}")
plt.plot(delta*np.arange(trotter_steps+1), mag_exact_tot, '--', label=f"exact: $L={L}$")
plt.scatter(delta*np.arange(trotter_steps+1), mag_mps_tot, s=20, marker='o', alpha=0.7, facecolors='none', edgecolors='orange', label=f"mps: $\delta={delta}$")
plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()

# fidelity
plt.title("Fidelity $\left<\psi_{MPS}(t)|\psi_{exact}(t)\\right>$: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
plt.plot(delta*np.arange(trotter_steps+1), overlap)


# In[8]:


chi = chain.sites[L//2].shape[0]
tensor_shapes(chain.sites)


# In[9]:


# a = np.array([1])
# env = ncon([a,a], [[-1],[-2]])
# for i in range(len(chain.sites)):
#     print(f"Site: {i+1}")
#     ten = chain.overlap_sites(chain.ancilla_sites[i], chain.sites[i])
#     print(ten.shape)
#     env = ncon([env,ten],[[1,2],[1,2,-1,-2]])
# env_f = ncon([a,a], [[-1],[-2]])
# res = ncon([env,env_f],[[1,2],[1,2]])
# print(res)


# In[10]:


for i in range(len(states)-1):
    print(f"Overlap states at t={i}, t={i+1}")
    print(states[i].T.conjugate() @ states[i+1])


# In[11]:


# compressing during trotter evolution

# we can put the initial state in the ancilla sites argument
# chain._random_state(seed=3, chi=16, ancilla=True)
# cf_time = time.perf_counter()
chain.canonical_form()
# print(f"Time of canonical form: {time.perf_counter()-cf_time}")
# norm_time = time.perf_counter()
chain._compute_norm(site=1)
# print(f"Time of compute norm: {time.perf_counter()-norm_time}")
chain.ancilla_sites = chain.sites
# we use as a guess state the ground state that is already in sites
Z = np.array([[1,0],[0,-1]])

err_tot = []
n_sweeps = 2
trotter = 55
t = 5.5
delta = t/trotter
states_mps = []
tot_time = time.perf_counter()
for trott in range(trotter):
    print(f"------ Trotter steps: {trott} -------")
    # site_vec = mps_to_vector(chain.sites)
    # states_mps.append(site_vec)
    chain.mpo_Ising_time_ev(delta=delta, h_ev=h_ev, J_ev=1)
    # chain._compute_norm(site=1)
    # mpo_time = time.perf_counter()
    chain.mpo_to_mps(ancilla=True)
    # print(f"Time of mpo on mps: {time.perf_counter()-mpo_time}")
    chain.canonical_form(ancilla=True, trunc=False)
    # ancilla_vec = mps_to_vector(chain.ancilla_sites)
    # print(f"Overlap states at t={trott}, t={trott+1}")
    # print(site_vec.T.conjugate() @ ancilla_vec)
    # tensor_shapes(chain.ancilla_sites)
    # chain._compute_norm(site=1, ancilla=True)
    # err = compression(chain, trunc=True, n_sweeps=n_sweeps)
    # compr_time = time.perf_counter()
    compression(chain, trunc=True, n_sweeps=n_sweeps)
    # print(f"Time of compression: {time.perf_counter()-compr_time}")
    # tensor_shapes(chain.sites)
    # chain.ancilla_sites = chain.sites
    # err_tot.append(err)

    # local
    mag = []
    for i in range(chain.L):
        chain.single_operator_Ising(site=i+1, op=Z)
        mag.append(chain.mpo_first_moment().real)
    mag_mps_loc.append(mag)

    # total
    chain.order_param_Ising(op=Z)
    mag_mps_tot.append(np.real(chain.mpo_first_moment()))

print(f"Trotter time for variational compression: {time.perf_counter()-tot_time}")

    # # fidelity
    # U_new = exact_evolution_operator(L=L, h_t=h_ev, delta=delta, trotter_step=trott+5)
    # psi_new = U_new @ psi_exact

    # # total
    # mag = (psi_new.T.conjugate() @ mag_tot_op @ psi_new).real
    # mag_exact_tot.append(mag)
    # # local
    # mag_exact = []
    # for i in range(L):
    #     mag_exact.append((psi_new.T.conjugate() @ mag_loc_op[i] @ psi_new).real)
    # mag_exact_loc.append(mag_exact)
    
    # psi_new_mps = mps_to_vector(chain.sites)
    # overlap.append(np.abs((psi_new_mps.T.conjugate() @ psi_new).real))


# In[12]:

trotter_steps = 50
t = 5
delta = t/trotter_steps
h_ev = 0.3
Ls = [7,9,11,13,15,17,19,21]
comp_time = []
for L in Ls:
    chi = 2**(L//2-1)
    Z = np.array([[1,0],[0,-1]])
    chain = MPS(L=L, d=2, model='Ising', chi=chi, h=0, eps=0, J=1)
    chain._random_state(seed=3, chi=chi)
    chain.canonical_form()
    chain.sweeping(trunc=True, n_sweeps=2)
    chain.flipping_mps()
    
    chain.canonical_form()
    chain._compute_norm(site=1)
    chain.ancilla_sites = chain.sites
    Z = np.array([[1,0],[0,-1]])

    n_sweeps = 2
    trotter = 50
    t = 5
    delta = t/trotter
    mag_mps_loc = []
    mag_mps_tot = []
    tot_time = time.perf_counter()
    for trott in range(trotter):
        print(f"------ Trotter steps: {trott} -------")
        chain.mpo_Ising_time_ev(delta=delta, h_ev=h_ev, J_ev=1)
        chain.mpo_to_mps(ancilla=True)
        chain.canonical_form(ancilla=True, trunc=False)
        compression(chain, trunc=True, n_sweeps=n_sweeps)

        # local
        mag = []
        for i in range(chain.L):
            chain.single_operator_Ising(site=i+1, op=Z)
            mag.append(chain.mpo_first_moment().real)
        mag_mps_loc.append(mag)

        # total
        chain.order_param_Ising(op=Z)
        mag_mps_tot.append(np.real(chain.mpo_first_moment()))

        # fidelity
        U_new = exact_evolution_operator(L=L, h_t=h_ev, delta=delta, trotter_step=trott+1)
        psi_new = U_new @ psi_exact

        # total
        mag = (psi_new.T.conjugate() @ mag_tot_op @ psi_new).real
        mag_exact_tot.append(mag)
        # local
        mag_exact = []
        for i in range(L):
            mag_exact.append((psi_new.T.conjugate() @ mag_loc_op[i] @ psi_new).real)
        mag_exact_loc.append(mag_exact)
        
        psi_new_mps = mps_to_vector(chain.sites)
        overlap.append(np.abs((psi_new_mps.T.conjugate() @ psi_new).real))

np.savetxt(f"results/times_data/variational_truncation_different_Ls_trotter_{trotter_steps}", comp_time)

# %%

trotter_steps = 50
t = 5
delta = t/trotter_steps
h_t = 0
h_ev = 0.3
L = 9
n_sweeps = 2

# exact
psi_exact = exact_initial_state(L=L, h_t=h_t)
Z = np.array([[1,0],[0,-1]])
# local
mag_loc_op = [single_site_op(op=Z, site=i, L=L) for i in range(1,L+1)]
# total
mag_tot_op = H_loc(L=L, op=Z)

comp_time = []
for i in range(1,L//2+1):
    chi = 2**i
    chain = MPS(L=L, d=2, model='Ising', chi=chi, h=0, eps=0, J=1)
    chain._random_state(seed=3, chi=chi)
    chain.canonical_form()
    chain.sweeping(trunc=True, n_sweeps=2)
    chain.flipping_mps()
    
    mag_mps_loc = []
    mag_mps_tot = []
    # local
    mag_loc = []
    for i in range(chain.L):
        chain.single_operator_Ising(site=i+1, op=Z)
        mag_loc.append(chain.mpo_first_moment())
    mag_mps_loc.append(mag_loc)

    # total
    chain.order_param_Ising(op=Z)
    mag_mps_tot.append(np.real(chain.mpo_first_moment()))

    # exact
    mag_exact_loc = []
    mag_exact_tot = []

    # local
    mag_exact = []
    for i in range(L):
        mag_exact.append((psi_exact.T.conjugate() @ mag_loc_op[i] @ psi_exact).real)
    mag_exact_loc.append(mag_exact)

    # total
    mag = psi_exact.T.conjugate() @ mag_tot_op @ psi_exact
    mag_exact_tot.append(mag.real)

    # fidelity
    psi_mps = mps_to_vector(chain.sites)
    overlap = []
    overlap.append(np.abs((psi_exact.T.conjugate() @ psi_mps).real))

    chain.canonical_form()
    chain._compute_norm(site=1)
    chain.ancilla_sites = chain.sites

    tot_time = time.perf_counter()
    for trott in range(trotter_steps):
        print(f"------ Trotter steps: {trott} -------")
        chain.mpo_Ising_time_ev(delta=delta, h_ev=h_ev, J_ev=1)
        chain.mpo_to_mps(ancilla=True)
        chain.canonical_form(ancilla=True, trunc=True)
        compression(chain, trunc_tol=False, trunc_chi=True, n_sweeps=n_sweeps)
        print(chain.chi)
        tensor_shapes(chain.sites)
        # local
        mag = []
        for i in range(chain.L):
            chain.single_operator_Ising(site=i+1, op=Z)
            mag.append(chain.mpo_first_moment().real)
        mag_mps_loc.append(mag)

        # total
        chain.order_param_Ising(op=Z)
        mag_mps_tot.append(np.real(chain.mpo_first_moment()))
        # comp_time.append(time.perf_counter()-tot_time)

        # exact
        U_new = exact_evolution_operator(L=L, h_t=h_ev, delta=delta, trotter_step=trott+1)
        psi_new = U_new @ psi_exact

        # total
        mag = (psi_new.T.conjugate() @ mag_tot_op @ psi_new).real
        mag_exact_tot.append(mag)
        
        # local
        mag_exact = []
        for i in range(L):
            mag_exact.append((psi_new.T.conjugate() @ mag_loc_op[i] @ psi_new).real)
        mag_exact_loc.append(mag_exact)
        
        # fidelity
        psi_new_mps = mps_to_vector(chain.sites)
        overlap.append(np.abs((psi_new_mps.T.conjugate() @ psi_new).real))
    
    np.savetxt(f"results/mag_data/mag_mps_tot_L_{L}_delta_{delta}_chi_{chi}", mag_mps_tot)
    np.savetxt(f"results/mag_data/mag_mps_loc_L_{L}_delta_{delta}_chi_{chi}", mag_mps_loc)
    np.savetxt(f"results/mag_data/mag_exact_tot_L_{L}_delta_{delta}_chi_{chi}", mag_exact_tot)
    np.savetxt(f"results/mag_data/mag_exact_loc_L_{L}_delta_{delta}_chi_{chi}", mag_exact_loc)
    np.savetxt(f"results/fidelity_data/fidelity_L_{L}_delta_{delta}_chi_{chi}", overlap)

# np.savetxt(f"results/times_data/variational_truncation_different_Ls_trotter_{trotter_steps}", comp_time)


# In[13]:

# visualization

# total
plt.title(f"Total Magnetization for $\delta = {delta}$ ;" + " $h_{ev} =$" + f"{h_ev}")
colors = create_sequential_colors(num_colors=len(range(2,L//2+2)), colormap_name='viridis')
for i in range(2,L//2+1):
    chi = 2**i
    mag_mps_tot = np.loadtxt(f"results/mag_data/mag_mps_tot_L_{L}_delta_{delta}_chi_{chi}")
    plt.scatter(delta*np.arange(trotter_steps+1), mag_mps_tot, s=25, marker='o', alpha=0.8, facecolors='none', edgecolors=colors[i-1], label=f"mps: $\chi={chi}$")

plt.plot(delta*np.arange(trotter_steps+1), mag_exact_tot, color='violet', label=f"exact: $L={L}$")
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
# overlap.reverse()
# overlap.append(fidelity)
# overlap.reverse()
plt.title("Fidelity $\left<\psi_{MPS}(t)|\psi_{exact}(t)\\right>$: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
for i in range(2,L//2+1):
    chi = 2**i
    fidelity = np.loadtxt(f"results/fidelity_data/fidelity_L_{L}_delta_{delta}_chi_{chi}")
    plt.scatter(delta*np.arange(trotter_steps+1), fidelity, s=20, marker='o', alpha=0.7, facecolors='none', edgecolors=colors[i-1], label=f"mps: $\chi={chi}$")

plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()

# In[14]:


plt.plot(np.asarray(err_tot)[:,0].real)
a = np.linspace(0, len(err_tot)-1, n_sweeps)
# for sw in a:
#     plt.vlines(sw, ymin=min(err_tot),ymax=max(err_tot), colors='violet', linestyles='dashed')




