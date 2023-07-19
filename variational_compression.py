#%%
from mps_class_v9 import MPS
import numpy as np
from utils import *
from ncon import ncon
import scipy
from scipy.linalg import expm
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# %%
# ---------------------------------------------------------
# mps: h_t = 0 --> h_t = h_ev
# ---------------------------------------------------------
# initialize the chain in h_t = 0
L = 11
spin = MPS(L=L, d=2, model="Ising", chi=32, J=1, h=0, eps=0)
spin._random_state(seed=3, chi=32)
spin.canonical_form()
energies = spin.sweeping(trunc=True)
spin.flipping_mps()
# %%
# time evolution to h_t = h_ev
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
trotter_steps = 6
delta = 0.6
h_ev = 0.1
J_ev = 1
fidelity = True
mag_mpo_tot, overlap = spin.direct_mpo_evolution(trotter_steps=trotter_steps, 
                          delta=delta, 
                          h_ev=h_ev, 
                          J_ev=J_ev, 
                          fidelity=fidelity)

# %%
# visualization
plt.title(f"MPS: $\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
plt.imshow(mag_mpo_tot, cmap="seismic", vmin=-1, vmax=1, aspect=1)
plt.show()
if fidelity:
    plt.title("Fidelity $\left<\psi_{MPS}(t)|\psi_{MPS}(t=0)\\right>$: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
    plt.plot(overlap)
    plt.show()
# %%
# compression functions

def lin_sys(classe, M, N_eff, site, l_shape, r_shape):
    M_new = M.flatten()
    new_site = scipy.linalg.solve(N_eff, M_new)
    new_site = new_site.reshape((l_shape[0], classe.d, r_shape[0]))
    classe.sites[site - 1] = new_site
    return classe

def compute_M(classe, site, rev=False):
        """
        _compute_M

        This function computes the rank-3 tensor, in a specific site,
        given by the contraction of our variational state (phi) saved in self.sites,
        and the uncompressed state (psi) saved in self.ancilla_sites.

        site: int - site where to execute the tensor contraction

        """
        array_1 = classe.ancilla_sites
        array_2 = classe.sites
        if rev:
            array_1 = classe.sites
            array_2 = classe.ancilla_sites
        a = np.array([1])
        env = ncon([a,a,a,a],[[-1],[-2],[-3],[-4]])
        left = env

        for i in range(site-1):
            ten = classe.overlap_sites(array_1=array_1[i], array_2=array_2[i])
            env = ncon([env,ten],[[-1,-2,1,2],[1,2,-3,-4]])
        left = env
        # print("The left overlap of the state:")
        # print(left)
        env = ncon([a,a,a,a],[[-1],[-2],[-3],[-4]])
        right = env
        for i in range(classe.L-1, site-1, -1):
            ten = classe.overlap_sites(array_1=array_1[i], array_2=array_2[i])
            env = ncon([ten,env],[[-1,-2,1,2],[1,2,-3,-4]])
        right = env
        # print("The right overlap of the state:")
        # print(right)

        M = ncon([a,a,left,array_1[site - 1],right,a,a],[[1],[2],[1,2,3,-1],[3,-2,4],[4,-3,5,6],[5],[6]])
        return M

def error(classe, site, M, N_eff):
    phi_psi = ncon([M,classe.sites[site-1].conjugate()],[[1,2,3],[1,2,3]])
    psi_psi = classe._compute_norm(site=site, ancilla=True)
    phi_phi = classe._compute_norm(site=site)
    M_rev = compute_M(classe, site, rev=True)
    psi_phi = ncon([M_rev,classe.ancilla_sites[site-1].conjugate()],[[1,2,3],[1,2,3]])
    error = psi_psi - psi_phi - phi_psi + phi_phi
    return error

def update_state(classe, sweep, site, trunc, e_tol=10 ** (-15), precision=2):   
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
            u, s, v = np.linalg.svd(m, full_matrices=False)
            if trunc:
                condition = s >= e_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / np.linalg.norm(s_trunc)
                bond_l = u.shape[0] // classe.d
                u = u.reshape(bond_l, classe.d, u.shape[1])
                u = u[:, :, : len(s)]
                v = v[: len(s), :]
            else:
                u = u.reshape(
                    classe.sites[site - 1].shape[0], classe.d, classe.sites[site - 1].shape[2]
                )
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
            if trunc:
                condition = s >= e_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / np.linalg.norm(s_trunc)
                bond_r = v.shape[1] // classe.d
                v = v.reshape(v.shape[0], classe.d, bond_r)
                v = v[: len(s), :, :]
                u = u[:, : len(s)]
            else:
                v = v.reshape(
                    classe.sites[site - 1].shape[0], classe.d, classe.sites[site - 1].shape[2]
                )
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

def compression(classe, trunc, e_tol=10 ** (-15), n_sweeps=2, precision=2):
    sweeps = ["right", "left"]
    sites = np.arange(1, classe.L + 1).tolist()
    errors = []

    iter = 1
    for n in range(n_sweeps):
        print(f"Sweep n: {n}\n")
        for i in range(classe.L - 1):
            N_eff, l_shape, r_shape = classe.N_eff(site=sites[i])
            M = compute_M(classe, sites[i])
            lin_sys(classe, M, N_eff, sites[i], l_shape, r_shape)
            err = error(classe,  site=sites[i], N_eff=N_eff, M=M)
            print(f"error per site {sites[i]}: {err:.5f}")
            errors.append(err)
            update_state(classe, sweeps[0], sites[i], trunc, e_tol, precision)
            iter += 1

        sweeps.reverse()
        sites.reverse()
    
    return classe, errors[-1]

def exact_time_ev(L, delta, h_t):
    X = np.array([[0,1],[1,0]])
    Z = np.array([[1,0],[0,-1]])
    H = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=0)
    e, v = np.linalg.eig(H)
    psi = v[:,0]
    flip = single_site_op(op=X, site=L // 2 + 1, L=L)
    psi = flip @ psi
    H_ev = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=h_t)
    U = expm(-1j*delta*H_ev)
    U_new = truncation(array=U, threshold=1e-16)
    U_new = csr_matrix(U_new)
    return U_new, psi

# %%
L = 11
opt_chain = MPS(L=L, d=2, model="Ising", chi=100, J=1, h=0, eps=0)
opt_chain._random_state(seed=3, chi=100, type_shape="rectangular")
opt_chain.canonical_form(trunc=False)
opt_chain._compute_norm(site=1)
opt_chain.ancilla_sites = spin.sites
tensor_shapes(opt_chain.sites)
# %%
opt_chain_compressed, errors = compression(classe=opt_chain, trunc=True)
# %%
plt.plot(errors.real,'o')
# %%
psi_compressed = mps_to_vector(opt_chain_compressed.sites)
psi_uncompressed = mps_to_vector(spin.sites)
fidelity = psi_uncompressed.T.conjugate() @ psi_compressed
print(f"Fidelity: {fidelity}")
# %%
# comparison with exact
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
fidelity = psi_compressed.T.conjugate() @ psi_new
print(f"Fidelity: {fidelity}")
# %%
# trotter compression
# initialize the chain in h_t = 0
L = 11
spin = MPS(L=L, d=2, model="Ising", chi=32, J=1, h=0, eps=0)
spin._random_state(seed=3, chi=32)
spin.canonical_form()
energies = spin.sweeping(trunc=True)
spin.flipping_mps()
trotter_steps = 8
delta = 0.6
h_ev = 0.3
chi_max = [32,64]
fidelity = False
mag_mpo_tot, errors, overlap = spin.compressed_mpo_evolution(trotter_steps=trotter_steps, fidelity=fidelity, delta=delta, h_ev=h_ev, chi_max=chi_max)
# %%
# visualization
plt.title(f"MPS: $\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
plt.imshow(mag_mpo_tot, cmap="seismic", vmin=-1, vmax=1, aspect=1)
plt.show()
plt.title(f"Error for $\chi_{{max}}$: $\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
plt.plot(errors)
plt.show()
if fidelity:
    plt.title("Fidelity $\left<\psi_{MPS}(t)|\psi_{MPS}(t=0)\\right>$: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
    plt.plot(overlap)
    plt.show()


# %%
# comparison with exact
L = 11
delta = 0.6
h_t = 0.3
Z = np.array([[1,0],[0,-1]])
U_new, psi = exact_time_ev(L, delta, h_t)
psi_new = psi
psi_compressed = mps_to_vector(spin.sites)
mag_exact_tot = []
magnetization = [single_site_op(op=Z, site=i, L=L) for i in range(1,L+1)]
for T in range(trotter_steps):
    psi_new = U_new @ psi_new
    mag_exact = []
    for i in range(L):
        mag_exact.append((psi_new.T.conjugate() @ magnetization[i] @ psi_new).real)
    print(f"----- trotter step {T} --------")
    mag_exact_tot.append(mag_exact)
fidelity = psi_compressed.T.conjugate() @ psi_new
# %%
plt.title(f"Exact: $\delta = {delta}$; $h_{{t-ev}} = {h_t}$")
plt.imshow(mag_exact_tot, cmap="seismic", vmin=-1, vmax=1, aspect=1)
plt.show()
print(f"Fidelity: {fidelity}")
# %%
