# %%
# import packages
from mps_class_v9 import MPS
from utils import *
import matplotlib.pyplot as plt
from ncon import ncon
import scipy
from scipy.sparse import csr_array

# %%
# exact state and evolution
L = 9
h_t = 0
h_ev = 0.3
t = 0.4
trotter_steps = 4
delta = t/trotter_steps
psi_exact = exact_initial_state(L=L, h_t=h_t)

# we save the exact states to compute the fidelity
exact_states = []
exact_states.append(psi_exact)
# define the local and total magnetization operators
Z = np.array([[1,0],[0,-1]])
# local
mag_loc = [single_site_op(op=Z, site=i, L=L) for i in range(1,L+1)]
# total
mag_tot_op = H_loc(L=L, op=Z)

mag_exact_loc = []
mag_exact_tot = []

# local
mag_exact = []
for i in range(L):
    mag_exact.append((psi_exact.T.conjugate() @ mag_loc[i] @ psi_exact).real)
mag_exact_loc.append(mag_exact)

# total
mag = psi_exact.T.conjugate() @ mag_tot_op @ psi_exact
mag_exact_tot.append(mag)

for trott in range(1,trotter_steps+1):
    # compute the U in the time we are interested in, that is, delta*trott
    U = exact_evolution_operator(L, h_t=h_ev, delta=delta, trotter_step=trott)
    # final state after U time evolution
    psi_new = U @ psi_exact
    # exact_states.append(psi_new)
    # compute the total and local magnetization at that time
    
    # local
    mag_exact = []
    for i in range(L):
        mag_exact.append((psi_new.T.conjugate() @ mag_loc[i] @ psi_new).real)
    print(f"----- trotter step {trott} --------")
    mag_exact_loc.append(mag_exact)

    # total
    mag = psi_new.T.conjugate() @ mag_tot_op @ psi_new
    mag_exact_tot.append(mag)

# %%
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

# %%
# -----------------------------------------
"""
Now we can try to automize the time evolution and compress the bond dimesion.
This procedure works as follows:
- we apply the time ev mpo (bond dim - w) to an initial state (bond dim - m) and variationally find an mps with bond dimension m' < w*m
- we minimize a distance measure between the mpo applied on the initial state and the variational compressed state.
- we take track of the error
"""
# %%
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

def braket(ket, bra, w):
    sandwich = ncon([ket,w,bra.conjugate()],[[-1,1,-4],[-2,-5,1,2],[-3,2,-6]])
    return sandwich

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
        w = classe.w
        if rev:
            array_1 = classe.sites
            array_2 = classe.ancilla_sites
        a = np.array([1])
        env = ncon([a,a,a],[[-1],[-2],[-3]])
        left = env

        for i in range(site-1):
            ten = braket(ket=array_1[i], bra=array_2[i], w=w[i])
            env = ncon([env,ten],[[1,2,3],[1,2,3,-1,-2,-3]])
        left = env
        # print("The left overlap of the state:")
        # print(left)
        env = ncon([a,a,a],[[-1],[-2],[-3]])
        right = env
        for i in range(classe.L-1, site-1, -1):
            ten = braket(ket=array_1[i], bra=array_2[i], w=w[i])
            # print(f"braket shape: {ten.shape}")
            # print(f"env shape: {env.shape}")
            env = ncon([ten,env],[[-1,-2,-3,1,2,3],[1,2,3]])
        right = env
        # print("The right overlap of the state:")

        M = ncon([left,array_1[site - 1],w[site - 1],right],[[1,4,-1],[1,2,3],[4,5,2,-2],[3,5,-3]])
        return M

def error(classe, site, M, N_eff):
    AM = ncon([M,classe.sites[site-1].conjugate()],[[1,2,3],[1,2,3]])
    A = classe.sites[site-1].flatten()
    AN_effA = ncon([A,N_eff,A.conjugate()],[[1],[1,2],[2]])
    error = AN_effA - 2*AM.real
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
            # m = truncation(m, threshold=1e-16)
            # m = csr_matrix(m)
            # print(m)
            # u, s, v = scipy.sparse.linalg.svds(m, k=min(m.shape)-1)
            u, s, v = np.linalg.svd(m, full_matrices=False)
            
            print(f"Schmidt sum: {sum(s**2)}")
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
            print(f"Schmidt sum: {sum(s**2)}")
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
            print(f"\n============= Site: {sites[i]} ===================\n")
            N_eff, l_shape, r_shape = classe.N_eff(site=sites[i])
            N_eff = truncation(array=N_eff, threshold=1e-15)
            N_eff_sp = csr_matrix(N_eff)
            print("After N_eff")
            classe._compute_norm(site=1)

            # plt.title("Real part")
            # plt.imshow(N_eff.real, cmap='viridis')
            # plt.show()
            # plt.title("Imaginary part")
            # plt.imshow(N_eff.imag, cmap='viridis')
            # plt.show()
            M = compute_M(classe, sites[i])

            # A = classe.sites[sites[i]-1]
            # A_new = truncation(A, threshold=1e-16)
            # A_new = A_new.reshape(A_new.shape[0]*A_new.shape[1],A_new.shape[2])
            # A_new = csr_matrix(A_new)
            # print(A_new)
            # print(A.shape)
            print("After M")
            classe._compute_norm(site=1)

            t_plus_dt = ncon([classe.sites[sites[i]-1].conjugate(),M],[[1,2,3],[1,2,3]])
            print(f"The overlap of states at t and t+dt is: {t_plus_dt}")
            # lin_sys(classe, M, N_eff_sp, sites[i], l_shape, r_shape)
            classe.sites[sites[i]-1] = M
            print("After linear system")
            classe._compute_norm(site=1)
            
            err = error(classe,  site=sites[i], N_eff=N_eff, M=M)
            print("After err")
            classe._compute_norm(site=1)

            # print(f"error per site {sites[i]}: {err:.5f}")
            errors.append(err)
            update_state(classe, sweeps[0], sites[i], trunc, e_tol, precision)
            print("After update state")
            # classe.canonical_form()
            classe._compute_norm(site=1)
            

            iter += 1

        sweeps.reverse()
        sites.reverse()
    
    return errors[-1]

# %%
# main compression algorithm

# ground state
L = 9
chi = 16
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
psi_mps = mps_to_vector(chain.sites)
fidelity = np.abs((exact_states[0].T.conjugate() @ psi_mps).real)
print(f'fidelity before evolution: {fidelity}')

# %%
# direct mpo evolution
mag_tot_ev, mag_loc_ev, overlap = chain.direct_mpo_evolution(trotter_steps=trotter_steps, delta=delta, h_ev=h_ev, J_ev=1, fidelity=True, trunc=True)

# total
mag_mps_tot = mag_tot + mag_tot_ev

# local
mag_loc_ev.reverse()
mag_mps_loc = mag_loc_ev
mag_mps_loc.append(mag_loc)
mag_mps_loc.reverse()

# fidelity
overlap.reverse()
overlap.append(fidelity)
overlap.reverse()

# %%
# visualization 

def plot_side_by_side_(data1, data2, cmap='viridis', title1='Imshow 1', title2='Imshow 2'):
    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the imshow plots on the subplots
    im1 = ax1.imshow(data1, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    im2 = ax2.imshow(data2, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Set titles for the subplots
    ax1.set_title(title1)
    ax2.set_title(title2)

    # Remove ticks from the colorbar subplot
    # ax2.set_xticks([])
    ax2.set_yticks([])

    # Create a colorbar for the second plot on the right
    cbar = fig.colorbar(im2, ax=ax2)
    cbar = fig.colorbar(im1, ax=ax1)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

# Local data
data1 = mag_exact_loc
data2 = mag_mps_loc
title1 = "Exact quench (local mag)"
title2 = "MPS quench (local mag)"
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

# %%
# save results
np.savetxt(f"results\mag_data\magnetization_mps_tot_L_{L}_delta_{delta}_h_ev_{h_ev}", mag_mps_tot)
np.savetxt(f"results\mag_data\magnetization_mps_loc_L_{L}_delta_{delta}_h_ev_{h_ev}", mag_mps_loc)
np.savetxt(f"results\mag_data\magnetization_exact_tot_L_{L}_delta_{delta}_h_ev_{h_ev}", mag_exact_tot)
np.savetxt(f"results\mag_data\magnetization_exact_loc_L_{L}_delta_{delta}_h_ev_{h_ev}", mag_exact_loc)
np.savetxt(f"results\\fidelity_data\\fidelity_L_{L}_delta_{delta}_h_ev_{h_ev}", overlap)

# %%
# compressing during trotter evolution

# we can put the initial state in the ancilla sites argument
chain.ancilla_sites = chain.sites
chain.canonical_form()
chain._compute_norm(site=1)
# we use as a guess state the ground state that is already in sites
Z = np.array([[1,0],[0,-1]])

err_tot = []
for t in range(1):
    print(f"------ Trotter steps: {t+5} -------")
    chain.mpo_Ising_time_ev(delta=delta, h_ev=h_ev, J_ev=1)
    chain._compute_norm(site=1)
    err = compression(chain, trunc=True, n_sweeps=2)
    chain.ancilla_sites = chain.sites
    err_tot.append(err)

    # # local
    # mag = []
    # for i in range(chain.L):
    #     chain.single_operator_Ising(site=i+1, op=Z)
    #     mag.append(chain.mpo_first_moment().real)
    # mag_mps_loc.append(mag)
    # 

    # total
    chain.order_param_Ising(op=Z)
    mag_mps_tot.append(np.real(chain.mpo_first_moment()))
    
    # fidelity
    U_new = exact_evolution_operator(L=L, h_t=h_ev, delta=delta, trotter_step=t+5)
    psi_new = U_new @ psi_exact
    mag = (psi_new.T.conjugate() @ mag_tot_op @ psi_new).real
    mag_exact_tot.append(mag)
    psi_new_mps = mps_to_vector(chain.sites)
    overlap.append(np.abs((psi_new_mps.T.conjugate() @ psi_new).real))
    
# %%
# visualization

# total
chi = chain.sites[L//2].shape[0]
plt.plot(delta*np.arange(5+1), mag_exact_tot, '--', label=f"exact: $L={L}$")
plt.plot(delta*np.arange(5+1), mag_mps_tot, 'o', label=f"mps: $\chi={chi}$")
plt.legend()
plt.show()

# fidelity
# overlap.reverse()
# overlap.append(fidelity)
# overlap.reverse()
plt.plot(delta*np.arange(5+1), overlap)

# %%
# visualizing different trotterizations

# total
trotter_steps = [200,100,50,20]
deltas = t/np.array([200,100,50,20])
plt.title(f"Total Magnetization for MPS $vs$ exact " + "$h_{ev} =$" + f"{h_ev}")

colors = create_sequential_colors(num_colors=len(trotter_steps), colormap_name='gist_rainbow')
# colors.reverse()
i = 0
plt.plot(delta*np.arange(trott+1), mag_exact_tot, '--', linewidth=2, color="gold", label=f"exact: $L={L}$")
for delta, trott in zip(deltas,trotter_steps):
    mag_mps_tot = np.loadtxt(f"results\mag_data\magnetization_mps_tot_L_{L}_delta_{delta}_h_ev_{h_ev}")
    plt.scatter(delta*np.arange(trott+1), mag_mps_tot, s=20, marker='o', alpha=1, facecolors='none', edgecolors=colors[i], label=f"mps: $\delta={delta}$")
    i += 1
plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()

# # fidelity
# plt.title("Fidelity $\left<\psi_{MPS}(t)|\psi_{exact}(t)\\right>$: " + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$")
# plt.plot(delta*np.arange(trotter_steps+1), overlap)
# %%

# %%
plt.colorbar()