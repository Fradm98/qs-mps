# %%
# import packages
from qs_mps.utils import create_sequential_colors, tensor_shapes, mps_to_vector
from qs_mps.applications.Z2.ground_state_multiprocessing import ground_state_Z2
from qs_mps.applications.Z2.exact_hamiltonian import H_Z2_gauss
from qs_mps.sparse_hamiltonians_and_operators import sparse_pauli_x
from qs_mps.mps_class import MPS
import numpy as np
from scipy.sparse import identity
import matplotlib.pyplot as plt
import time

# %%
# finding the ground state of the vacuum sector
L = 3
l = 1
d = int(2**l)
chi = 2  # this is interpreted as d**(int(log2(chi))) --> e.g. chi=8 == 4**3=64
array = np.linspace(1e-7, 10, 100)
hs = [h for h in array]
model = "Z2_dual"
path = "D:/code/projects/1_Z2"
multpr = False
param = hs
# %%
if __name__ == "__main__":
    i = 0
    # for L in Ls:
    args_mps = {
        "L": L,
        "d": d,
        "chi": chi,
        "model": model,
        "trunc_tol": False,
        "trunc_chi": True,
        "where": L // 2,
        "path": path,
    }

    energies_h, entropy_h = ground_state_Z2(
        args_mps=args_mps, multpr=multpr, param=param
    )

    # %%
    # exact
    eig_exact = []
    eig_first = []
    colors = create_sequential_colors(L-1, "viridis")
    alphas = [1,0.7]
    
    for s in range(1,l+1):
        W = []
        W_exact = []
        for h in hs:
            ladder = MPS(L=L, d=d, model=model, h=h, eps=0, J=1, chi=chi)
            ladder.L = ladder.L - 1

            ladder.load_sites(path=path)
            psi_mps = mps_to_vector(ladder.sites)
            print("Psi of mps:")
            print(psi_mps)
            # ladder.Z2.wilson_Z2_dual(mpo_sites=[1], ls=[s]) #list(range(s))
            # ladder.w = ladder.Z2.mpo
            # W.append(ladder.mpo_first_moment().real)
        
        
            # exact
            Z2_exact = H_Z2_gauss(L=3, l=2, model="Z2", lamb=h, U=1e+3)
            e, v = Z2_exact.diagonalize()
            # eig_exact.append(np.min(e))
            # eig_first.append(np.sort(e)[1])
            psi = v[:,0]
            print("Psi exact:")
            print(psi)

            loop = Z2_exact.latt.plaquettes(from_zero=True)
            plaq = Z2_exact.plaquette_term(loop[0])
            # plaq = Z2_exact.plaquette_term(loop[s+1])
            exp_val_wilson_loop = np.real(psi.T @ plaq @ psi)
            W_exact.append(np.abs(exp_val_wilson_loop))


        # print(loop)
        # plt.plot(hs, W, "+", color=colors[s-1], label=f"Plaquette {s}")
        plt.plot(hs, W_exact, "+", color='r', alpha=alphas[s-1], label=f"exact {s}")

    # plt.scatter(hs,
    #             energies_h,
    #             marker="o",
    #             alpha=0.8,
    #             facecolors="none",
    #             edgecolors=colors[i],
    #             label=f"L: {L}"
    #             )
    # plt.plot(hs,eig_exact,'--',color='red', label='exact gs')
plt.vlines(x=1/3.044, ymin=0, ymax=1)
# plt.title(f"Energy(h)")
# plt.xscale('symlog')
# plt.yscale('log')
plt.xlabel("electric local parameter (h)")
# plt.ylim([-350, -50])
plt.legend(loc="lower right")
plt.show() 
# print(ladder.Z2.latt._lattice_drawer.draw_lattice())

# %%
print(ladder.Z2.latt._lattice_drawer.draw_lattice())

plt.vlines(x=3.044, ymin=np.min(W), ymax=np.max(W))
# plt.title(f"Energy(h)")
# plt.yscale('log')
plt.xlabel("electric local parameter (h)")
# plt.ylim([-350, -50])
plt.legend()
plt.show()
# %%
plt.plot(hs, entropy_h)
plt.vlines(x=3.044, ymin=np.min(entropy_h), ymax=np.max(entropy_h))
