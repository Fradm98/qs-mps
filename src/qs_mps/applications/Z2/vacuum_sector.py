# %%
# import packages
from qs_mps.utils import create_sequential_colors, tensor_shapes
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
L = 4
l = 2
d = int(2**l)
chi = 2  # this is interpreted as d**(int(log2(chi))) --> e.g. chi=8 == 4**3=64
array = np.linspace(1e-7, 10, 100)
hs = [h for h in array]
model = "Z2_dual"
energies_h = []
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
    }

    energies_h, entropy_h = ground_state_Z2(
        args_mps=args_mps, multpr=multpr, param=param
    )

    # %%
    # exact
    eig_exact = []
    eig_first = []
    W = []
    for h in hs:
        ladder = MPS(L=L, d=d, model=model, h=h, eps=0, J=1, chi=chi)
        ladder.L = ladder.L - 1

        ladder.load_sites("/Users/fradm98/Desktop/qs-mps")
        ladder.Z2.wilson_Z2_dual(mpo_sites=[0, 1], ls=[1])
        ladder.w = ladder.Z2.mpo
        W.append(ladder.mpo_first_moment())
        # Z2_exact = H_Z2_gauss(L=5, l=2, model="Z2", lamb=h, U=1e+5)
        # e, v = Z2_exact.diagonalize()
        # eig_exact.append(np.min(e))
        # eig_first.append(np.sort(e)[1])
        # psi = v[:,0]

        # loop = Z2_exact.latt.plaquettes(from_zero=True)
        # print(loop)
        # plaq = Z2_exact.plaquette_term(loop[0])
        # exp_val_wilson_loop = np.real(psi.T @ plaq @ psi)
        # W.append(exp_val_wilson_loop)

    # plt.scatter(hs,
    #             energies_h,
    #             marker="o",
    #             alpha=0.8,
    #             facecolors="none",
    #             edgecolors=colors[i],
    #             label=f"L: {L}"
    #             )
    # plt.plot(hs,eig_exact,'--',color='red', label='exact gs')
    print(ladder.Z2.latt._lattice_drawer.draw_lattice())
    plt.plot(hs, W, "-", color="red", label="Wilson loop")
    plt.vlines(x=3.044, ymin=np.min(W), ymax=np.max(W))
    # plt.plot(hs,eig_first,'--',color='blue', label='exact 1º excited')
    # print(f"Point of max energy: {array[np.argmax(energies_h)]}")
# plt.title(f"Energy(h)")
# plt.xscale('symlog')
plt.xlabel("electric local parameter (h)")
# plt.ylim([-350, -50])
plt.legend()
plt.show()
# %%
plt.plot(hs, W, "-", color="red", label="Wilson loop")
plt.vlines(x=1 / 3.044, ymin=np.min(W), ymax=np.max(W))
# %%
