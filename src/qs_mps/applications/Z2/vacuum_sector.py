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


# finding the ground state of the vacuum sector
L = 5
l = 2
d = int(2**l)
chi = 16  # this is interpreted as d**(int(log2(chi))) --> e.g. chi=8 == 4**3=64
type_shape = "rectangular"
array = np.linspace(0.001, 10, 100)
hs = [h for h in array]
model = "Z2_dual"
path = "/Users/fradm98/Desktop/projects/1_Z2"
multpr = False
param = hs

if __name__ == "__main__":
    i = 0
    # for L in Ls:
    args_mps = {
        "L": L,
        "d": d,
        "chi": chi,
        "type_shape": type_shape,
        "model": model,
        "trunc_tol": False,
        "trunc_chi": True,
        "where": L // 2,
        "path": path,
        "save": True,
        "precision": 2,
        "sector": "vacuum_sector",
        "bond": True,
        "n_sweeps": 2,
        "conv_tol": 1e-10,
        "charges_x": None,
        "charges_y": None,
    }

    energies_h, entropy_h, schmidt_vals = ground_state_Z2(
        args_mps=args_mps, multpr=multpr, param=param
    )


# exact
eig_exact = []
eig_first = []
colors_mps = create_sequential_colors(20, "Blues")
colors = create_sequential_colors(20, "Reds")
markers = ["x","+","1","2"]

# alphas = [1,0.7]
l = 3
dof = (2*l*L - l - L)
print(dof)
v0 = np.array([-0.25 for _ in range(2**dof)])
for s in range(L-1):
    W = []
    W_exact = []
    for h in hs:
        print(f"h: {h}")
        ladder = MPS(L=L, d=d, model=model, h=h, eps=0, J=1, chi=chi)
        ladder.L = ladder.L - 1

        ladder.load_sites(path=path)
        ladder.Z2.wilson_Z2_dual(mpo_sites=[s], ls=[1]) #list(range(s))
        ladder.w = ladder.Z2.mpo
        W.append(ladder.mpo_first_moment().real)
    
        # exact
        Z2_exact = H_Z2_gauss(L=L, l=l, model="Z2", lamb=h, U=1e+3)
        H, e, v = Z2_exact.diagonalize(v0=v0)
        if s == 0:
            eig_exact.append(np.min(e))
        # eig_first.append(np.sort(e)[1])
        psi = v[:,0]
        v0 = psi
        # print("Psi exact:")
        # print(psi)

        loop = Z2_exact.latt.plaquettes(from_zero=True)
        plaq = Z2_exact.plaquette_term(loop[s])
        # plaq = Z2_exact.plaquette_term(loop[s+1])
        exp_val_wilson_loop = np.real(psi.T @ plaq @ psi)
        W_exact.append(exp_val_wilson_loop)


    # print(loop)
    plt.plot(hs, W, marker=markers[s], color="darkturquoise", label=f"mps {s}")
    plt.plot(hs, W_exact, markers[s], color="firebrick", label=f"exact {s}")


plt.xlabel("electric local parameter (h)")
plt.legend(loc="lower right")
plt.savefig("wilson_loop.png")
# plt.vlines(x=3.044, ymin=0, ymax=1)
# plt.title(f"Energy(h)")
# plt.xscale('symlog')
# plt.yscale('log')
# plt.xlabel("electric local parameter (h)")
# # plt.ylim([-350, -50])
# plt.legend(loc="lower right")
plt.figure().clear()
# plt.figure().clear()

fig = plt.figure()
plt.title(f"Energy(h)")
plt.scatter(hs,
            energies_h,
            marker="o",
            alpha=1,
            facecolors="none",
            edgecolors='g',
            label=f"mps L: {L}"
            )
plt.plot(hs,eig_exact,'--',color='red', label='exact gs')
plt.xlabel("electric local parameter (h)")
plt.legend(loc="lower right")
plt.savefig("energy.png")


print(ladder.Z2.latt._lattice_drawer.draw_lattice())


# print(ladder.Z2.latt._lattice_drawer.draw_lattice())

# plt.vlines(x=3.044, ymin=np.min(W), ymax=np.max(W))
# # plt.title(f"Energy(h)")
# # plt.yscale('log')
# plt.xlabel("electric local parameter (h)")
# # plt.ylim([-350, -50])
# plt.legend()
# plt.show()
# # %%
# plt.plot(hs, entropy_h)
# plt.vlines(x=3.044, ymin=np.min(entropy_h), ymax=np.max(entropy_h))