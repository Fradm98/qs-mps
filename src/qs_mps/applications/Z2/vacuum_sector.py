# %%
# import packages
from qs_mps.mps_class import MPS
import matplotlib.pyplot as plt
from qs_mps.utils import create_sequential_colors, tensor_shapes
import numpy as np
from .ground_state_multiprocessing import ground_state_Z2
import time

# %%
# finding the ground state of the vacuum sector
Ls = [4]
d = 4
chi = 16 # this is interpreted as d**(int(log2(chi))) --> e.g. chi=8 == 4**3=64
array = np.linspace(100,1000,10)
hs = [h for h in array]
model = "Z2_two_ladder"
charges = [1,1,-1,-1,1,1]
energies_h = []
multpr = False
param = hs
if __name__ == '__main__':
    colors = create_sequential_colors(len(Ls), 'viridis')
    i = 0
    for L in Ls:
        args_mps = {"L": L, "d": d, "chi": chi, "model": model, "charges": charges, "trunc_tol": True, "trunc_chi": False}

        energies_h = ground_state_Z2(args_mps=args_mps, multpr=multpr, param=param)

        plt.scatter(hs,
                    energies_h,
                    marker="o",
                    alpha=0.8,
                    facecolors="none",
                    edgecolors=colors[i],
                    label=f"L: {L}"
                    )
        print(f"Point of max energy: {array[np.argmax(energies_h)]}")
        i += 1
    plt.title(f"Energy(h)")
    plt.xlabel("eletric local parameter (h)")
    # plt.ylim([-350, -50])
    plt.legend()
    plt.show()
    
# %%
