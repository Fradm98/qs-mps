import os
import sys

# Add the root directory (containing 'src') to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# import packages
from src.mps.mps_class import MPS
import matplotlib.pyplot as plt
from src.mps._mps_class.utils import create_sequential_colors, tensor_shapes
import numpy as np

# finding the ground state of the vacuum sector
L = 20
d = 4
chi = 4 # this is interpreted as d**(int(log2(chi))) --> e.g. chi=8 == 4**3=64
array = np.linspace(0.55,0.65,25)
hs = [h for h in array]
model = "Z2_two_ladder"
charges = [1,1,1,1,1,1]
energies_h = []
colors = create_sequential_colors(len(hs), 'viridis')
i = 0
for h in hs:
    ladder = MPS(L=L, d=d, model=model, chi=chi, charges=charges, h=h)
    ladder._random_state(seed = 7, chi=chi)
    ladder.canonical_form()
    energy = ladder.sweeping(trunc_tol=True, trunc_chi=False)
    plt.title(f"Energy during the sweeping")
    plt.plot(energy, color=colors[i])
    plt.xlabel("Sweeping iterations")
    energies_h.append(energy[-1])
    i += 1
plt.show()

plt.title(f"Energy(h)")
plt.scatter(hs,
            energies_h,
            marker="o",
            alpha=0.8,
            facecolors="none",
            edgecolors="deepskyblue",
            )
plt.xlabel("eletric local parameter (h)")
plt.show()
print(array[np.argmax(energies_h)])


