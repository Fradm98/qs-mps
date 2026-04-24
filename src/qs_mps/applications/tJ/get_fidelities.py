# imports
import numpy as np
from scipy.sparse import csc_array
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# default parameters of the plot layout
plt.rcParams["text.usetex"] = True  # use latex
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

from qs_mps.applications.Z2.exact_hamiltonian import *
from qs_mps.applications.ISING.utils import *
from qs_mps.sparse_hamiltonians_and_operators import *
from qs_mps.mps_class import MPS
from qs_mps.utils import anim, get_cx, get_cy

# metadata of the tensors
model = "heis"
chi = 200
h_i, h_f = -1.05, -0.9
a = 5e-3
precision = 4
couplings = np.arange(h_i,h_f,a)
npoints = len(couplings)
DMRG2 = False
d = 3
J = 1
eps_gs = -1e-2
Ls = [100,120,140,160,180]
path = "/Users/fradm/Desktop/projects/Fidelities_with_TN"

# main algorithm to get fiedelities
fidelity = []
for L in Ls:
    fidelity_L = []
    for k in range(len(couplings)-1):
        print(f"L: {L}, coupling: {couplings[k]:.4f}")
        heis_chain_g = MPS(L=L, d=d, model=model, chi=chi, h=couplings[k], eps=eps_gs, J=J, bc='obc')
        heis_chain_g_dg = MPS(L=L, d=d, model=model, chi=chi, h=couplings[k+1], eps=eps_gs, J=J, bc='obc')
        
        heis_chain_g.load_sites(path=path, precision=precision)
        heis_chain_g_dg.load_sites(path=path, precision=precision)
        heis_chain_g.ancilla_sites = heis_chain_g_dg.sites.copy()

        fidelity_L.append(heis_chain_g._compute_norm(site=1, mixed=True).copy())
    fidelity.append(fidelity_L)
fidelity = np.array(fidelity)

# plot discrete fidelity susceptibilities
colors = create_sequential_colors(len(Ls))

for i, L in enumerate(Ls):
    dfs = discrete_fidelity_susceptibility(fid=fidelity[i,:], a=a)
    # plt.plot(couplings[:-1], dfs, color=colors[i], label=f"$L: {L}$")
    plt.plot(couplings[:-1], dfs/L, color=colors[i], label=f"$L: {L}$")

plt.legend()
plt.savefig(f"{path}/figures/discrete_fid_sus_Ls_{Ls[0]}-{Ls[-1]}_h_{h_i}-{h_f}_npoints_{npoints}_eps_{eps_gs}_diff_eps_chi_{chi}.pdf", dpi=300, format="pdf")
