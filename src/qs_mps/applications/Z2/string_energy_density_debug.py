import numpy as np
from qs_mps.utils import load_list_of_lists, tensor_shapes
from qs_mps.mps_class import MPS
import matplotlib.pyplot as plt

path_tensor = "/Users/fradm98/Desktop/projects/1_Z2"
charges_x = [[4,6],[3,7],[2,8],[1,9]]
charges_y = [2,2]
precision = 2
interval = np.logspace(-1,1,20)

# # compute the electric energy density for the vacuum
# eed_vacuum = []
# corr_vacuum = []
# for h in interval:
#     lattice_mps = MPS(L=11,d=2**4, model="Z2_dual", chi=256, h=h, J=1, w=[])
#     lattice_mps.L = lattice_mps.L - 1
#     lattice_mps.load_sites(path=path_tensor, precision=precision)
#     lattice_mps.connected_correlator(site=1, lad=2)
#     # eed = lattice_mps.electric_energy_density_Z2(site=4)
#     # eed_vacuum.append(eed)

# for cx in charges_x:
#     eed_charges = []
#     corr_charges = []
#     for h in interval:
#         lattice_mps = MPS(L=11,d=2**4, model="Z2_dual", chi=256, h=h, w=[])
#         lattice_mps.L = lattice_mps.L - 1
#         lattice_mps.Z2.add_charges(cx,charges_y)
#         lattice_mps.load_sites(path=path_tensor, precision=precision, cx=cx, cy=charges_y)
#         lattice_mps.connected_correlator(site=1, lad=2)
#         # eed = lattice_mps.electric_energy_density_Z2(site=4)
#         # eed_charges.append(eed)
    
#     # eed_conn = (eed_charges - eed_vacuum)
#     corr_conn = (np.asarray(corr_charges) - np.asarray(corr_vacuum))

#     # plt.title(f"Energy density for charges {cx}")
#     # plt.imshow(eed_conn)
#     # plt.show()
#     plt.title(f"Correlations for charges {cx}")
#     plt.imshow(corr_conn)
#     plt.show()

from matplotlib import colors
corr_vacuum = np.load("/Users/fradm98/Library/CloudStorage/GoogleDrive-fra.di.marcantonio@gmail.com/My Drive/projects/1_Z2/results/mag_data/connected_correlator_s_4_l_2_Z2_dual_direct_lattice_4x10_vacuum_sector_[]-[]_h_-1.0-1.0_delta_20_chi_256.npy")
corr_charges = np.load("/Users/fradm98/Library/CloudStorage/GoogleDrive-fra.di.marcantonio@gmail.com/My Drive/projects/1_Z2/results/mag_data/connected_correlator_s_4_l_2_Z2_dual_direct_lattice_4x10_2_particle(s)_sector_[2, 8]-[2, 2]_h_-1.0-1.0_delta_20_chi_256.npy")

conn_corr = corr_charges - corr_vacuum
print(corr_vacuum[0])
plt.imshow(conn_corr.T/2,aspect=15, cmap="seismic", interpolation="gaussian",  norm=colors.SymLogNorm(linthresh=1e-4,linscale=1e-4,
                                              vmin=-1.0, vmax=1.0, base=10))
plt.hlines(y=2+(np.log(2)/2),xmin=0,xmax=100,colors='gold',linestyles=':',linewidth=2)
plt.hlines(y=2-(np.log(2)/2),xmin=0,xmax=100,colors='gold',linestyles=':',linewidth=2)
# plt.imshow(corr_coeff.T/2,aspect=5, cmap="viridis", interpolation="gaussian", vmin=-1, vmax=1)
plt.colorbar()
plt.show()