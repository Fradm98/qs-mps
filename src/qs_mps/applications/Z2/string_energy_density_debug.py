import numpy as np
from qs_mps.utils import load_list_of_lists, tensor_shapes
from qs_mps.mps_class import MPS
import matplotlib.pyplot as plt

path_tensor = "/Users/fradm98/Desktop/projects/1_Z2"
charges_x = [0, 4]
charges_y = [2, 2]
precision = 3
interval = np.linspace(0.1, 1.0, 100)
L = 5
l = 4
chi = 16

# compute the electric energy density for the vacuum
eed_vacuum = []
corr_vacuum = []
for h in interval:
    print(f"vacuum for h: {h}")
    lattice_mps = MPS(L=L, d=2**l, model="Z2_dual", chi=chi, h=h)
    lattice_mps.L = lattice_mps.L - 1
    lattice_mps.load_sites(path=path_tensor, precision=precision)
    eed = lattice_mps.electric_energy_density_Z2(site=L // 2 - 1)
    # corr = lattice_mps.connected_correlator(site=L//2, lad=2)
    eed_vacuum.append(eed)
    # corr_vacuum.append(corr)

eed_charges = []
corr_charges = []
for h in interval:
    print(f"charges for h: {h}")
    lattice_mps = MPS(L=L, d=2**l, model="Z2_dual", chi=chi, h=h)
    lattice_mps.L = lattice_mps.L - 1
    lattice_mps.Z2.add_charges(charges_x, charges_y)
    lattice_mps.load_sites(
        path=path_tensor, precision=precision, cx=charges_x, cy=charges_y
    )
    eed = lattice_mps.electric_energy_density_Z2(site=L // 2 - 1)
    # corr = lattice_mps.connected_correlator(site=L//2, lad=2)
    eed_charges.append(eed)
    # corr_charges.append(corr)

eed_conn = np.array(eed_charges) - np.array(eed_vacuum)
# corr_conn = (np.array(corr_charges) - np.array(corr_vacuum))

plt.title(f"Energy density")
plt.imshow(eed_conn, aspect=0.1, origin="lower")

# plt.title(f"Correlations")
# plt.imshow(corr_conn, aspect=0.1, origin='lower')
plt.colorbar()
plt.show()

eed_string = []
for eed_lad in eed_conn:
    eed_sum_lad = 0
    for i, eed_x in enumerate(eed_lad):
        eed_sum_lad += eed_lad * ((i + 1) ** 2)
    eed_sum_lad = eed_sum_lad / sum(eed_lad)
    eed_string.append(eed_sum_lad)

plt.plot(eed_string)
plt.show()
# from matplotlib import colors
# corr_vacuum = np.load("/Users/fradm98/Library/CloudStorage/GoogleDrive-fra.di.marcantonio@gmail.com/My Drive/projects/1_Z2/results/mag_data/connected_correlator_s_4_l_2_Z2_dual_direct_lattice_4x10_vacuum_sector_[]-[]_h_-1.0-1.0_delta_20_chi_256.npy")
# corr_charges = np.load("/Users/fradm98/Library/CloudStorage/GoogleDrive-fra.di.marcantonio@gmail.com/My Drive/projects/1_Z2/results/mag_data/connected_correlator_s_4_l_2_Z2_dual_direct_lattice_4x10_2_particle(s)_sector_[2, 8]-[2, 2]_h_-1.0-1.0_delta_20_chi_256.npy")

# conn_corr = np.array([corr_charges - corr_vacuum[:,0,i] for i in range(5)]).reshape((20,5))
# print(conn_corr)
# plt.imshow(conn_corr.T/2,aspect=1, cmap="seismic", interpolation="gaussian",  norm=colors.SymLogNorm(linthresh=1e-4,linscale=1e-4,
#                                               vmin=-1.0, vmax=1.0, base=10))
# plt.hlines(y=2+(np.log(2)/2),xmin=0,xmax=19,colors='gold',linestyles=':',linewidth=2)
# plt.hlines(y=2-(np.log(2)/2),xmin=0,xmax=19,colors='gold',linestyles=':',linewidth=2)
# # plt.imshow(corr_coeff.T/2,aspect=5, cmap="viridis", interpolation="gaussian", vmin=-1, vmax=1)
# plt.colorbar()
# plt.show()
