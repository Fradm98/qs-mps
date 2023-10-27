# %%
# import packages
from mps_class import MPS
from utils import *
import matplotlib.pyplot as plt
from ncon import ncon
import scipy
from scipy.sparse import csr_array
import time

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

# variational compression changing bond dimension
trotter_steps = 500
h_ev = 0.3
L = 30
model = "Ising"
flip = True
where = L//2

# fixed parameteres
t = 10
delta = t / trotter_steps
h_t = 0
n_sweeps = 8
# %%
# exact results with with the hamiltonian of Jesus

trotter_steps_j = 500
file_path_tot = f"D:/code/projects/0_ISING/results/exact/mag_data/mag_exact_tot_{model}_L_{L}_flip_{flip}_h_ev_{h_ev}_trotter_steps_{trotter_steps}_t_{t}"
file_path_loc = f"D:/code/projects/0_ISING/results/exact/mag_data/mag_exact_loc_{model}_L_{L}_flip_{flip}_h_ev_{h_ev}_trotter_steps_{trotter_steps}_t_{t}"
file_path_loc_X = f"D:/code/projects/0_ISING/results/exact/mag_data/mag_exact_loc_X_{model}_L_{L}_flip_{flip}_h_ev_{h_ev}_trotter_steps_{trotter_steps}_t_{t}"
file_path_entropy = f"D:/code/projects/0_ISING/results/exact/entropy/exact_entropy_{where}_{model}_L_{L}_flip_{flip}_h_ev_{h_ev}_trotter_steps_{trotter_steps}_t_{t}"
mag_tot_jesus = access_txt(file_path=file_path_tot, column_index=0)
mag_loc_Z_jesus = access_txt(file_path=file_path_loc, column_index=L//2)
mag_loc_X_jesus = access_txt(file_path=file_path_loc_X, column_index=0)
entropy_jesus = access_txt(file_path=file_path_entropy, column_index=0)
# %%
# visualization

# ---------------------------------------------------------
# total
# ---------------------------------------------------------
chis = [2,4,16,32,64,128]
plt.title(
    f"Total Magnetization for $\delta = {delta}$ ;" + " $h_{ev} =$" + f"{h_ev}",
    fontsize=14,
)
colors = create_sequential_colors(num_colors=len(chis), colormap_name="viridis")

for i, chi in enumerate(chis):
    mag_mps_tot = np.loadtxt(
        f"G:/My Drive/projects/0_ISING/results/mag_data/mag_mps_tot_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    plt.scatter(
        delta * np.arange(trotter_steps + 1),
        mag_mps_tot,
        s=25,
        marker="o",
        alpha=0.8,
        facecolors="none",
        edgecolors=colors[i],
        label=f"mps: $\chi={chi}$",
    )

# plt.plot(
#     delta * np.arange(trotter_steps + 1),
#     mag_tot_jesus,
#     color="indianred",
#     label=f"Exact L={L}",
# )
# plt.plot(
#     delta * np.arange(trotter_steps + 1),
#     mag_exact_tot,
#     color="indianred",
#     label=f"exact: $L={L}$",
# )
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\sum_{i=1}^L \sigma_i^z$")
plt.legend()
# plt.savefig(f"G:/My Drive/projects/0_ISING/figures/magnetization/total_mag_L_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()

# ---------------------------------------------------------
# local Z
# ---------------------------------------------------------
plt.title(
    f"Magnetization Z in the middle site for $\delta = {delta}$ ;"
    + " $h_{ev} =$"
    + f"{h_ev}",
    fontsize=14,
)
for i, chi in enumerate(chis):
    mag_mps_tot = np.loadtxt(
        f"G:/My Drive/projects/0_ISING/results/mag_data/mag_mps_loc_Z_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    plt.scatter(
        delta * np.arange(trotter_steps + 1),
        mag_mps_tot,
        s=25,
        marker="o",
        alpha=0.8,
        facecolors="none",
        edgecolors=colors[i],
        label=f"mps: $\chi={chi}$",
    )

# plt.plot(
#     delta * np.arange(trotter_steps + 1),
#     mag_loc_Z_jesus,
#     color="indianred",
#     label=f"Exact L={L}",
# )
# plt.plot(
#     delta * np.arange(trotter_steps + 1),
#     mag_exact_tot,
#     color="indianred",
#     label=f"exact: $L={L}$",
# )
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\sigma_{L/2}^z$")
plt.legend()
# plt.savefig(f"G:/My Drive/projects/0_ISING/figures/magnetization/local_mag_Z_L_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()

# ---------------------------------------------------------
# local X
# ---------------------------------------------------------
plt.title(
    f"Magnetization X in the middle site for $\delta = {delta}$ ;"
    + " $h_{ev} =$"
    + f"{h_ev}",
    fontsize=14,
)
for i, chi in enumerate(chis):
    mag_mps_tot = np.loadtxt(
        f"G:/My Drive/projects/0_ISING/results/mag_data/mag_mps_loc_X_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    plt.scatter(
        delta * np.arange(trotter_steps + 1),
        mag_mps_tot,
        s=25,
        marker="o",
        alpha=0.8,
        facecolors="none",
        edgecolors=colors[i],
        label=f"mps: $\chi={chi}$",
    )

# plt.plot(
#     delta * np.arange(trotter_steps + 1),
#     mag_loc_X_jesus,
#     color="indianred",
#     label=f"Exact L={L}",
# )
# plt.plot(
#     delta * np.arange(trotter_steps + 1),
#     mag_exact_tot,
#     color="indianred",
#     label=f"exact: $L={L}$",
# )
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\sigma_{L/2}^x$")
plt.legend()
# plt.savefig(f"G:/My Drive/projects/0_ISING/figures/magnetization/local_mag_X_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()

# %%
# ---------------------------------------------------------
# Local data
# ---------------------------------------------------------
# data1 = mag_exact_loc
# data2 = mag_mps_loc
# title1 = "Exact quench (local mag)"
# title2 = f"MPS quench (local mag) $\chi={chi}$"
# plot_side_by_side(
#     data1=data1, data2=data2, cmap="seismic", title1=title1, title2=title2
# )
for chi in chis:
    mag_mps_loc = np.loadtxt(
        f"G:/My Drive/projects/0_ISING/results/mag_data/mag_mps_loc_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    data1 = mag_mps_loc
    title1 = f"MPS quench (local mag) $\chi={chi}$"
    cmap = "seismic"

    # Plot the imshow plots on the subplots
    plt.imshow(data1.T, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ticks = np.arange(trotter_steps + 1)
    labels = delta * ticks
    steps = len(ticks) // 5
    ticks = ticks[::steps]
    labels = labels[::steps]
    plt.xticks(ticks=ticks, labels=labels.astype(int))
    # Set titles for the subplots
    plt.title(title1)

    # Remove ticks from the colorbar subplot

    # Create a colorbar for the second plot on the right
    # plt.colorbar()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

# %%
# ---------------------------------------------------------
# fidelity
# ---------------------------------------------------------
# plt.title(
#     "Fidelity $\left<\psi_{MPS}(t)|\psi_{exact}(t)\\right>$: "
#     + f"$\delta = {delta}$; $h_{{t-ev}} = {h_ev}$"
# )
# for i in range(1, L // 2 + 1):  # L//2+1
#     chi = 2**i
#     fidelity = np.loadtxt(
#         f"results/fidelity_data/fidelity_L_{L}_delta_{delta}_chi_{chi}"
#     )
#     errors = load_list_of_lists(
#         f"results/errors_data/errors_L_{L}_delta_{delta}_chi_{chi}"
#     )
#     plt.plot(delta * np.arange(trotter_steps + 1),[1-delta**2-float(error[-1]) for _ , error in zip(range(trotter_steps+1), errors)],
#              '--',
#              color=colors[i - 1],
#              alpha=0.6,
#              label=f"trotter + trunc error $\chi={chi}$")

#     plt.scatter(
#         delta * np.arange(trotter_steps + 1),
#         fidelity,
#         s=20,
#         marker="o",
#         alpha=0.7,
#         facecolors="none",
#         edgecolors=colors[i - 1],
#         label=f"fidelity $\chi={chi}$",
#     )

# plt.xlabel("time (t = $\delta$ T)")
# plt.legend()
# plt.show()

# ---------------------------------------------------------
# errors
# ---------------------------------------------------------

chis = [2, 4, 16, 32, 64, 128]
colors = create_sequential_colors(num_colors=len(chis), colormap_name="viridis")
plt.title(f"Truncation error $vs$ trotter steps", fontsize=14)
for i, chi in enumerate(chis):  # L//2+1
    errors = load_list_of_lists(
        f"G:/My Drive/projects/0_ISING/results/errors_data/errors_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    last_errors = [float(sublist[-1]) for sublist in errors]
    # last_error, num_zeros = replace_zeros_with_nan(last_errors)
    plt.scatter(
        np.arange(trotter_steps + 1),
        last_errors,
        marker="o",
        s=10,
        alpha=0.6,
        # color=colors[i],
        facecolors=colors[i],
        edgecolors=colors[i],
        label=f"$\left|\left| |\phi\\rangle - |\psi\\rangle \\right|\\right|^2$ $\chi={chi}$",
    )

plt.ylabel(
    "$\mathcal{A^* N A}$ - 2$\Re(\mathcal{A^*M})$ + $\langle \psi| \psi\\rangle$"
)
plt.xlabel("Trotter Steps (T)")
plt.yscale("log")
# plt.ylim(1e-17,1e-14)
plt.legend()
# plt.savefig(f"D:/code/projects/0_ISING/figures/errors_L_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()

# %%
# ---------------------------------------------------------
# entropy
# -------------------------------------------------------

plt.title(
    f" {where}-th Bond Entanglement Entropy: "
    + f"L={L}, $\delta = {delta}$; $h_{{t-ev}} = {h_ev}$",
    fontsize=14,
)
chis = [2, 4, 16, 32, 64, 128]
colors = create_sequential_colors(num_colors=len(chis), colormap_name="viridis")

for i, chi in enumerate(chis):  # L//2+1
    entropy_chi = [0]
    schmidt_vals = load_list_of_lists(
        f"G:/My Drive/projects/0_ISING/results/bonds_data/{where}_bond_schmidt_values_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    for s in schmidt_vals:
        entropy = von_neumann_entropy(np.asarray(s))
        entropy_chi.append(entropy)

    plt.scatter(
        np.arange(trotter_steps + 1),
        np.asarray(entropy_chi),
        s=28,
        marker="+",
        alpha=0.7,
        facecolors=colors[i],
        # edgecolors=colors[i],
        label=f"mps: $\chi={chi}$",
    )

# plt.plot(
#     np.arange(trotter_steps + 1),
#     entropy_jesus,
#     color="indianred",
#     label="$S_{exact}$",
# )
ticks = np.arange(trotter_steps + 1)
labels = delta * np.arange(trotter_steps + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
labels = labels[::steps]
plt.xticks(ticks=ticks, labels=labels)
plt.ylabel("entanglement von neumann entropy $(S_{\chi})$")
plt.xlabel("time (t = $\delta$ T)")
plt.legend(fontsize=10)
# plt.savefig(f"D:/code/projects/0_ISING/figures/entropy/entropy_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()
# %%
## errors with exact
colors = create_sequential_colors(num_colors=len(chis), colormap_name="viridis")

# -----------------------
# total magnetization
# -----------------------
plt.title(
    f"Error Total Magnetization for $\delta = {delta}$ ;" + " $h_{ev} =$" + f"{h_ev}",
    fontsize=14,
)
for i, chi in enumerate(chis):
    mag_mps_tot = np.loadtxt(
        f"D:/code/projects/0_ISING/results/mag_data/mag_mps_tot_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    # mag_mps_tot = mag_mps_tot[::5]
    # error_mag_tot = np.abs(np.asarray(mag_tot_maricarmen) - np.asarray(mag_mps_tot))
    error_mag_tot = np.abs(np.asarray(mag_tot_jesus) - np.asarray(mag_mps_tot))
    plt.plot(
        delta * np.arange(trotter_steps + 1),
        error_mag_tot,
        color=colors[i],
        label=f"$\chi={chi}$",
    )
ticks = delta * np.arange(trotter_steps + 1)
labels = delta * np.arange(trotter_steps + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
steps = len(labels) // 5
labels = labels[::steps]
plt.xticks(ticks=ticks, labels=labels)
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\left|M^{exact} - M^{mps}\\right|$")
plt.yscale("log")
plt.legend()
# plt.savefig(f"D:/code/projects/0_ISING/figures/magnetization/error_total_mag_L_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()
# -----------------------
# local Z magnetization
# -----------------------
plt.title(
    f"Error Middle Chain Z Magnetization for $\delta = {delta}$ ;"
    + " $h_{ev} =$"
    + f"{h_ev}",
    fontsize=14,
)
for i, chi in enumerate(chis):
    mag_mps_loc = np.loadtxt(
        f"D:/code/projects/0_ISING/results/mag_data/mag_mps_loc_Z_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    # mag_mps_loc = mag_mps_loc[::5]
    error_mag_tot = np.abs(np.asarray(mag_loc_Z_jesus) - np.asarray(mag_mps_loc))
    # error_mag_tot = np.abs(np.asarray(mag_loc_Z_jesus) - np.asarray(mag_mps_loc))
    plt.plot(
        delta * np.arange(trotter_steps + 1),
        error_mag_tot,
        color=colors[i],
        label=f"$\chi={chi}$",
    )
ticks = delta * np.arange(trotter_steps + 1)
labels = delta * np.arange(trotter_steps + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
steps = len(labels) // 5
labels = labels[::steps]
plt.xticks(ticks=ticks, labels=labels)
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\left|{\sigma_{L/2}^z}^{ex} - {\sigma_{L/2}^z}^{mps}\\right|$")
plt.yscale("log")
plt.legend()
# plt.savefig(f"D:/code/projects/0_ISING/figures/magnetization/error_local_mag_Z_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()
# -----------------------
# local X magnetization
# -----------------------
plt.title(
    f"Error Middle Chain X Magnetization for $\delta = {delta}$ ;"
    + " $h_{ev} =$"
    + f"{h_ev}",
    fontsize=14,
)
for i, chi in enumerate(chis):
    mag_mps_loc = np.loadtxt(
        f"D:/code/projects/0_ISING/results/mag_data/mag_mps_loc_X_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    # mag_mps_loc = mag_mps_loc[::5]
    error_mag_tot = np.abs(np.asarray(mag_loc_X_jesus) - np.asarray(mag_mps_loc))
    plt.plot(
        delta * np.arange(trotter_steps + 1),
        error_mag_tot,
        color=colors[i],
        label=f"$\chi={chi}$",
    )
ticks = delta * np.arange(trotter_steps + 1)
labels = delta * np.arange(trotter_steps + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
steps = len(labels) // 5
labels = labels[::steps]
plt.xticks(ticks=ticks, labels=labels)
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\left|{\sigma_{L/2}^x}^{ex} - {\sigma_{L/2}^x}^{mps}\\right|$")
plt.yscale("log")
plt.legend()
# plt.savefig(f"D:/code/projects/0_ISING/figures/magnetization/error_local_mag_X_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()
# -----------------------
# entropy
# -----------------------
plt.title(
    f"Error {where}-th Bond Entanglement Entropy: "
    + f"L={L}, $\delta = {delta}$; $h_{{t-ev}} = {h_ev}$",
    fontsize=14,
)
for i, chi in enumerate(chis):  # L//2+1
    entropy_chi = [0]
    schmidt_vals = load_list_of_lists(
        f"D:/code/projects/0_ISING/results/bonds_data/{where}_bond_schmidt_values_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    for s in schmidt_vals:
        entropy = von_neumann_entropy(np.asarray(s))
        entropy_chi.append(entropy)
    # entropy_chi_red = entropy_chi[::5]
    error_mc = np.abs(np.asarray(entropy_jesus) - np.asarray(entropy_chi))
    plt.plot(
        delta * np.arange(trotter_steps + 1), error_mc, color=colors[i], label=f" $\chi={chi}$"
    )
ticks = delta * np.arange(trotter_steps + 1)
labels = delta * np.arange(trotter_steps + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
steps = len(labels) // 5
labels = labels[::steps]
plt.xticks(ticks=ticks, labels=labels)
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$|S_{exact} - S_{mps}|$")
plt.yscale("log")
plt.legend()
# plt.savefig(f"D:/code/projects/0_ISING/figures/entropy/error_entropy_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()
# %%
