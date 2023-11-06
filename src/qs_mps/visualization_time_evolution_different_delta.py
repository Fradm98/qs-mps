# %%
# import packages
from qs_mps.mps_class import MPS
from qs_mps.utils import *
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
chi = 16
h_ev = 0.3
L = 15
model = "Ising"
flip = True

# fixed parameteres
t = 10
h_t = 0
n_sweeps = 8
# where = L//2 - 1
where = "all"

# %%
# exact results
""" 
info:
[t,magZ,magX,S,magXtot,magZtot]
"""
if flip:
    file_path = f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/maricarmen_results/results_L15_Z0_flip_g0_3.txt"
    mag_tot_maricarmen = [L - 2]
    mag_loc_Z_maricarmen = [-1]

else:
    file_path = f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/maricarmen_results/results_L15_Z0_g0_3.txt"
    mag_tot_maricarmen = [L]
    mag_loc_Z_maricarmen = [1]

column_index = 5
mag_tot_maricarmen = mag_tot_maricarmen + access_txt(
    file_path=file_path, column_index=column_index
)
mag_tot_maricarmen.pop(-1)

column_index = 1
mag_loc_Z_maricarmen = mag_loc_Z_maricarmen + access_txt(
    file_path=file_path, column_index=column_index
)
mag_loc_Z_maricarmen.pop(-1)

column_index = 2
mag_loc_X_maricarmen = [0]
mag_loc_X_maricarmen = mag_loc_X_maricarmen + access_txt(
    file_path=file_path, column_index=column_index
)
mag_loc_X_maricarmen.pop(-1)

column_index = 3
entropy_maricarmen = [0]
entropy_maricarmen = entropy_maricarmen + access_txt(
    file_path=file_path, column_index=column_index
)
entropy_maricarmen.pop(-1)
# %%
# results with with the hamiltonian of Jesus
trotter_steps_j = 100
delta_j = t / trotter_steps_j
file_path_tot = f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/exact/mag_data/mag_exact_tot_{model}_L_{L}_flip_{flip}_h_ev_{h_ev}_trotter_steps_{trotter_steps_j}_t_{t}"
file_path_loc = f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/exact/mag_data/mag_exact_loc_{model}_L_{L}_flip_{flip}_h_ev_{h_ev}_trotter_steps_{trotter_steps_j}_t_{t}"
file_path_loc_Z = f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/exact/mag_data/mag_exact_loc_Z_{model}_L_{L}_flip_{flip}_h_ev_{h_ev}_trotter_steps_{trotter_steps_j}_t_{t}"
file_path_loc_X = f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/exact/mag_data/mag_exact_loc_X_{model}_L_{L}_flip_{flip}_h_ev_{h_ev}_trotter_steps_{trotter_steps_j}_t_{t}"
file_path_entr = f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/exact/entropy/exact_entropy_{where}_{model}_L_{L}_flip_{flip}_h_ev_{h_ev}_trotter_steps_{trotter_steps_j}_t_{t}"
file_path_entr_tot = f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/exact/entropy/exact_entropy_tot_{model}_L_{L}_flip_{flip}_h_ev_{h_ev}_trotter_steps_{trotter_steps_j}_t_{t}"
mag_loc_jesus = np.loadtxt(fname=file_path_loc)
mag_loc_Z_jesus = access_txt(file_path=file_path_loc_Z, column_index=0)
mag_tot_jesus = access_txt(file_path=file_path_tot, column_index=0)
mag_loc_X_jesus = access_txt(file_path=file_path_loc_X, column_index=0)
entropy_jesus = access_txt(file_path=file_path_entr, column_index=0)

# # %%
# # visualization

# # ---------------------------------------------------------
# # total
# # ---------------------------------------------------------
# trotter_steps = [500, 1000, 10000]
# plt.title(
#     f"Total Magnetization for $\chi = {chi}$ ;" + " $h_{ev} =$" + f"{h_ev}", fontsize=14
# )
# colors = create_sequential_colors(
#     num_colors=len(trotter_steps), colormap_name="viridis"
# )

# mult_factor = [5, 10, 100]
# markers = ["+","x","1"]
# for i, trotter_step in enumerate(trotter_steps):
#     delta = t / trotter_step
#     mag_mps_tot = np.loadtxt(
#         f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/mag_data/mag_mps_tot_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
#     )
#     plt.scatter(
#         delta * np.arange(trotter_step + 1),
#         mag_mps_tot,
#         s=25,
#         marker=markers[i],
#         alpha=0.8,
#         # facecolors="none",
#         color=colors[i],
#         label=f"mps: $\delta={delta}$",
#     )

# # plt.plot(
# #     mult_factor[i] * delta * np.arange(100 + 1),
# #     mag_tot_maricarmen,
# #     color="indianred",
# #     label=f"Exact L={L}",
# # )
# plt.plot(
#     mult_factor[i] * delta * np.arange(100 + 1),
#     mag_tot_jesus,
#     color="indianred",
#     label=f"Exact L={L}",
# )

# plt.xlabel("time (t = $\delta$ T)")
# plt.ylabel("$\sum_{i=1}^L \sigma_i^z$")
# plt.legend()
# plt.savefig(
#     f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/magnetization/total_mag_L_{L}_flip_{flip}_chi_{chi}_h_ev_{h_ev}.png"
# )
# plt.show()

# # ---------------------------------------------------------
# # local Z
# # ---------------------------------------------------------
# plt.title(
#     f"Magnetization Z in the middle site for $\chi = {chi}$ ;"
#     + " $h_{ev} =$"
#     + f"{h_ev}",
#     fontsize=14,
# )
# for i, trotter_step in enumerate(trotter_steps):
#     delta = t / trotter_step
#     mag_mps_tot = np.loadtxt(
#         f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/mag_data/mag_mps_loc_Z_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
#     )
#     plt.scatter(
#         delta * np.arange(trotter_step + 1),
#         mag_mps_tot,
#         s=25,
#         marker=markers[i],
#         alpha=0.8,
#         # facecolors="none",
#         color=colors[i],
#         label=f"mps: $\delta={delta}$",
#     )

# # plt.plot(
# #     mult_factor[i] * delta * np.arange(100 + 1),
# #     mag_loc_Z_maricarmen,
# #     color="indianred",
# #     label=f"Exact L={L}",
# # )
# plt.plot(
#     mult_factor[i] * delta * np.arange(100 + 1),
#     mag_loc_Z_jesus,
#     color="indianred",
#     label=f"Exact L={L}",
# )
# plt.xlabel("time (t = $\delta$ T)")
# plt.ylabel("$\sigma_{L/2}^z$")
# plt.legend()
# plt.savefig(
#     f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/magnetization/local_mag_Z_L_{L}_flip_{flip}_chi_{chi}_h_ev_{h_ev}.png"
# )
# plt.show()

# # ---------------------------------------------------------
# # local X
# # ---------------------------------------------------------
# plt.title(
#     f"Magnetization X in the middle site for $\chi = {chi}$ ;"
#     + " $h_{ev} =$"
#     + f"{h_ev}",
#     fontsize=14,
# )
# for i, trotter_step in enumerate(trotter_steps):
#     delta = t / trotter_step
#     mag_mps_tot = np.loadtxt(
#         f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/mag_data/mag_mps_loc_X_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
#     )
#     plt.scatter(
#         delta * np.arange(trotter_step + 1),
#         mag_mps_tot,
#         s=25,
#         marker=markers[i],
#         alpha=0.8,
#         # facecolors="none",
#         color=colors[i],
#         label=f"mps: $\delta={delta}$",
#     )

# # plt.plot(
# #     mult_factor[i] * delta * np.arange(100 + 1),
# #     mag_loc_X_maricarmen,
# #     color="indianred",
# #     label=f"Exact L={L}",
# # )
# plt.plot(
#     mult_factor[i] * delta * np.arange(100 + 1),
#     mag_loc_X_jesus,
#     color="indianred",
#     label=f"Exact L={L}",
# )

# plt.xlabel("time (t = $\delta$ T)")
# plt.ylabel("$\sigma_{L/2}^x$")
# plt.legend()
# plt.savefig(
#     f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/magnetization/local_mag_X_{L}_flip_{flip}_chi_{chi}_h_ev_{h_ev}.png"
# )
# plt.show()

# # %%
# # ---------------------------------------------------------
# # Local data
# # ---------------------------------------------------------
# # data1 = mag_exact_loc
# # data2 = mag_mps_loc
# # title1 = "Exact quench (local mag)"
# # title2 = f"MPS quench (local mag) $\delta={delta}$"
# # plot_side_by_side(
# #     data1=data1, data2=data2, cmap="seismic", title1=title1, title2=title2
# # )
# for i, trotter_step in enumerate(trotter_steps):
#     delta = t / trotter_step
#     mag_mps_loc = np.loadtxt(
#         f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/mag_data/mag_mps_loc_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
#     )
#     data1 = mag_mps_loc
#     title1 = f"MPS quench (local mag) $\delta={delta}$"
#     cmap = "seismic"

#     # Plot the imshow plots on the subplots
#     plt.imshow(data1.T, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
#     ticks = np.arange(trotter_step + 1)
#     labels = delta * ticks
#     steps = len(ticks) // 5
#     ticks = ticks[::steps]
#     labels = labels[::steps]
#     plt.xticks(ticks=ticks, labels=labels.astype(int))
#     # Set titles for the subplots
#     plt.title(title1)

#     # Remove ticks from the colorbar subplot

#     # Create a colorbar for the second plot on the right
#     # plt.colorbar()

#     # Adjust layout and display the plot
#     plt.tight_layout()
#     plt.show()

# # %%
# ---------------------------------------------------------
# fidelity
# ---------------------------------------------------------
# plt.title(
#     "Fidelity $\left<\psi_{MPS}(t)|\psi_{exact}(t)\\right>$: "
#     + f"$\chi = {chi}$; $h_{{t-ev}} = {h_ev}$"
# )
# for i in range(1, L // 2 + 1):  # L//2+1
#     chi = 2**i
#     fidelity = np.loadtxt(
#         f"results/fidelity_data/fidelity_L_{L}_delta_{delta}_chi_{chi}"
#     )
#     errors = load_list_of_lists(
#         f"results/errors_data/errors_L_{L}_delta_{delta}_chi_{chi}"
#     )
#     plt.plot(delta * np.arange(trotter_step + 1),[1-delta**2-float(error[-1]) for _ , error in zip(range(trotter_step+1), errors)],
#              '--',
#              color=colors[i - 1],
#              alpha=0.6,
#              label=f"trotter + trunc error $\delta={delta}$")

#     plt.scatter(
#         delta * np.arange(trotter_step + 1),
#         fidelity,
#         s=20,
#         marker="o",
#         alpha=0.7,
#         facecolors="none",
#         edgecolors=colors[i - 1],
#         label=f"fidelity $\delta={delta}$",
#     )

# plt.xlabel("time (t = $\delta$ T)")
# plt.legend()
# plt.show()

# # ---------------------------------------------------------
# # errors
# # ---------------------------------------------------------

# colors = create_sequential_colors(
#     num_colors=len(trotter_steps), colormap_name="viridis"
# )
# plt.title(f"Truncation error $vs$ trotter steps", fontsize=14)
# for i, trotter_step in enumerate(trotter_steps):
#     delta = t / trotter_step
#     errors = load_list_of_lists(
#         f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/errors_data/errors_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
#     )
#     last_errors = [float(sublist[-1]) for sublist in errors]
#     # last_error, num_zeros = replace_zeros_with_nan(last_errors)
#     plt.scatter(
#         np.arange(trotter_step + 1),
#         last_errors,
#         marker="o",
#         s=10,
#         alpha=0.6,
#         # color=colors[i],
#         facecolors=colors[i],
#         edgecolors=colors[i],
#         label=f"$\left|\left| |\phi\\rangle - |\psi\\rangle \\right|\\right|^2$ $\delta={delta}$",
#     )

# plt.ylabel(
#     "$\mathcal{A^* N A}$ - 2$\Re(\mathcal{A^*M})$ + $\langle \psi| \psi\\rangle$"
# )
# plt.xlabel("Trotter Steps (T)")
# plt.yscale("log")
# # plt.ylim(1e-17,1e-14)
# plt.legend()
# plt.savefig(
#     f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/error/errors_L_{L}_flip_{flip}_chi_{chi}_h_ev_{h_ev}.png"
# )
# plt.show()

# %%
# ---------------------------------------------------------
# entropy
# -------------------------------------------------------
trotter_steps = [500, 1000, 10000] # , 
entr_factor = [20, 10, 1]
mult_factor = [5, 10, 100]

plt.title(
    "Middle Chain Entanglement Entropy: "
    + f"L={L}, $\chi = {chi}$; $h_{{t-ev}} = {h_ev}$",
    fontsize=14,
)

colors = create_sequential_colors(
    num_colors=len(trotter_steps), colormap_name="viridis"
)

for i, trotter_step in enumerate(trotter_steps):
    delta = t / trotter_step
    entropy_chi = [0]
    schmidt_vals = np.loadtxt(
        f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/bonds_data/middle_chain_schmidt_values_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    for s in schmidt_vals:
        entropy = von_neumann_entropy(s)
        entropy_chi.append(entropy)

    # plt.scatter(
    #     entr_factor[i] * np.arange(trotter_step + 1),
    #     np.asarray(entropy_chi),
    #     s=28,
    #     marker="+",
    #     alpha=0.7,
    #     facecolors=colors[i],
    #     # edgecolors=colors[i],
    #     label=f"mps: $\delta={delta}$",
    # )
    entropy_chi = entropy_chi[::mult_factor[i]]
    plt.scatter(
        mult_factor[0] * np.arange(len(entropy_jesus)),
        np.asarray(entropy_chi),
        s=28,
        marker="+",
        alpha=0.7,
        facecolors=colors[i],
        # edgecolors=colors[i],
        label=f"mps: $\delta={delta}$",
    )

# for i in range(L-3):
#     entropy_jesus = access_txt(file_path=file_path_entr, column_index=i)
#     plt.plot(
#     np.arange(len(entropy_jesus) + 1),
#     entropy_jesus,
#     alpha=(0.06*i+0.05),
#     color="indianred",
#     label="$S_{exact}$" + f" site: {i+1}",
# )
# plt.plot(
#     mult_factor[i] * np.arange(len(entropy_jesus) + 1),
#     entropy_jesus,
#     color="indianred",
#     label="$S_{exact}$",
# )
plt.plot(
    mult_factor[0] * np.arange(len(entropy_jesus)),
    entropy_jesus,
    color="indianred",
    label="$S_{exact}$" + f" bond: {where}",
)
# for i in range(L-3):
#     entropy_tot_jesus = access_txt(file_path=file_path_entr_tot, column_index=i)
#     plt.plot(
#         mult_factor[-1] * np.arange(100 + 1),
#         entropy_jesus,
#         color="indianred",
#         label=f"i = {i+2}",
#         alpha=0.5,
#         linewidth=0.1,
#     )

ticks = np.arange(trotter_steps[0] + 1)
labels = t/trotter_steps[0] * np.arange(trotter_steps[0] + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
labels = labels[::steps]
plt.xticks(ticks=ticks, labels=labels)
plt.ylabel("entanglement von neumann entropy $(S_{\chi})$")
plt.xlabel("time (t = $\delta$ T)")
plt.legend(fontsize=10)
plt.savefig(
    f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/entropy/entropy_{where}_{L}_flip_{flip}_chi_{chi}_h_ev_{h_ev}.png"
)
plt.show()

# %%
## errors with exact
trotter_steps = [500, 1000, 10000]
mult_factor = [5, 10, 100]
colors = create_sequential_colors(
    num_colors=len(trotter_steps), colormap_name="viridis"
)

# -----------------------
# total magnetization
# -----------------------
plt.title(
    f"Error Total Magnetization for $\chi = {chi}$ ;" + " $h_{ev} =$" + f"{h_ev}",
    fontsize=14,
)
for i, trotter_step in enumerate(trotter_steps):
    delta = t / trotter_step
    mag_mps_tot = np.loadtxt(
        f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/mag_data/mag_mps_tot_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    mag_mps_tot = mag_mps_tot[:: mult_factor[i]]
    error_mag_tot = np.abs(np.asarray(mag_tot_jesus) - np.asarray(mag_mps_tot))
    plt.plot(
        mult_factor[i] * delta * np.arange(100 + 1),
        error_mag_tot,
        color=colors[i],
        label=f"$\delta={delta}$",
    )

ticks = mult_factor[i] * delta * np.arange(100 + 1)
labels = delta * np.arange(trotter_step + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
steps = len(labels) // 5
labels = labels[::steps]
plt.xticks(ticks=ticks, labels=labels)
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\left|M^{exact} - M^{mps}\\right|$")
plt.yscale("log")
plt.legend()
plt.savefig(
    f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/magnetization/error_total_mag_L_{L}_flip_{flip}_chi_{chi}_h_ev_{h_ev}.png"
)
plt.show()
# -----------------------
# local Z magnetization
# -----------------------
plt.title(
    f"Error Middle Chain Z Magnetization for $\chi = {chi}$ ;"
    + " $h_{ev} =$"
    + f"{h_ev}",
    fontsize=14,
)
for i, trotter_step in enumerate(trotter_steps):
    delta = t / trotter_step
    mag_mps_loc = np.loadtxt(
        f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/mag_data/mag_mps_loc_Z_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    mag_mps_loc = mag_mps_loc[:: mult_factor[i]]
    error_mag_tot = np.abs(np.asarray(mag_loc_Z_jesus) - np.asarray(mag_mps_loc))
    plt.plot(
        mult_factor[i] * delta * np.arange(100 + 1),
        error_mag_tot,
        color=colors[i],
        label=f"$\delta={delta}$",
    )
ticks = mult_factor[i] * delta * np.arange(100 + 1)
labels = delta * np.arange(trotter_step + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
steps = len(labels) // 5
labels = labels[::steps]
plt.xticks(ticks=ticks, labels=labels)
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\left|{\sigma_{L/2}^z}^{ex} - {\sigma_{L/2}^z}^{mps}\\right|$")
plt.yscale("log")
plt.legend()
plt.savefig(
    f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/magnetization/error_local_mag_Z_{L}_flip_{flip}_chi_{chi}_h_ev_{h_ev}.png"
)
plt.show()
# -----------------------
# local X magnetization
# -----------------------
plt.title(
    f"Error Middle Chain X Magnetization for $\chi = {chi}$ ;"
    + " $h_{ev} =$"
    + f"{h_ev}",
    fontsize=14,
)
for i, trotter_step in enumerate(trotter_steps):
    delta = t / trotter_step
    mag_mps_loc = np.loadtxt(
        f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/mag_data/mag_mps_loc_X_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    mag_mps_loc = mag_mps_loc[:: mult_factor[i]]
    error_mag_tot = np.abs(np.asarray(mag_loc_X_jesus) - np.asarray(mag_mps_loc))
    plt.plot(
        mult_factor[i] * delta * np.arange(100 + 1),
        error_mag_tot,
        color=colors[i],
        label=f"$\delta={delta}$",
    )
ticks = mult_factor[i] * delta * np.arange(100 + 1)
labels = delta * np.arange(trotter_step + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
steps = len(labels) // 5
labels = labels[::steps]
plt.xticks(ticks=ticks, labels=labels)
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\left|{\sigma_{L/2}^x}^{ex} - {\sigma_{L/2}^x}^{mps}\\right|$")
plt.yscale("log")
plt.legend()
plt.savefig(
    f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/magnetization/error_local_mag_X_{L}_flip_{flip}_chi_{chi}_h_ev_{h_ev}.png"
)
plt.show()
# -----------------------
# entropy
# -----------------------
plt.title(
    "Error Middle Chain Entanglement Entropy: "
    + f"L={L}, $\chi = {chi}$; $h_{{t-ev}} = {h_ev}$",
    fontsize=14,
)
for i, trotter_step in enumerate(trotter_steps):
    delta = t / trotter_step
    entropy_chi = [0]
    schmidt_vals = np.loadtxt(
        f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/bonds_data/middle_chain_schmidt_values_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    for s in schmidt_vals:
        entropy = von_neumann_entropy(s)
        entropy_chi.append(entropy)
    entropy_chi_red = entropy_chi[:: mult_factor[i]]
    error_mc = np.abs(np.asarray(entropy_jesus) - np.asarray(entropy_chi_red))
    plt.plot(
        mult_factor[i] * delta * np.arange(100 + 1),
        error_mc,
        color=colors[i],
        label=f" $\delta={delta}$",
    )
ticks = mult_factor[i] * delta * np.arange(100 + 1)
labels = delta * np.arange(trotter_step + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
steps = len(labels) // 5
labels = labels[::steps]
plt.xticks(ticks=ticks, labels=labels)
plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$|S_{exact} - S_{mps}|$")
plt.yscale("log")
plt.legend()
plt.savefig(
    f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/entropy/error_entropy_{where}_{L}_flip_{flip}_chi_{chi}_h_ev_{h_ev}.png"
)
plt.show()
# %%
plt.title(
    "Exact Entanglement Entropy evolution: "
    + f"L={L}, $\chi = {chi}$; $h_{{t-ev}} = {h_ev}$",
    fontsize=14,
)
entropy_tot = np.loadtxt(fname=file_path_entr)
plt.imshow(entropy_tot, cmap='viridis', aspect='auto')
plt.colorbar()
ticks = range(0,L-3)
labels = range(2,L-1)
plt.xticks(ticks=ticks, labels=labels)
plt.xlabel("bonds")
ticks = np.arange(100 + 1)
labels = delta * np.arange(trotter_step + 1)
steps = len(ticks) // 5
ticks = ticks[::steps]
steps = len(labels) // 5
labels = labels[::steps]
plt.yticks(ticks=ticks, labels=labels)
plt.ylabel("time (t = $\delta$ T)")
plt.savefig(
    f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/entropy/exact_entanglement_entropy_{where}_{L}_flip_{flip}_chi_{chi}_h_ev_{h_ev}.png"
)
plt.show()