#%%
# import packages
from mps_class import MPS
from utils import *
import matplotlib.pyplot as plt
from ncon import ncon
import scipy
from scipy.sparse import csr_array
import time

# variational compression changing bond dimension
trotter_steps = 500
h_ev = 0.3
L = 15
model = "Ising"
flip = True

# fixed parameteres
t = 10
delta = t / trotter_steps
h_t = 0
n_sweeps = 8


# %%
# visualization

# ---------------------------------------------------------
# total
# ---------------------------------------------------------
chis = [2,4,16]
plt.title(f"Total Magnetization for $\delta = {delta}$ ;" + " $h_{ev} =$" + f"{h_ev}")
colors = create_sequential_colors(
    num_colors=len(chis), colormap_name="viridis"
)
for i,chi in enumerate(chis):
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
#     mag_exact_tot,
#     color="indianred",
#     label=f"exact: $L={L}$",
# )
plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()

plt.title(f"Magnetization Z in the middle site for $\delta = {delta}$ ;" + " $h_{ev} =$" + f"{h_ev}")
for i,chi in enumerate(chis):
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
#     mag_exact_tot,
#     color="indianred",
#     label=f"exact: $L={L}$",
# )
plt.xlabel("time (t = $\delta$ T)")
plt.legend()
plt.show()

plt.title(f"Magnetization X in the middle site for $\delta = {delta}$ ;" + " $h_{ev} =$" + f"{h_ev}")
for i,chi in enumerate(chis):
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
#     mag_exact_tot,
#     color="indianred",
#     label=f"exact: $L={L}$",
# )
plt.xlabel("time (t = $\delta$ T)")
plt.legend()
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

chis = [2,4,16]
colors = create_sequential_colors(
    num_colors=len(chis), colormap_name="viridis"
)
plt.title(f"Truncation error $vs$ trotter steps")
for i, chi in enumerate(chis):  # L//2+1
    errors = load_list_of_lists(
        f"G:/My Drive/projects/0_ISING/results/\errors_data/errors_{model}_L_{L}_flip_{flip}_delta_{delta}_chi_{chi}"
    )
    last_errors = [float(sublist[-1]) for sublist in errors]
    last_error, num_zeros = replace_zeros_with_nan(last_errors)
    plt.plot(np.arange(trotter_steps + 1),
        last_error,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=5,
        alpha=0.6,
        color=colors[i],
        # facecolors=colors[i],
        # edgecolors=colors[i],
        label=f"$\left|\left| |\phi\\rangle - |\psi\\rangle \\right|\\right|^2$ $\chi={chi}$")

plt.ylabel("$\mathcal{A* N A}$ - 2$\Re(\mathcal{A*M})$ + $\langle \psi| \psi\\rangle$")
plt.xlabel("Trotter Steps (T)")
plt.yscale('log')
# plt.ylim(1e-17,1e-14)
plt.legend()
# plt.savefig(f"C:/Users/HP/Desktop/mps/results/figures/errors_L_{L}_delta_{delta}_h_ev_{h_ev}_different_chis.png")
plt.show()

# %%
# ---------------------------------------------------------
# entropy
# -------------------------------------------------------
plt.title(
    "Middle Chain Entanglement Entropy: " + f"L={L}, $\delta = {delta}$; $h_{{t-ev}} = {h_ev}$"
)
chis = [2,4,16]
colors = create_sequential_colors(
    num_colors=len(chis), colormap_name="viridis"
)
for i, chi in enumerate(chis):  # L//2+1
    entropy_chi = [0]
    for trott in range(trotter_steps):
        schmidt_vals = np.loadtxt(
            f"G:/My Drive/projects/0_ISING/results/bonds_data/schmidt_values_middle_chain_Ising_flip_{flip}_L_{L}_chi_{chi}_trotter_step_{trott}_delta_{delta}"
        )
        entropy = von_neumann_entropy(schmidt_vals)
        entropy_chi.append(entropy)
    
    # mag_mps_loc = np.loadtxt(
    #     f"C:/Users/HP/Desktop/mps/results/\mag_data/mag_mps_loc_L_{L}_delta_{delta}_chi_{chi}"
    # )
    # data1 = mag_mps_loc
    # cmap = "seismic"
    # plt.imshow(data1.T, cmap=cmap, vmin=-1, vmax=1, aspect="auto", alpha=0.4)

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
    # plt.vlines(x=idxs[i]*delta, ymin=min(entropy_chi), ymax=max(entropy_chi), colors='coral', linestyles='dashdot')
    # ticks = delta*np.arange(trotter_steps + 1) 
    # labels = delta * ticks
    # steps = len(ticks) // 5
    # ticks = ticks[::steps]
    # labels = labels[::steps]
    # plt.xticks(ticks=ticks, labels=labels.astype(int))
    
# plt.plot(
#     delta * np.arange(trotter_steps + 1),
#     np.log2(true_bond_dim),
#     '--',
    # label="upper bound $log(\chi_{true})$",
    # )
plt.ylabel("entanglement von neumann entropy $(S_{\chi})$")
plt.xlabel("time (t = $\delta$ T)")
plt.legend(fontsize=10)
plt.savefig(f"G:/My Drive/projects/0_ISING/figures/ent_entropy_L_{L}_delta_{delta}_h_ev_{h_ev}_chi_{chi}_with_mag_loc.png")
plt.show()

# %%
# bond dimensions
true_bond_dim = np.loadtxt(
            f"G:/My Drive/projects/0_ISING/results/bonds_data/bond_dimension_tol_truncated_Ising_L_{L}_delta_{delta}_trotter_{trotter_steps}_h_ev_{h_ev}"
        )
true_bond_dim.astype(int)
element = 1
true_bond_dim = np.concatenate(([element], true_bond_dim))
chis = [2,4,8,16,32,64,128]
colors = create_sequential_colors(
    num_colors=len(chis), colormap_name="viridis"
)
idxs = []
plt.title(f"Tolerance-truncated bond dimension")
for i, chi in enumerate(chis):  # L//2+1
    plt.hlines(y=chi, xmin=0, xmax=delta*trotter_steps, colors=colors[i], label=f"$\chi={chi}$")
    abs_diff = np.abs(true_bond_dim - chi)

    # Find the index of the minimum absolute difference
    idx = np.argmin(abs_diff)
    idxs.append(idx)
    print(idx)
    print(true_bond_dim[idx])
    plt.vlines(x=idx*delta, ymin=min(true_bond_dim), ymax=max(true_bond_dim), colors='coral', linestyles='dashdot', linewidth=0.7)
plt.plot(delta * np.arange(trotter_steps + 1),
        true_bond_dim, color='coral', label="true $\chi$")
plt.yscale('log')
plt.legend()
plt.ylabel("Bond Dimension $\chi$")
plt.xlabel("time (t = $\delta$ T)")
plt.savefig(f"G:/My Drive/projects/0_ISING/figures/bond_dimension_L_{L}_delta_{delta}_h_ev_{h_ev}_different_chis.png")
plt.show()
# %%
