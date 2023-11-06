# %%
from utils import *
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

# fixed parameteres
L = 15
trotter_steps = 100
t = 10
delta = t / trotter_steps
h_t = 0
h_ev = 0.3
n_sweeps = 8
flip = True
model = "Ising"

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

psi, mag_loc_Z_jesus, mag_loc_X_jesus, mag_tot_jesus, entropy_jesus = exact_evolution_sparse(L=L, h_t=h_t, h_ev=h_ev, delta=delta, trotter_steps=trotter_steps, h_l=0, flip=True)
# psi, mag_loc_sp, mag_tot_sp = exact_evolution(L=L, h_t=h_t, h_ev=h_ev, delta=delta, trotter_steps=trotter_steps, h_l=0, flip=True)

np.savetxt(
        f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/mag_data/mag_exact_loc_Z_{model}_L_{L}_flip_{flip}_delta_{delta}_Jesus",
        mag_loc_Z_jesus,
    )

np.savetxt(
        f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/mag_data/mag_exact_loc_X_{model}_L_{L}_flip_{flip}_delta_{delta}_Jesus",
        mag_loc_X_jesus,
    )

np.savetxt(
        f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/entropy/entropy_{model}_L_{L}_flip_{flip}_delta_{delta}_Jesus",
        entropy_jesus,
    )

# %%

mag_loc_Z_jesus = access_txt(f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/results/mag_data/mag_exact_loc_{model}_L_{L}_flip_{flip}_delta_{delta}_Jesus",
                             L // 2)

plt.title(
    f"Magnetization Z in the middle site for $\delta = {delta}$ ;"
    + " $h_{ev} =$"
    + f"{h_ev}",
    fontsize=14,
)
plt.plot(
    delta * np.arange(100 + 1),
    mag_loc_Z_maricarmen,
    color="indianred",
    label=f"MariCarmen L={L}",
)
plt.plot(
    delta * np.arange(101),
    mag_loc_Z_jesus,
    "x",
    color="deepskyblue",
    label=f"Jesus L={L}",
)

plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\sigma_{L/2}^z$")
plt.legend()
# plt.savefig(f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/magnetization/local_mag_Z_L_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()

plt.title(
    f"Total Magnetization for $\delta = {delta}$ ;"
    + " $h_{ev} =$"
    + f"{h_ev}",
    fontsize=14,
)
plt.plot(
    delta * np.arange(100 + 1),
    mag_tot_maricarmen,
    color="indianred",
    label=f"MariCarmen L={L}",
)
plt.plot(
    delta * np.arange(101),
    mag_tot_jesus,
    "x",
    color="deepskyblue",
    label=f"Jesus L={L}",
)

plt.xlabel("time (t = $\delta$ T)")
plt.ylabel("$\sum_{i=1}^L \sigma_i^z$")
plt.legend()
# plt.savefig(f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/magnetization/local_mag_Z_L_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()
# %%
# -----------------------
# total magnetization
# -----------------------
plt.title(
    f"Error Total Magnetization for $\delta = {delta}$ ;" + " $h_{ev} =$" + f"{h_ev}",
    fontsize=14,
)
mag_tot_sp = [mag_tot_jesus[i][0] for i in range(len(mag_tot_jesus))]
error_mag_tot = np.abs(np.asarray(mag_tot_maricarmen) - np.asarray(mag_tot_sp))
plt.plot(
        delta * np.arange(100 + 1),
        error_mag_tot,
        color="red",
        label=f"error",
    )
ticks = delta * np.arange(100 + 1)
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
# plt.savefig(f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/magnetization/error_total_mag_L_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()

# %%
# -----------------------
# local Z magnetization
# -----------------------
plt.title(
    f"Error Middle Chain Z Magnetization for $\delta = {delta}$ ;"
    + " $h_{ev} =$"
    + f"{h_ev}",
    fontsize=14,
)

error_mag_tot = np.abs(np.asarray(mag_loc_Z_maricarmen) - np.asarray(mag_loc_Z_jesus))
plt.plot(
        delta * np.arange(100 + 1),
        error_mag_tot,
        color="red",
        label=f"error",
    )
ticks = delta * np.arange(100 + 1)
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
# plt.savefig(f"/Users/fradm98/Google Drive/My Drive/projects/0_ISING/figures/magnetization/error_local_mag_Z_{L}_flip_{flip}_delta_{delta}_h_ev_{h_ev}.png")
plt.show()
# %%
