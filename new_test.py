# %%
from mps_class_v9 import MPS
from utils import *
import numpy as np
import matplotlib.pyplot as plt
# %%
L = 9
d = 2
chi = 4
h_0 = 0
J = 1
chain = MPS(L=L, d=d, model="Ising", chi=chi, h=h_0, J=J, eps=0)
# %%
chain._random_state(seed=7)
chain.canonical_form()
chain.sweeping(trunc=True)
# %%
Z = np.array([[1,0],[0,-1]])
chain.mps_local_exp_val(op=Z)
# %%
chain.flipping_mps()
mag_mps_loc = []
mag_mps_tot = []
mag_mps_loc.append(np.real(chain.mps_local_exp_val(op=Z)))
chain.order_param_Ising(op=Z)
mag_mps_tot.append(chain.mpo_first_moment().real)
# ============================
# ============================
# ============================
# ============================
# ============================

# %%
def trunc_mps(mps, chi):
    chi = int(np.log2(chi))
    assert (
        mps.L >= 2 * chi
    ), "The spin chain is too small for the selected bond dimension chi"
    j = 0
    sites = mps.sites
    for i in range(chi):
        mps.sites[j] = sites[j][: (mps.d ** i), : , : (mps.d ** (i + 1))]
        j += 1
    for _ in range(mps.L - (2 * chi)):
        mps.sites[j] = sites[j][: (mps.d ** chi), : , : (mps.d ** chi)]
        j += 1
    for i in range(chi):
        mps.sites[j] = sites[j][: (mps.d ** (chi - i)), : , : (mps.d ** (chi - i - 1))]
        j += 1
    mps.canonical_form(trunc=False, svd_direction="left")
    return mps
# %%
# attempt of loop
trotter_steps = 5
delta = 0.6
h_ev = 0.3
trunc = True
chi_max = 19
for T in range(trotter_steps):
    print(f"Trotter step: {T}")
    chain.mpo_Ising_time_ev(delta=delta, h_ev=h_ev, J_ev=J)
    new_chain = chain.mpo_to_mps()
    if trunc:
        new_chain = trunc_mps(chain, chi=chi_max)
    mag_mps_loc.append(np.real(new_chain.mps_local_exp_val(op=Z)))
    new_chain.order_param_Ising(op=Z)
    mag_mps_tot.append(new_chain.mpo_first_moment().real)

# %%
# exact
psi_0 = exact_initial_state(L=L, h_t=0)
mag_tot = H_loc(L=L, op=Z)
magnetization = [single_site_op(op=Z, site=i, L=L) for i in range(1,L+1)]
mag_exact_tot = []
mag_exact_loc = []
local_mag = []
mag_exact_tot.append(psi_0.T.conjugate() @ mag_tot @ psi_0)
for i in range(L):
    local_mag.append((psi_0.T.conjugate() @ magnetization[i] @ psi_0).real)
mag_exact_loc.append(local_mag)
for T in range(trotter_steps):
    print(f"Trotter step: {T}")
    U_ev = exact_evolution_operator(L=L, h_t=h_ev, delta=delta, trotter_step=T)
    psi = U_ev @ psi_0
    mag = psi.T.conjugate() @ mag_tot @ psi
    mag_exact_tot.append(mag)
    local_mag = []
    for i in range(L):
        local_mag.append((psi.T.conjugate() @ magnetization[i] @ psi).real)
    mag_exact_loc.append(local_mag)
# %%
plt.plot(mag_exact_tot, label="exact")
plt.plot(mag_mps_tot, 'o', label="mps")
plt.legend()
plt.show()

fig, axs = plt.subplots(1,2)
axs[0].imshow(mag_exact_loc, cmap='seismic', vmax=1, vmin=-1)
axs[0].set_title('Exact')
axs[1].imshow(mag_mps_loc, cmap='seismic', vmax=1, vmin=-1)
axs[1].set_title('MPS')

# %%
