import numpy as np
from scipy.sparse import csc_array
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt
import matplotlib as mpl

# default parameters of the plot layout
plt.rcParams["text.usetex"] = True  # use latex
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

from qs_mps.applications.Z2.exact_hamiltonian import *
from qs_mps.sparse_hamiltonians_and_operators import *
from qs_mps.mps_class import MPS
from qs_mps.utils import anim, get_cx, get_cy

L = 11
d = 2
model = "Ising"
J = 1
h_l = 1e-5
h_i = 0.1
chi = 50

ising_chain = MPS(L=L, d=d, model=model, chi=chi, J=J, h=h_i, eps=h_l)
ising_chain._random_state(seed=3, type_shape="rectangular", chi=chi)
ising_chain.canonical_form()
ising_chain.DMRG(trunc_chi=True, trunc_tol=False, where=L//2, long="Z", trans="X")
ising_chain.check_canonical(site=1)

Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])
ising_chain.local_param(site=5, op='Z')
print(ising_chain.mpo_first_moment())
# flipping middle spin
# ising_chain.flipping_mps(op="X")
for i in range(L):
    ising_chain.local_param(site=i+1, op='Z')
    print(f"site: {i+1}, {ising_chain.mpo_first_moment()}")

npoints = 200
delta = 0.01
h_ev = 0.5
J_ev = 1
n_sweeps = 8
bond = True
exact = False
obs = ["lm", "losch"]
obs_freq = 1
training = True
chi_max = 200
path_tensor = "D:/projects/0_ISING/"
parent_path = "D:/projects/0_ISING/"
# create a run group for saving observables
h5file = f"{parent_path}/results/results_time_2.hdf5"
params = dict(L=L, delta=delta, 
            T=npoints, of=obs_freq, h_i=h_i, h_ev=h_ev, J_ev=J_ev, chi=chi, chi_max=chi_max)

run_group = create_run_group(h5file, params)

import datetime as dt

date_start = dt.datetime.now()
print(f"\n*** Starting TEBD evolution in {dt.datetime.now()} ***\n")

# trotter evolution
(errors,
 entropies,
 svs,
 local_magnetization,
 overlaps,
 braket_ex_sp,
 braket_ex_mps,
 braket_mps_sp,
 chi_sat) = ising_chain.TEBD_variational_ising(trotter_steps=npoints,
                                                     delta=delta,
                                                     J_ev=J_ev,
                                                     h_ev=h_ev,
                                                     n_sweeps=n_sweeps,
                                                     conv_tol=1e-10,
                                                     bond=bond,
                                                     where=L//2,
                                                     exact=exact,
                                                     obs=obs,
                                                     obs_freq=obs_freq,
                                                     training=training,
                                                     chi_max=chi_max,
                                                     path=path_tensor,
                                                     run_group=run_group,
                                                     save_file=h5file
    )

t_final = dt.datetime.now() - date_start
print(f"Total time for TEBD evolution of {npoints} trotter steps is: {t_final}")

np.save(
        f"{parent_path}/results/errors/errors_time_ev_training_{training}_{model}_L_{L}_h_{h_i}-{h_ev}_delta_{delta}_trotter_steps_{npoints}_chi_{chi_max}.npy",
        errors,
        )
np.save(
        f"{parent_path}/results/entropies/entropies_time_ev_bond_{bond}_{model}_L_{L}_h_{h_i}-{h_ev}_delta_{delta}_trotter_steps_{npoints}_chi_{chi_max}.npy",
        entropies,
        )
np.save(
        f"{parent_path}/results/magnetization/local_magnetization_time_ev_{model}_L_{L}_h_{h_i}-{h_ev}_delta_{delta}_trotter_steps_{npoints}_chi_{chi_max}.npy",
        local_magnetization,
        )
np.save(
        f"{parent_path}/results/overlaps/overlaps_time_ev_{model}_L_{L}_h_{h_i}-{h_ev}_delta_{delta}_trotter_steps_{npoints}_chi_{chi_max}.npy",
        overlaps,
        )
np.save(
        f"{parent_path}/results/entropies/chi_saturation_time_ev_{model}_L_{L}_h_{h_i}-{h_ev}_delta_{delta}_trotter_steps_{npoints}_chi_{chi_max}.npy",
        chi_sat,
        )