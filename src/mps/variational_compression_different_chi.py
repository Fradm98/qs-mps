# import packages
from mps_class import MPS
from utils import *
import matplotlib.pyplot as plt
from ncon import ncon
import scipy
from scipy.sparse import csr_array
import time
import argparse

# variational compression changing bond dimension
# parameters

parser = argparse.ArgumentParser(prog="Time Evolution")
parser.add_argument("L", help="Spin chain length", type=int)
parser.add_argument(
    "trotter_steps",
    help="It will give you the accuracy of the trotter evolution for fixed t",
    type=int,
)
parser.add_argument(
    "h_ev", help="It will give you the magnitude of the quench", type=float
)
parser.add_argument("chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument(
    "-f", "--flip", help="Flip the middle site or not", action="store_true"
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Ising", type=str
)
parser.add_argument(
    "-t", "--time", help="Final time of the evolution", default=10, type=float
)
parser.add_argument(
    "-h_ti",
    "--h_transverse_init",
    help="Initial transverse field before the quench",
    default=0,
    type=float,
)
parser.add_argument(
    "-s",
    "--number_sweeps",
    help="Number of sweeps during the compression algorithm for each trotter step",
    default=8,
    type=int,
)
parser.add_argument(
    "-fid",
    "--fidelity",
    help="Fidelity with exact solution. Doable only for small Ising chains",
    action="store_true",
)
parser.add_argument(
    "-cv",
    "--conv_tol",
    help="Convergence tolerance of the compression algorithm",
    default=1e-15,
    type=float,
)
args = parser.parse_args()
delta = args.time / args.trotter_steps


# # ---------------------------------------------------------
# # exact
# psi_new, mag_exact_loc, mag_exact_tot = exact_evolution(
#     L=L, h_t=h_t, h_ev=h_ev, delta=delta, trotter_steps=trotter_steps
# )

# np.savetxt(
#     f"results/mag_data/mag_exact_tot_L_{L}_delta_{delta}_trott_{trotter_steps}",
#     mag_exact_tot,
# )
# np.savetxt(
#     f"results/mag_data/mag_exact_loc_L_{L}_delta_{delta}_trott_{trotter_steps}",
#     mag_exact_loc,
# )

# ---------------------------------------------------------
# variational truncation mps
for chi in args.chis:  # L // 2 + 1
    chain = MPS(
        L=args.L, d=2, model=args.model, chi=chi, h=args.h_transverse_init, eps=0, J=1
    )
    chain._random_state(seed=3, chi=chi)
    chain.canonical_form(trunc_chi=False, trunc_tol=True)
    # chain.sweeping(trunc_chi=False, trunc_tol=True, n_sweeps=2)
    init_state = np.zeros((1, 2, 1))
    init_state[0, 0, 0] = 1
    for i in range(chain.L):
        chain.sites[i] = init_state
    (
        mag_mps_tot,
        mag_mps_loc_X,
        mag_mps_loc,
        overlap,
        errors,
        schmidt_values,
    ) = chain.variational_mps_evolution(
        trotter_steps=args.trotter_steps,
        delta=delta,
        h_ev=args.h_ev,
        flip=args.flip,
        fidelity=False,
        conv_tol=1e-15,
        n_sweeps=args.number_sweeps,
    )
    np.savetxt(
        f"/data/fdimarca/projects/0_ISING/results/mag_data/mag_mps_tot_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}",
        mag_mps_tot,
    )
    np.savetxt(
        f"/data/fdimarca/projects/0_ISING/results/mag_data/mag_mps_loc_X_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}",
        mag_mps_loc_X,
    )
    np.savetxt(
        f"/data/fdimarca/projects/0_ISING/results/mag_data/mag_mps_loc_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}",
        mag_mps_loc,
    )
    mag_mps_loc_Z = access_txt(
        f"/data/fdimarca/projects/0_ISING/results/mag_data/mag_mps_loc_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}",
        args.L // 2,
    )
    np.savetxt(
        f"/data/fdimarca/projects/0_ISING/results/mag_data/mag_mps_loc_Z_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}",
        mag_mps_loc_Z,
    )
    # np.savetxt(
    #     f"/data/fdimarca/projects/0_ISING/results/fidelity_data/fidelity_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}", overlap
    # )
    save_list_of_lists(
        f"/data/fdimarca/projects/0_ISING/results/errors_data/errors_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}",
        errors,
    )
    np.savetxt(
        f"/data/fdimarca/projects/0_ISING/results/bond_data/middle_chain_schmidt_values_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}",
        schmidt_values,
    )
