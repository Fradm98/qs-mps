# import packages
from qs_mps.mps_class import MPS
from qs_mps.utils import *
import matplotlib.pyplot as plt
from ncon import ncon
import scipy
from scipy.sparse import csr_array
import time
import argparse

# TIME EVOLUTION BLOCK DECIMATION with variational compression changing bond dimension
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
    "-f", "--flip", help="Flip the middle site or not. By defalut True", action="store_false"
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
    help="Number of sweeps during the compression algorithm for each trotter step. By default 8",
    default=8,
    type=int,
)
parser.add_argument(
    "-cv",
    "--conv_tol",
    help="Convergence tolerance of the compression algorithm. By default 1e-10",
    default=1e-10,
    type=float,
)
parser.add_argument(
    "-fid",
    "--fidelity",
    help="Fidelity with exact solution. Doable only for small Ising chains",
    action="store_true",
)
parser.add_argument(
    "-b",
    "--bond",
    help="Save the schmidt values for one bond. If False save for each bond. By default True",
    action="store_false",
)
parser.add_argument(
    "-w",
    "--where",
    help="Bond where we want to observe the Schmidt values, should be between 1 and (L-1)",
    default=-1,
    type=int,
)
parser.add_argument(
    "-p",
    "--path",
    help="Path to the drive. Available values are 'pc', 'mac', 'marcos', 'other'",
    default="mac",
    type=str,
)
args = parser.parse_args()
delta = args.time / args.trotter_steps
if args.where == -1:
    args.where = (args.L // 2)
elif args.where == -2:
    args.bond = False


if args.path == 'pc':
    path = "/Users/fradm/Google Drive/My Drive"
elif args.path == 'mac':
    path = "/Users/fradm/Google Drive/My Drive"
if args.path == 'marcos':
    path = "/Users/fradm/Google Drive/My Drive"
if args.path == 'other':
    path = "replace_with_your_path"
    raise SyntaxError("specify your path in the main script")
else:
    raise SyntaxError("choose among the available paths: 'pc, 'mac', 'marcos', 'other'")


# ---------------------------------------------------------
# variational truncation mps
# ---------------------------------------------------------
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
        entropies,
    ) = chain.TEBD_variational(
        trotter_steps=args.trotter_steps,
        delta=delta,
        h_ev=args.h_ev,
        flip=args.flip,
        n_sweeps=args.number_sweeps,
        conv_tol=args.conv_tol,
        fidelity=args.fidelity,
        bond=args.bond,
        where=args.where,
    )

    if args.bond == False:
        args.where = "all"

    np.savetxt(
        f"{path}/projects/0_ISING/results/mag_data/mag_mps_tot_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}",
        mag_mps_tot,
    )
    np.savetxt(
        f"{path}/projects/0_ISING/results/mag_data/mag_mps_loc_X_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}",
        mag_mps_loc_X,
    )
    np.savetxt(
        f"{path}/projects/0_ISING/results/mag_data/mag_mps_loc_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}",
        mag_mps_loc,
    )
    mag_mps_loc_Z = access_txt(
        f"{path}/projects/0_ISING/results/mag_data/mag_mps_loc_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}",
        args.L // 2,
    )
    np.savetxt(
        f"{path}/projects/0_ISING/results/mag_data/mag_mps_loc_Z_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}",
        mag_mps_loc_Z,
    )
    # np.savetxt(
    #     f"{path}/projects/0_ISING/results/fidelity_data/fidelity_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}", overlap
    # )
    save_list_of_lists(
        f"{path}/projects/0_ISING/results/errors_data/errors_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}",
        errors,
    )
    save_list_of_lists(
        f"{path}/projects/0_ISING/results/entropy/{args.where}_bond_entropy_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}",
        entropies,
    )
    if args.where == 'all':
        entropy_mid = access_txt(
            f"{path}/projects/0_ISING/results/entropy/{args.where}_bond_sentropy_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}",
            args.L // 2,
        )
        np.savetxt(
            f"{path}/projects/0_ISING/results/entropy/{args.L//2}_bond_entropy_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}",
            mag_mps_loc_Z,
        )

# different folder paths:
"""
# external drive:
D:/code/

# Google drive on my PC:
G:/My Drive/

# Google drive on my Mac:
/Users/fradm98/Google Drive/My Drive/

# Google drive on MarcOS:
/Users/fradm/Google Drive/My Drive/
"""
