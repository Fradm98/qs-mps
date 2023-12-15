import numpy as np
import argparse
from qs_mps.sparse_hamiltonians_and_operators import exact_evolution_sparse
from qs_mps.utils import access_txt

parser = argparse.ArgumentParser(prog="Exact Ground State and Time Evolution - Ising")
parser.add_argument("L", help="Spin chain length", type=int)
parser.add_argument(
    "trotter_steps",
    help="It will give you how many steps you need to reach time t",
    type=int,
)
parser.add_argument(
    "h_ev", help="It will give you the magnitude of the quench", type=float
)
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
    "-w",
    "--where",
    help="Where to compute the Schmidt decomposition",
    default=-1,
    type=int,
)
parser.add_argument(
    "-b",
    "--bond",
    help="Perform a bond Schmidt decomposition, By default True",
    action="store_false",
)

args = parser.parse_args()

if args.where == -1:
    args.where = args.L // 2
elif args.where == -2:
    args.bond = False

(
    psi_new,
    mag_exact_loc,
    mag_exact_loc_X,
    mag_exact_tot,
    entropy_tot,
) = exact_evolution_sparse(
    L=args.L,
    h_t=args.h_transverse_init,
    h_ev=args.h_ev,
    time=args.time,
    trotter_steps=args.trotter_steps,
    flip=args.flip,
    where=args.where,
    bond=args.bond,
)

if args.bond == False:
    args.where = "all"

np.savetxt(
    f"D:/code/projects/0_ISING/results/exact/mag_data/mag_exact_tot_{args.model}_L_{args.L}_flip_{args.flip}_h_ev_{args.h_ev}_trotter_steps_{args.trotter_steps}_t_{args.time}",
    mag_exact_tot,
)
np.savetxt(
    f"D:/code/projects/0_ISING/results/exact/mag_data/mag_exact_loc_X_{args.model}_L_{args.L}_flip_{args.flip}_h_ev_{args.h_ev}_trotter_steps_{args.trotter_steps}_t_{args.time}",
    mag_exact_loc_X,
)
np.savetxt(
    f"D:/code/projects/0_ISING/results/exact/mag_data/mag_exact_loc_{args.model}_L_{args.L}_flip_{args.flip}_h_ev_{args.h_ev}_trotter_steps_{args.trotter_steps}_t_{args.time}",
    mag_exact_loc,
)
mag_exact_loc_Z = access_txt(
    f"D:/code/projects/0_ISING/results/exact/mag_data/mag_exact_loc_{args.model}_L_{args.L}_flip_{args.flip}_h_ev_{args.h_ev}_trotter_steps_{args.trotter_steps}_t_{args.time}",
    args.L // 2,
)
np.savetxt(
    f"D:/code/projects/0_ISING/results/exact/mag_data/mag_exact_loc_Z_{args.model}_L_{args.L}_flip_{args.flip}_h_ev_{args.h_ev}_trotter_steps_{args.trotter_steps}_t_{args.time}",
    mag_exact_loc_Z,
)
np.savetxt(
    f"D:/code/projects/0_ISING/results/exact/entropy/exact_entropy_{args.where}_{args.model}_L_{args.L}_flip_{args.flip}_h_ev_{args.h_ev}_trotter_steps_{args.trotter_steps}_t_{args.time}",
    entropy_tot,
)
