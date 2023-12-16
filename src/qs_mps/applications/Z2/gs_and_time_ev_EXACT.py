import numpy as np
import argparse
from qs_mps.sparse_hamiltonians_and_operators import exact_evolution_sparse
from qs_mps.utils import access_txt

# DENSITY MATRIX RENORMALIZATION GROUP to find ground states of the Z2 Pure Gauge Theory 
# changing the transverse field parameters in its dual formulation

parser = argparse.ArgumentParser(prog="gs_search_Z2")
parser.add_argument("L", help="Number of rungs per ladder", type=int)
parser.add_argument("l", help="Number of ladders in the lattice", type=int)
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "h_i", help="Starting value of h (external transverse field on the dual lattice)", type=float
)
parser.add_argument(
    "h_f", help="Final value of h (external transverse field on the dual lattice)", type=float
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
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
