import argparse
import numpy as np
from qs_mps.mps_class import MPS
from qs_mps.utils import get_precision, save_list_of_lists, access_txt
from qs_mps.gs_multiprocessing import ground_state_ising

# DENSITY MATRIX RENORMALIZATION GROUP to find ground states changing the transverse field
# parameters

parser = argparse.ArgumentParser(prog="gs_search")
parser.add_argument("L", help="Spin chain length", type=int)
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "h_i", help="Starting value of h (external transverse field)", type=float
)
parser.add_argument(
    "h_f", help="Final value of h (external transverse field)", type=float
)
parser.add_argument(
    "path", help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'", type=str
)
parser.add_argument("chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Ising", type=str
)
parser.add_argument(
    "-s",
    "--number_sweeps",
    help="Number of sweeps during the compression algorithm for each trotter step",
    default=8,
    type=int,
)
parser.add_argument(
    "-cv",
    "--conv_tol",
    help="Convergence tolerance of the compression algorithm",
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
    "-d",
    "--dimension",
    help="Physical dimension. By default 2",
    default=2,
    type=int,
)

args = parser.parse_args()

# define the interval of equally spaced values of external field
interval = np.linspace(args.h_i, args.h_f, args.npoints)

# take the path and precision to save files
if args.path == 'pc':
    path_drive = "G:/My Drive/projects/0_ISING"
elif args.path == 'mac':
    path_drive = "/Users/fradm98/Google Drive/My Drive/projects/0_ISING"
elif args.path == 'marcos':
    path_drive = "/Users/fradm/Google Drive/My Drive/projects/0_ISING"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")

path = f"{path_drive}/results/tensors"
num = (args.h_f-args.h_i)/args.npoints
precision = get_precision(num)

if args.where == -1:
    args.where = (args.L // 2)
elif args.where == -2:
    args.bond = False
# ---------------------------------------------------------
# DMRG
# ---------------------------------------------------------
for chi in args.chis:  # L // 2 + 1
    args_mps = {
                'L':args.L,
                'd':args.dimension,
                'model':args.model,
                'chi':chi, 
                'path':path_drive, 
                'precision':precision, 
                'trunc_chi':True, 
                'trunc_tol':False, 
                'where': args.where, 
                'bond': args.bond,
                'J': 1,
                'eps': 0,
                }
    if __name__ == '__main__':
        energy_chi, entropy_chi = ground_state_ising(args_mps=args_mps, multpr=True, param=interval)
        


        if args.bond == False:
            args.where = "all"
        
        save_list_of_lists(
            f"{path_drive}/results/energy_data/energies_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
            energy_chi,
        )
        save_list_of_lists(
            f"{path_drive}/results/entropy/{args.where}_bond_entropy_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
            entropy_chi,
        )
        if args.where == 'all':
            entropy_mid = access_txt(
                f"{path_drive}/results/entropy/{args.where}_bond_entropy_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                args.L // 2,
            )
            np.savetxt(
                f"{path_drive}/results/entropy/{args.L // 2}_bond_entropy_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                entropy_mid,
            )