# import packages
import argparse
from qs_mps.mps_class import MPS
from qs_mps.utils import *

parser = argparse.ArgumentParser(prog="gs_search_Z2")
parser.add_argument("l", help="Number of ladders in the direct lattice", type=int)
parser.add_argument("L", help="Number of rungs per ladder", type=int)
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
parser.add_argument("o", help="Observable we want to compute. Available are 'wl'", type=str)
parser.add_argument("sites", help="Number of sites in the wilson loop", type=int)
parser.add_argument("ladders", help="Number of ladders in the wilson loop", type=int)

parser.add_argument("chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
)

args = parser.parse_args()

# define the physical dimension
d = int(2**(args.l))

# define the interval of equally spaced values of external field
interval = np.linspace(args.h_i, args.h_f, args.npoints)

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/1_Z2"
    path_tensor = "D:/code/projects/1_Z2"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/Users/fradm98/Desktop/projects/1_Z2"
elif args.path == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/Users/fradm/Desktop/projects/1_Z2"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")

num = (args.h_f - args.h_i) / args.npoints
precision = get_precision(num)


if args.sites == 1:
    sites = 0
if args.ladders == 1:
    ladders = 1
# ---------------------------------------------------------
# Wilson Loop
# ---------------------------------------------------------
for chi in args.chis:
    W = []
    for h in interval:
        lattice_mps = MPS(L=args.L, d=d, model=args.model, chi=chi, h=h)
        lattice_mps.L = lattice_mps.L - 1

        lattice_mps.load_sites(path=path_tensor, precision=precision)
        if args.o == "wl":
            lattice_mps.Z2.wilson_Z2_dual(mpo_sites=[sites], ls=[ladders]) #list(range(s))
        lattice_mps.w = lattice_mps.Z2.mpo
        W.append(lattice_mps.mpo_first_moment().real)
    

    np.savetxt(
                f"{parent_path}/results/wilson_loops/wilson_loop_{args.model}_direct_lattice_{args.l}x{args.L-1}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                W,
            )