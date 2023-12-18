# import packages
import argparse
from qs_mps.applications.Z2.exact_hamiltonian import H_Z2_gauss
from qs_mps.utils import *

parser = argparse.ArgumentParser(prog="observables_Z2_exact")
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
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
)
parser.add_argument(
    "-U", "--gauss", help="Gauss constraint parameter", default=1e+3, type=float
)

args = parser.parse_args()

# define the interval of equally spaced values of external field
interval = np.linspace(args.h_i, args.h_f, args.npoints)

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/1_Z2"
    path_eigvec = "D:/code/projects/1_Z2"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2"
    path_eigvec = "/Users/fradm98/Desktop/projects/1_Z2"
elif args.path == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
    path_eigvec = "/Users/fradm/Desktop/projects/1_Z2"
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

W = []
for h in interval:
    Z2_exact = H_Z2_gauss(L=args.L, l=args.l, model=args.model, lamb=h, U=1e+3)
    psi = np.load(f"{path_eigvec}/results/eigenvectors/ground_state_direct_lattice_{args.l-1}x{args.L-1}_{Z2_exact.sector}_U_{args.gauss}_h_{h:.{precision}f}.npy")

    if args.o == "wl":
        loop = Z2_exact.latt.plaquettes(from_zero=True)
        plaq = Z2_exact.plaquette_term(loop[sites])
        # plaq = Z2_exact.plaquette_term(loop[s+1])
        exp_val_wilson_loop = np.real(psi.T @ plaq @ psi)
        W.append(exp_val_wilson_loop)



np.savetxt(
            f"{parent_path}/results/exact/wilson_loops/wilson_loop_{args.model}_direct_lattice_{args.l-1}x{args.L-1}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}",
            W,
        )