# import packages
import argparse
from qs_mps.mps_class import MPS
from qs_mps.utils import *

parser = argparse.ArgumentParser(prog="observables_Ising_mps")
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "h_i", help="Starting value of h external transverse field", type=float
)
parser.add_argument(
    "h_f", help="Final value of h external transverse field", type=float
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("o", help="Observable we want to compute. Available are 'mag'", type=str)
parser.add_argument("-L", "--Ls", help="Spin chain lengths", nargs="*", type=int)
parser.add_argument("-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument("-d", "--dimension", help="Physical dimension. By default 2", default=2, type=int)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Ising", type=str
)
parser.add_argument(
    "-mo", "--moment", help="Moment degree of the Free energy. E.g. Magnetization -> First Moment, Susceptibility -> Second Moment, etc. Available are 1,2,4", default=1, type=int
)

args = parser.parse_args()

# define the interval of equally spaced values of external field
interval = np.linspace(args.h_i, args.h_f, args.npoints)

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/0_ISING"
    path_tensor = "D:/code/projects/0_ISING"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/0_ISING"
    path_tensor = "/Users/fradm98/Desktop/projects/0_ISING"
elif args.path == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/0_ISING"
    path_tensor = "/Users/fradm/Desktop/projects/0_ISING"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")

num = (args.h_f - args.h_i) / args.npoints
precision = get_precision(num) 


# define moment
if args.moment == 1:
    moment = "first"
if args.moment == 2:
    moment = "second"
if args.moment == 4:
    moment = "fourth"

# ---------------------------------------------------------
# Observables
# ---------------------------------------------------------
for L in args.Ls:
    for chi in args.chis:
        M = []
        for J in interval:
            for h in interval:
                chain_mps = MPS(L=L, d=args.dimension, model=args.model, chi=chi, h=h, J=J)

                chain_mps.load_sites(path=path_tensor, precision=precision)
                
                if args.o == "mag":
                    print(f"Magnetization for L:{L}, D:{chi}, h:{h:.{precision}f}, J:{J:.{precision}f}")
                    chain_mps.order_param()
                    if args.moment == 1:
                        M.append(chain_mps.mpo_first_moment().real)
                    elif args.moment == 2:
                        M.append(chain_mps.mpo_second_moment().real/(chain_mps.L**2))
                    elif args.moment == 4:
                        M.append(chain_mps.mpo_fourth_moment().real/(chain_mps.L**4))
                else:
                    raise ValueError("Select a valid observable. Available are 'mag'")

        if args.o == "mag":
            M = np.array(M)
            M = np.array_split(M, args.npoints)
            np.save(
                    f"{parent_path}/results/mag_data/magnetization_{moment}_moment_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_J_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                    M,
                )