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
parser.add_argument(
    "o", help="Observable we want to compute. Available are 'mag_x', mag_z", type=str
)
parser.add_argument("-L", "--Ls", help="Spin chain lengths", nargs="*", type=int)
parser.add_argument(
    "-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int
)
parser.add_argument(
    "-d", "--dimension", help="Physical dimension. By default 2", default=2, type=int
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Ising", type=str
)
parser.add_argument(
    "-mo",
    "--moment",
    help="Moment degree of the Free energy. E.g. Magnetization -> First Moment, Susceptibility -> Second Moment, etc. Available are 1,2,4",
    default=1,
    type=int,
)
parser.add_argument(
    "-ss",
    "--sites",
    help="2-site DMRG algorithm used for the optimization. By default False",
    action="store_true",
)

args = parser.parse_args()

# define the interval of equally spaced values of external field
interval_x = np.linspace(args.h_i, args.h_f, args.npoints)
interval_y = np.linspace(1,1,1)

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/0_ISING"
    path_tensor = "D:/code/projects/0_ISING"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/0_ISING"
    path_tensor = "/Users/fradm98/Desktop/projects/0_ISING"
elif args.path == "marcos":
    # parent_path = "/Users/fradm/Google Drive/My Drive/projects/0_ISING"
    path_tensor = "/Users/fradm/Desktop/projects/0_ISING"
    parent_path = "/Users/fradm/Desktop/projects/0_ISING"
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

if args.sites:
    DMRG_sites = 2
else:
    DMRG_sites = 1
# ---------------------------------------------------------
# Observables
# ---------------------------------------------------------
for L in args.Ls:
    for chi in args.chis:
        M = []
        for J in interval_y:
            for h in interval_x:
                chain_mps = MPS(
                    L=L, d=args.dimension, model=args.model, chi=chi, h=h, J=J, eps=1e-5
                )

                chain_mps.load_sites(path=path_tensor, precision=precision, DMRG2=args.sites)

                if "mag" in args.o:
                    print(
                        f"Magnetization for L:{L}, D:{chi}, h:{h:.{precision}f}, J:{J:.{precision}f}"
                    )
                    if args.o == "mag_x":
                        op = "X"
                    elif args.o == "mag_z":
                        op = "Z"
                    chain_mps.order_param(op=op)
                    if args.moment == 1:
                        M.append(chain_mps.mpo_first_moment().real)
                    elif args.moment == 2:
                        M.append(
                            chain_mps.mpo_second_moment().real / (chain_mps.L**2)
                        )
                    elif args.moment == 4:
                        M.append(
                            chain_mps.mpo_fourth_moment().real / (chain_mps.L**4)
                        )
                else:
                    raise ValueError("Select a valid observable. Available are 'mag'")

        if "mag" in args.o:
            M = np.array(M)
            M = np.array_split(M, args.npoints)
            np.save(
                f"{parent_path}/results/mag_data/magnetization_{op}_{moment}_moment_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_J_{1}_DMRG-{DMRG_sites}_delta_{args.npoints}_chi_{chi}.npy",
                M,
            )
