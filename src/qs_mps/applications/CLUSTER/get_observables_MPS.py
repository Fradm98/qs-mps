# import packages
import argparse
from qs_mps.mps_class import MPS
from qs_mps.utils import *

parser = argparse.ArgumentParser(prog="observables_CLUSTER_mps")
parser.add_argument("L", help="Number of spins", type=int)
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
# parser.add_argument(
#     "k_i", help="Starting value of J2 next nearest interaction coefficient", type=float
# )
# parser.add_argument(
#     "k_f", help="Final value of J2 next nearest interaction coefficient", type=float
# )
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("o", help="Observable we want to compute. Available are 'wl', 'el', 'thooft'", type=str)
parser.add_argument("chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument("-d", help="Physical dimension. By default 2", default=2, type=int)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Cluster", type=str
)
parser.add_argument(
    "-mo", "--moment", help="Moment degree of the Free energy. E.g. Magnetization -> First Moment, Susceptibility -> Second Moment, etc. Available are 1,2,4", default=1, type=int
)

args = parser.parse_args()

# define the interval of equally spaced values of external field
interval_h = np.linspace(args.h_i, args.h_f, args.npoints)

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/3_CLUSTER"
    path_tensor = "D:/code/projects/3_CLUSTER"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/3_CLUSTER"
    # path_tensor = "/Users/fradm98/Desktop/projects/3_CLUSTER"
    # /Volumes/Untitled/code/projects/3_CLUSTER/results/tensors/shapes_sites_Cluster_L_20_chi_50_h_0.01_J_0.01
    path_tensor = "/Users/fradm98/Untitled/code/projects/3_CLUSTER"
elif args.path == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/3_CLUSTER"
    path_tensor = "/Users/fradm/Desktop/projects/3_CLUSTER"
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
for chi in args.chis:
    M = []
    LM = []
    for h in interval_h:
        for j in interval_h:
            chain_mps = MPS(L=args.L, d=args.d, model=args.model, chi=chi, h=h, J=j)

            chain_mps.load_sites(path=path_tensor, precision=precision)
            
            if args.o == "mag":
                print(f"Magnetization for h:{h:.{precision}f}, j:{j:.{precision}f}")
                chain_mps.order_param()
                if args.moment == 1:
                    M.append(chain_mps.mpo_first_moment().real/chain_mps.L)
                elif args.moment == 2:
                    M.append(chain_mps.mpo_second_moment().real/(chain_mps.L**2))
                elif args.moment == 4:
                    M.append(chain_mps.mpo_fourth_moment().real/(chain_mps.L**4))
            
            elif args.o == "loc_mag":
                print(f"Local magnetization for h:{h:.{precision}f}, j:{j:.{precision}f}")
                chain_mps.local_param(site=(chain_mps.L // 2))
                LM.append(chain_mps.mpo_first_moment().real)
            else:
                raise ValueError("Select a valid observable. Available are 'mag', 'loc_mag'")

    if args.o == "mag":
        M = np.array(M)
        M = np.array_split(M, args.npoints)
        np.save(
                    f"{parent_path}/results/mag_data/magnetization_{moment}_moment_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_j_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                    M,
                )
    elif args.o == "loc_mag":
        LM = np.array(LM)
        LM = np.array_split(LM, args.npoints)
        np.save(
                    f"{parent_path}/results/mag_data/local_magnetization_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_j_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                    LM,
                )