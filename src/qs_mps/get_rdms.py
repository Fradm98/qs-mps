import argparse
import numpy as np
from qs_mps.mps_class import MPS
from qs_mps.utils import get_precision

parser = argparse.ArgumentParser(prog="get_rdms")
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "hx_i", help="Starting value of x parameter in the phase space", type=float
)
parser.add_argument(
    "hx_f", help="Final value of x parameter in the phase space", type=float
)
parser.add_argument(
    "hy_i", help="Starting value of y parameter in the phase space", type=float
)
parser.add_argument(
    "hy_f", help="Final value of y parameter in the phase space", type=float
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("-L", "--Ls", help="Number of sites in the chain", nargs="+", type=int)
parser.add_argument("-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument(
    "-m", "--model", help="Model to simulate, available are 'Ising', 'Z2', 'ANNNI', 'Cluster', 'XXZ'. By default Cluster", default="Cluster", type=str
)
parser.add_argument(
    "-e", "--eps", help="Weight to the degeneracy breaking term", default=1e-5, type=float
)
parser.add_argument(
    "-w", "--which", help="Number of sites for the reduced density matrices. It will be centered in the middle of the chain", default=1, type=int
)
parser.add_argument(
    "-d", "--deg", help="Degeneracy method used during the DMRG. Depends on the model used", default=1, type=int
)
parser.add_argument(
    "-i",
    "--interval",
    help="Type of interval spacing. Available are 'log', 'lin'",
    default="lin",
    type=str
)

args = parser.parse_args()

# define the physical dimension
d = int(2)

# define the interval of equally spaced values of external field
if args.interval == "lin":
    interval_x = np.linspace(args.hx_i, args.hx_f, args.npoints)
    interval_y = np.linspace(args.hy_i, args.hy_f, args.npoints)
    num = (args.hx_f - args.hx_i) / args.npoints
    precision = get_precision(num)
elif args.interval == "log":
    interval_x = np.logspace(args.hx_i, args.hx_f, args.npoints)
    interval_y = np.logspace(args.hy_i, args.hy_f, args.npoints)
    num = (interval_x[-1]-interval_x[0]) / args.npoints
    precision = get_precision(num)

if args.model == "Ising":
    model_path = "0_ISING"

elif args.model == "Z2":
    model_path = "1_Z2"

elif args.model == "ANNNI":
    model_path = "2_ANNNI"

elif args.model == "Cluster" or args.model == "Cluster-XY":
    model_path = "3_CLUSTER"

elif args.model == "XXZ":
    model_path = "4_XXZ"
else:
    raise SyntaxError("Model not valid. Choose among 'Ising', 'Z2', 'ANNNI', 'Cluster', 'Cluster-XY', 'XXZ'")
# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = f"G:/My Drive/projects/{model_path}"
    path_tensor = f"D:/code/projects/{model_path}"
    path_rdms = "C:/Users/HP/Desktop/fidelity-phase-tran"
elif args.path == "mac":
    parent_path = f"/Users/fradm98/Google Drive/My Drive/projects/{model_path}"
    path_tensor = f"/Users/fradm98/Desktop/projects/{model_path}"
    path_rdms = "/Users/fradm98/Desktop/fidelity-phase-tran"
elif args.path == "marcos":
    parent_path = f"/Users/fradm/Google Drive/My Drive/projects/{model_path}"
    path_tensor = f"/Users/fradm/Desktop/projects/{model_path}"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")

print(f"Calculating rdms...")
print(f"{args.model} model selected")
for L in args.Ls:
    start = L // 2 - args.which // 2
    # Generate the list of subsystems using list comprehension
    sites = [start + i for i in range(args.which)]
    for chi in args.chis:  # L // 2 + 1
        print(f"L:{L}, chi:{chi}")
        rdms_chi = []
        for hy in interval_y:
            rdms_y = []
            for hx in interval_x:
                print(f"for hx:{hx:.{precision}f}, hy:{hy:.{precision}f}")
                if args.model == "Ising" or args.model == "Cluster":
                    chain = MPS(L=L, d=d, chi=chi, model=args.model, eps=args.eps, h=hx, J=hy)
                elif args.model == "ANNNI" or args.model == "XXZ":
                    chain = MPS(L=L, d=d, chi=chi, model=args.model, eps=args.eps, k=hx, h=hy, J=1)
                elif args.model == "Cluster-XY":
                    chain = MPS(L=L, d=d, chi=chi, model=args.model, eps=args.eps, lx=0, ly=hy, h=hx, J=1)
                chain.load_sites(path=path_tensor, precision=precision)
                rdm = chain.reduced_density_matrix(sites)
                rdms_y.append(rdm)
            rdms_chi.append(rdms_y)
        
        
        rdms_chi = np.asarray(rdms_chi).reshape((len(interval_x),len(interval_y),d**args.which,d**args.which))
        rdms_chi = rdms_chi[::-1]
        np.save(f"{parent_path}/results/rdms_data/{args.which}_sites-rdms_dmrg_{args.model}_L_{L}_hx_{args.hx_i}-{args.hx_f}_hy_{args.hy_i}-{args.hy_f}_delta_{args.npoints}_degeneracy_method_{args.deg}_eps_{args.eps}_guess_path_chi_{chi}.npy", rdms_chi)