import argparse
import numpy as np
from qs_mps.mps_class import MPS
from qs_mps.utils import get_precision, save_list_of_lists, access_txt
from qs_mps.gs_multiprocessing import ground_state_ising

# DENSITY MATRIX RENORMALIZATION GROUP to find ground states of the 
# 1D Ising Transverse Field model changing the transverse field parameters

parser = argparse.ArgumentParser(prog="gs_search_Ising")
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
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("-L", "--Ls", help="Spin chain lengths", nargs="*", type=int)
parser.add_argument("-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument("-d","--dimension", help="Physical dimension. By default 2", default=2, type=int)
parser.add_argument(
    "-ty", "--type_shape", help="Type of shape of the bond dimension. Available are: 'trapezoidal', 'pyramidal', 'rectangular'", default="trapezoidal", type=str
)
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

path = f"{parent_path}/results/tensors"
num = (args.h_f - args.h_i) / args.npoints
precision = get_precision(num)

Z = np.array([[1, 0], [0, -1]])

# ---------------------------------------------------------
# DMRG
# ---------------------------------------------------------
for L in args.Ls:
    if args.where == -1:
        args.where = L // 2
    elif args.where == -2:
        args.bond = False
    for chi in args.chis:  # L // 2 + 1
        args_mps = {
            "L": L,
            "d": args.dimension,
            "model": args.model,
            "chi": chi,
            "path": path_tensor,
            "type_shape": args.type_shape,
            "precision": precision,
            "trunc_chi": True,
            "trunc_tol": False,
            "where": args.where,
            "bond": args.bond,
            "J": 1,
            "eps": 0,
        }
        if __name__ == "__main__":
            
            
            energy_chi = []
            entropy_chi = []
            schmidt_vals_chi = []
            for J in interval:
                init_state = np.zeros((1, 2, 1))
                init_state[0, 0, 0] = 1
                init_tensor = [init_state for _ in range(L)]
                for h in interval:
                    precision = args_mps["precision"]
                    chain = MPS(
                        L=args_mps["L"],
                        d=args_mps["d"],
                        model=args_mps["model"],
                        chi=args_mps["chi"],
                        h=h,
                        J=J,
                        eps=args_mps["eps"],
                    )
                    chain.sites = init_tensor
                    chain.enlarge_chi(type_shape="rectangular", prnt=False)
                    # chain._random_state(seed=7, chi=args_mps["chi"], type_shape=args_mps["type_shape"])
                    chain.canonical_form(trunc_chi=False, trunc_tol=True)
                    # total
                    chain.order_param(op=Z)
                    mag = np.real(chain.mpo_first_moment())
                    if mag < 0:
                        chain.flipping_all()

                    energy, entropy, schmidt_vals = chain.DMRG(
                        trunc_tol=args_mps["trunc_tol"],
                        trunc_chi=args_mps["trunc_chi"],
                        where=args_mps["where"],
                        bond=args_mps["bond"],
                    )

                    print(f"energy of h:{h:.{precision}f}, J:{J:.{precision}f} is:\n {energy}")
                    print(f"Schmidt values in the middle of the chain:\n {schmidt_vals}")

                    chain.save_sites(path=args_mps["path"], precision=args_mps["precision"])
                    energy_chi.append(energy)
                    entropy_chi.append(entropy)
                    schmidt_vals_chi.append(schmidt_vals)

            # energy_chi, entropy_chi, schmidt_vals_chi = ground_state_ising(
            #     args_mps=args_mps, multpr=False, param=interval
            # )

            if args.bond == False:
                args.where = "all"

            energy_chi = np.array(energy_chi)
            energy_chi = np.array_split(energy_chi, args.npoints)
            save_list_of_lists(
                f"{parent_path}/results/energy_data/energies_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                energy_chi,
            )
            entropy_chi = np.array(entropy_chi)
            entropy_chi = np.array_split(entropy_chi, args.npoints)
            save_list_of_lists(
                f"{parent_path}/results/entropy/{args.where}_bond_entropy_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                entropy_chi,
            )
            schmidt_vals_chi = np.array(schmidt_vals_chi)
            schmidt_vals_chi = np.array_split(schmidt_vals_chi, args.npoints)
            save_list_of_lists(
                f"{parent_path}/results/entropy/{args.where}_schmidt_values_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                schmidt_vals_chi,
            )
            if args.where == "all":
                entropy_mid = access_txt(
                    f"{parent_path}/results/entropy/{args.where}_bond_entropy_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                    L // 2,
                )
                np.savetxt(
                    f"{parent_path}/results/entropy/{L // 2}_bond_entropy_{args.model}_L_{L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                    entropy_mid,
                )
