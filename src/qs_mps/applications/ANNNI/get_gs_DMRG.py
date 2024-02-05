import argparse
import numpy as np
from qs_mps.mps_class import MPS
from qs_mps.utils import get_precision, save_list_of_lists, access_txt
from qs_mps.applications.ANNNI.ground_state_multiprocessing import ground_state_ANNNI

# DENSITY MATRIX RENORMALIZATION GROUP to find ground states of the Z2 Pure Gauge Theory 
# changing the transverse field parameters in its dual formulation

parser = argparse.ArgumentParser(prog="gs_search_ANNNI")
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
parser.add_argument(
    "k_i", help="Starting value of J2 next nearest interaction coefficient", type=float
)
parser.add_argument(
    "k_f", help="Final value of J2 next nearest interaction coefficient", type=float
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument("-d", help="Physical dimension. By default 2", default=2, type=int)
parser.add_argument(
    "-ty", "--type_shape", help="Type of shape of the bond dimension. Available are: 'trapezoidal', 'pyramidal', 'rectangular'", default="trapezoidal", type=str
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="ANNNI", type=str
)
parser.add_argument(
    "-mu", "--multpr", help="If True computes ground states with multiprocessing. By default True", action="store_false"
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
    "-v",
    "--save",
    help="Save the tensors. By default True",
    action="store_false",
)
parser.add_argument(
    "-tr",
    "--training",
    help="Save all the energies also the ones during the variational optimization. By default True",
    action="store_false",
)

args = parser.parse_args()

# define the interval of equally spaced values of external field
interval_h = np.linspace(args.h_i, args.h_f, args.npoints)
interval_k = np.linspace(args.k_i, args.k_f, args.npoints)

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/2_ANNNI"
    path_tensor = "D:/code/projects/2_ANNNI"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/2_ANNNI"
    path_tensor = "/Users/fradm98/Desktop/projects/2_ANNNI"
elif args.path == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/2_ANNNI"
    path_tensor = "/Users/fradm/Desktop/projects/2_ANNNI"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")

num = (args.h_f - args.h_i) / args.npoints
precision = get_precision(num)

# where to look at for the entropy
if args.where == -1:
    args.where = args.L // 2
elif args.where == -2:
    args.bond = False

# ---------------------------------------------------------
# DMRG
# ---------------------------------------------------------
for chi in args.chis:  # L // 2 + 1
    args_mps = {
        "L": args.L,
        "d": args.d,
        "chi": chi,
        "type_shape": args.type_shape,
        "model": args.model,
        "trunc_tol": False,
        "trunc_chi": True,
        "where": args.where,
        "bond": args.bond,
        "path": path_tensor,
        "save": args.save,
        "precision": precision,
    }
    if __name__ == "__main__":
        energy_chi, entropy_chi = ground_state_ANNNI(
            args_mps=args_mps, multpr=args.multpr, param=[interval_h,interval_k]
        )

        if args.bond == False:
            args.where = "all"

        if args.training:
            save_list_of_lists(
                f"{parent_path}/results/energy_data/energies_{args.model}_spin_{args.L}_h_{args.h_i}-{args.h_f}_k_{args.k_i}-{args.k_f}_delta_{args.npoints}_chi_{chi}",
                energy_chi,
            )
            energy_gs = access_txt(
                    f"{parent_path}/results/energy_data/energies_{args.model}_spin_{args.L}_h_{args.h_i}-{args.h_f}_k_{args.k_i}-{args.k_f}_delta_{args.npoints}_chi_{chi}",
                    -1,
                )
            np.savetxt(
                f"{parent_path}/results/energy_data/energies_{args.model}_spin_{args.L}_h_{args.h_i}-{args.h_f}_k_{args.k_i}-{args.k_f}_delta_{args.npoints}_chi_{chi}",
                energy_gs,
            )
        else:
            np.savetxt(
                f"{parent_path}/results/energy_data/energies_{args.model}_spin_{args.L}_h_{args.h_i}-{args.h_f}_k_{args.k_i}-{args.k_f}_delta_{args.npoints}_chi_{chi}",
                energy_chi,
            )
             
        # save_list_of_lists(
        #     f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_spin_{args.L}_h_{args.h_i}-{args.h_f}_k_{args.k_i}-{args.k_f}_delta_{args.npoints}_chi_{chi}",
        #     entropy_chi,
        # )
        if args.where == "all":
            entropy_mid = access_txt(
                f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_spin_{args.L}_h_{args.h_i}-{args.h_f}_k_{args.k_i}-{args.k_f}_delta_{args.npoints}_chi_{chi}",
                (args.L-1) // 2,
            )
            np.savetxt(
                f"{parent_path}/results/entropy_data/{args.L // 2}_bond_entropy_{args.model}_spin_{args.L}_h_{args.h_i}-{args.h_f}_k_{args.k_i}-{args.k_f}_delta_{args.npoints}_chi_{chi}",
                entropy_mid,
            )