import argparse
import numpy as np
import datetime as dt
from qs_mps.mps_class import MPS
from qs_mps.utils import get_precision, save_list_of_lists, access_txt, get_cx, get_cy
from qs_mps.applications.Z2.ground_state_multiprocessing import ground_state_Z2
import sys
import time

# DENSITY MATRIX RENORMALIZATION GROUP to find ground states of the Z2 Pure Gauge Theory
# changing the transverse field parameters in its dual formulation

parser = argparse.ArgumentParser(prog="gs_search_Z2")
parser.add_argument("l", help="Number of ladders in the direct lattice", type=int)
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "h_i",
    help="Starting value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "h_f",
    help="Final value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument(
    "-L", "--Ls", help="Number of rungs per ladder", nargs="+", type=int
)
parser.add_argument(
    "-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int
)
parser.add_argument(
    "-cx",
    "--charges_x",
    help="a list of the first index of the charges",
    nargs="*",
    type=int,
)
parser.add_argument(
    "-cy",
    "--charges_y",
    help="a list of the second index of the charges",
    nargs="*",
    type=int,
)
parser.add_argument(
    "-R",
    "--length",
    help="String length in the two particle sector. By default 0 means we are in the vacuum",
    default=0,
    type=int,
)
parser.add_argument(
    "-ty",
    "--type_shape",
    help="Type of shape of the bond dimension. Available are: 'trapezoidal', 'pyramidal', 'rectangular'",
    default="rectangular",
    type=str,
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
)
parser.add_argument(
    "-mu",
    "--multpr",
    help="If True computes ground states with multiprocessing. By default False",
    action="store_true",
)
parser.add_argument(
    "-s",
    "--number_sweeps",
    help="Number of sweeps during the compression algorithm for each trotter step",
    default=10,
    type=int,
)
parser.add_argument(
    "-cv",
    "--conv_tol",
    help="Convergence tolerance of the compression algorithm",
    default=1e-12,
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
    help="Bond where we want to observe the Schmidt values, should be between 1 and (L)",
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
    help="Save all the energies during the variational optimization. By default True",
    action="store_false",
)
parser.add_argument(
    "-i",
    "--interval",
    help="Type of interval spacing. Available are 'log', 'lin'",
    default="lin",
    type=str,
)
parser.add_argument(
    "-bc",
    "--boundcond",
    help="Type of boundary conditions. Available are 'obc', 'pbc'",
    default="pbc",
    type=str,
)
parser.add_argument(
    "-log",
    "--logging",
    help="Name to log the output of the computation",
    default="output.out",
    type=str,
)
parser.add_argument(
    "-cc",
    "--chargeconv",
    help="Type of Charge convension for obc. Available are 'h', 'v'. By default 'h'",
    default="h",
    type=str,
)
parser.add_argument(
    "-p",
    "--precision",
    help="Precision to load and save tensors and observables. By default True 3",
    default=3,
    type=int,
)

args = parser.parse_args()

# # Redirect stdout and stderr to the log file
# sys.stdout = open(f'results/logs/{args.logging}', 'w')
# sys.stderr = sys.stdout


# define the physical dimension
d = int(2 ** (args.l))

# define the interval of equally spaced values of external field
if args.interval == "lin":
    interval = np.linspace(args.h_i, args.h_f, args.npoints)
    # num = (interval[-1] - interval[0]) / args.npoints
    # precision = get_precision(num)
    
elif args.interval == "log":
    interval = np.logspace(args.h_i, args.h_f, args.npoints)
    # precision = int(np.max([np.abs(args.h_f), np.abs(args.h_i)]))

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = f"C:/Users/HP/Desktop/projects/1_Z2"
    # parent_path = "G:/My Drive/projects/1_Z2"
    path_tensor = "D:/code/projects/1_Z2"
elif args.path == "mac":
    # parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/Users/fradm98/Desktop/projects/1_Z2"
    parent_path = path_tensor
elif args.path == "marcos":
    # parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/Users/fradm/Desktop/projects/1_Z2"
    parent_path = path_tensor
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")


# ---------------------------------------------------------
# DMRG
# ---------------------------------------------------------
for L in args.Ls:
    # define the sector by looking of the given charges
    if args.charges_x == [] and args.charges_y == []:
        sector = "vacuum_sector"
        charges_x = np.nan
        charges_y = np.nan
    else:
        sector = f"{len(args.charges_x)}_particle(s)_sector"
        charges_x = args.charges_x
        charges_y = args.charges_y
    # where to look at for the entropy
    if args.where == -1:
        args.where = L // 2
    elif args.where == -2:
        args.bond = False

    if args.length != 0:
        charges_x = get_cx(L, args.length)
        charges_y = get_cy(args.l, args.boundcond, args.charges_y, R=args.length)
        sector = f"{len(charges_x)}_particle(s)_sector"
    # init_state = np.zeros((d))
    # init_state[0] = 1
    # init_state = init_state.reshape((1,d,1))
    # init_tensor = [init_state for _ in range(L)]
    init_tensor = []
    for chi in args.chis:  # L // 2 + 1
        args_mps = {
            "L": L,
            "d": d,
            "chi": chi,
            "type_shape": args.type_shape,
            "model": args.model,
            "trunc_tol": False,
            "trunc_chi": True,
            "where": args.where,
            "bond": args.bond,
            "path": path_tensor,
            "save": args.save,
            "precision": args.precision,
            "sector": sector,
            "charges_x": charges_x,
            "charges_y": charges_y,
            "n_sweeps": args.number_sweeps,
            "conv_tol": args.conv_tol,
            "training": args.training,
            "guess": init_tensor,
            "bc": args.boundcond,
            "cc": args.chargeconv,
        }

        if __name__ == "__main__":
            date_start = dt.datetime.now()
            energy_chi, entropy_chi, schmidt_vals_chi, t_chi = ground_state_Z2(
                args_mps=args_mps, interval=interval, multpr=args.multpr
            )

            t_final = dt.datetime.now() - date_start

            print(f"time of the whole search for chi={chi} is: {t_final}")
            if args.bond == False:
                args.where = "all"

            if args.training:
                energy_chi = np.asarray(energy_chi)
                print(energy_chi.shape, energy_chi[-1])
                energy_chi = energy_chi.reshape((len(interval), len(energy_chi[0])))
                # print(energy_chi.shape)
                np.save(
                    f"{parent_path}/results/energy_data/energies_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                    energy_chi,
                )
                energy_last = []
                for i in range(len(interval)):
                    energy_last.append(energy_chi[i, -1])
                np.save(
                    f"{parent_path}/results/energy_data/energy_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                    energy_last,
                )

            else:
                np.save(
                    f"{parent_path}/results/energy_data/energy_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                    energy_chi,
                )

            save_list_of_lists(
                f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                entropy_chi,
            )
            # save_list_of_lists(
            #     f"{parent_path}/results/entropy_data/{args.where}_schmidt_vals_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
            #     schmidt_vals_chi,
            # )
            print(schmidt_vals_chi)
            np.save(
                f"{parent_path}/results/entropy_data/{args.where}_schmidt_vals_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                schmidt_vals_chi,
            )
            if args.where == "all":
                entropy_mid = access_txt(
                    f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                    (L) // 2,
                )
                np.savetxt(
                    f"{parent_path}/results/entropy_data/{args.L // 2}_bond_entropy_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                    entropy_mid,
                )
