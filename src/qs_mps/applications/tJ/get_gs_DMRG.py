import argparse
import numpy as np
import datetime as dt
from qs_mps.mps_class import MPS
from qs_mps.utils import get_precision, save_list_of_lists, access_txt, get_cx, get_cy
from qs_mps.applications.Z2.ground_state_multiprocessing import ground_state_Z2, ground_state_heis

# DENSITY MATRIX RENORMALIZATION GROUP to find ground states of the Z2 Pure Gauge Theory
# changing the transverse field parameters in its dual formulation

parser = argparse.ArgumentParser(prog="gs_search_tJV_model")
parser.add_argument("d", help="Number of states for one tensor", type=int)
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "Jz",
    help="zz coupling Jz",
    type=float,
)
parser.add_argument(
    "J_perp",
    help="S^+S^- coupling J_perp",
    type=float,
)
parser.add_argument(
    "t",
    help="hopping i,i+1 coupling t",
    type=float,
)
parser.add_argument(
    "tp",
    help="hopping i,i+2 coupling tp",
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
    "-ty",
    "--type_shape",
    help="Type of shape of the bond dimension. Available are: 'trapezoidal', 'pyramidal', 'rectangular'",
    default="rectangular",
    type=str,
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="tj", type=str
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
    "-bc",
    "--boundcond",
    help="Type of boundary conditions. Available are 'obc', 'pbc'",
    default="obc",
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
parser.add_argument(
    "-e",
    "--eps",
    help="Value of the epsilon coupling to penalize holes creation. By default 0",
    default=0,
    type=float,
)
parser.add_argument(
    "-exc",
    "--excited",
    help="First excited state. By default False",
    action="store_true",
)

args = parser.parse_args()

interval = np.linspace(args.t,args.t,args.npoints)
# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = f"C:/Users/HP/Desktop/projects/6_TJ"
    # parent_path = "G:/My Drive/projects/6_TJ"
    path_tensor = "D:/code/projects/6_TJ"
    parent_path = path_tensor
elif args.path == "mac":
    # parent_path = "/Users/fradm98/Google Drive/My Drive/projects/6_TJ"
    path_tensor = "/Users/fradm98/Desktop/projects/6_TJ"
    parent_path = path_tensor
elif args.path == "marcos":
    # parent_path = "/Users/fradm/Google Drive/My Drive/projects/6_TJ"
    path_tensor = "/Users/fradm/Desktop/projects/6_TJ"
    parent_path = path_tensor
elif args.path == "ngt":
    # parent_path = "/Users/fradm/Google Drive/My Drive/projects/6_TJ"
    path_tensor = "/eos/user/f/fdimarca/projects/6_TJ"
    parent_path = path_tensor
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos', 'ngt'")

precision = 3
# ---------------------------------------------------------
# DMRG
# ---------------------------------------------------------
for L in args.Ls:
    if args.where == -1:
        args.where = L // 2
    elif args.where == -2:
        args.bond = False

    init_tensor = []
    for chi in args.chis:  # L // 2 + 1
        args_mps = {
            "L": L,
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
            "precision": args.precision,
            "n_sweeps": args.number_sweeps,
            "conv_tol": args.conv_tol,
            "training": args.training,
            "guess": init_tensor,
            "bc": args.boundcond,
            "Jz": args.Jz,
            "J_perp": args.J_perp,
            "t": args.t,
            "tp": args.tp,
            "eps": args.eps,
            "excited": args.excited,
        }

        if __name__ == "__main__":
            date_start = dt.datetime.now()
            energy_chi, entropy_chi, schmidt_vals_chi, t_chi = ground_state_heis(
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
                if args.excited:
                    np.save(
                        f"{parent_path}/results/energy_data/first_excited_energies_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                        energy_chi,
                    )
                else:
                    np.save(
                        f"{parent_path}/results/energy_data/energies_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                        energy_chi,
                    )
                energy_last = []
                for i in range(len(interval)):
                    energy_last.append(energy_chi[i, -1])
                
                if args.excited:
                    np.save(
                        f"{parent_path}/results/energy_data/first_excited_energy_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                        energy_last,
                    )
                else:
                    np.save(
                        f"{parent_path}/results/energy_data/energy_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                        energy_last,
                    )

            else:
                if args.excited:
                    np.save(
                        f"{parent_path}/results/energy_data/first_excited_energy_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                        energy_chi,
                    )
                else:
                    np.save(
                    f"{parent_path}/results/energy_data/energy_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                    energy_chi,
                    )

            if args.excited:
                save_list_of_lists(
                    f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_first_excited_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                    entropy_chi,
                )
            else:
                save_list_of_lists(
                    f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                    entropy_chi,
                )

            # save_list_of_lists(
            #     f"{parent_path}/results/entropy_data/{args.where}_schmidt_vals_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
            #     schmidt_vals_chi,
            # )
            print(schmidt_vals_chi)
            if args.excited:
                np.save(
                    f"{parent_path}/results/entropy_data/{args.where}_schmidt_vals_first_excited_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                    schmidt_vals_chi,
                )
            else:
                np.save(
                    f"{parent_path}/results/entropy_data/{args.where}_schmidt_vals_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                    schmidt_vals_chi,
                )

            if args.where == "all":
                if args.excited:
                    entropy_mid = access_txt(
                        f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_first_excited_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                        (L) // 2,
                    )
                    np.savetxt(
                        f"{parent_path}/results/entropy_data/{L // 2}_bond_entropy_first_excited_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                        entropy_mid,
                    )
                else:
                    entropy_mid = access_txt(
                        f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                        (L) // 2,
                    )
                    np.savetxt(
                        f"{parent_path}/results/entropy_data/{L // 2}_bond_entropy_{args.model}_lattice_L_{L}_bc_{args.boundcond}_chi_{chi}_Jz_{args.Jz:.{precision}f}_J_perp_{args.J_perp:.{precision}f}_t_{args.t:.{precision}f}_tp_{args.tp:.{precision}f}",
                        entropy_mid,
                    )

