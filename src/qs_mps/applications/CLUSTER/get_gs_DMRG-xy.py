import argparse
import numpy as np
from qs_mps.mps_class import MPS
from qs_mps.utils import get_precision, save_list_of_lists, access_txt, swap_rows

# DENSITY MATRIX RENORMALIZATION GROUP to find ground states of the Cluster Theory 
# changing the transverse field parameters

parser = argparse.ArgumentParser(prog="gs_search_Cluster")
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "hx_i", help="Starting value of h (external transverse field)", type=float
)
parser.add_argument(
    "hx_f", help="Final value of h (external transverse field)", type=float
)
parser.add_argument(
    "hy_i", help="Starting value of h (external transverse field)", type=float
)
parser.add_argument(
    "hy_f", help="Final value of h (external transverse field)", type=float
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("-L", "--Ls", help="Number of sites in the chain", nargs="+", type=int)
parser.add_argument("-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument(
    "-ty", "--type_shape", help="Type of shape of the bond dimension. Available are: 'trapezoidal', 'pyramidal', 'rectangular'", default="rectangular", type=str
)
parser.add_argument(
    "-m", "--model", help="Model to simulate. By default Cluster", default="Cluster-XY", type=str
)
parser.add_argument(
    "-mu", "--multpr", help="If True computes ground states with multiprocessing. By default False", action="store_true"
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
    help="Save all the energies during the variational optimization. By default False",
    action="store_true",
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
    interval = np.linspace(args.hx_i, args.hx_f, args.npoints)
    num = (args.hx_f - args.hx_i) / args.npoints
    precision = get_precision(num)
elif args.interval == "log":
    interval = np.logspace(args.hx_i, args.hx_f, args.npoints)
    num = (interval[-1]-interval[0]) / args.npoints
    precision = get_precision(num)

interval_hx = interval
interval_hy = np.linspace(args.hy_i, args.hy_f, args.npoints)


# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/3_CLUSTER"
    path_tensor = "D:/code/projects/3_CLUSTER"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/3_CLUSTER"
    path_tensor = "/Users/fradm98/Desktop/projects/3_CLUSTER"
elif args.path == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/3_CLUSTER"
    path_tensor = "/Users/fradm/Desktop/projects/3_CLUSTER"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")




# ---------------------------------------------------------
# DMRG
# ---------------------------------------------------------
for L in args.Ls:
    # where to look at for the entropy
    if args.where == -1:
        args.where = L // 2
    elif args.where == -2:
        args.bond = False
    for chi in args.chis:  # L // 2 + 1

        energy_chi = []
        entropy_chi = []
        schmidt_vals_chi = []
        for hy in interval_hy:
            energy_J = []
            entropy_J = []
            schmidt_vals_J = []
            for hx in interval_hx:
                chain = MPS(L=L, d=d, chi=chi, model=args.model, eps=1e-5, h=hx, lx=0, ly=hy, J=1)
                if hx == interval_hx[0]:
                    chain._random_state(seed=3,chi=chi)
                    chain.canonical_form()
                else:
                    chain.sites = init_tensor.copy()
                # chain.enlarge_chi()
                energy, entropy, schmidt_vals = chain.DMRG(trunc_tol=False, trunc_chi=True, long="X", trans="Z", schmidt_tol=1e-15, conv_tol=1e-10, n_sweeps=2, bond=True, where=L//2)
                print(f"energy of hx:{hx:.{precision}f}, hy:{hy:.{precision}f} is:\n {energy}")
                print(f"Schmidt values in the middle of the chain:\n {schmidt_vals}")
                print(f"Entropy: {entropy}")

                if args.training:
                    energy_J.append(energy)
                else:
                    energy_J.append(energy[-1])
                entropy_J.append(entropy)
                schmidt_vals_J.append(schmidt_vals)
                chain.save_sites(path=path_tensor, precision=precision)
                init_tensor = chain.sites.copy()

            # energy_J.reverse()
            # entropy_J.reverse()
            # schmidt_vals_J.reverse()
            energy_chi.append(energy_J)
            entropy_chi.append(entropy_J)
            schmidt_vals_chi.append(schmidt_vals_J)

        if args.bond == False:
            args.where = "all"

        if args.training:
            energy_chi = np.swapaxes(swap_rows(np.asarray(energy_chi)), axis1=0, axis2=1)
            np.save(
                f"{parent_path}/results/energy_data/energies_{args.model}_L_{L}_h_{args.hx_i}-{args.hx_f}_J_{args.hy_i}-{args.hy_f}_delta_{args.npoints}_chi_{chi}",
                energy_chi,
            )
            for i in range(len(interval_hy)):
                for j in range(len(interval_hx)):
                    np.save(f"{parent_path}/results/energy_data/energy_{args.model}_L_{L}_h_{args.hx_i}-{args.hx_f}_J_{args.hy_i}-{args.hy_f}_delta_{args.npoints}_chi_{chi}", energy_chi[i][j][-1])

        else:
            np.save(
                f"{parent_path}/results/energy_data/energy_{args.model}_L_{L}_h_{args.hx_i}-{args.hx_f}_J_{args.hy_i}-{args.hy_f}_delta_{args.npoints}_chi_{chi}",
                energy_chi,
            )
            
        save_list_of_lists(
            f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_L_{L}_h_{args.hx_i}-{args.hx_f}_J_{args.hy_i}-{args.hy_f}_delta_{args.npoints}_chi_{chi}",
            entropy_chi,
        )
        save_list_of_lists(
            f"{parent_path}/results/entropy_data/{args.where}_schmidt_vals_{args.model}_L_{L}_h_{args.hx_i}-{args.hx_f}_J_{args.hy_i}-{args.hy_f}_delta_{args.npoints}_chi_{chi}",
            schmidt_vals_chi,
        )
        if args.where == "all":
            entropy_mid = access_txt(
                f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_L_{L}_h_{args.hx_i}-{args.hx_f}_J_{args.hy_i}-{args.hy_f}_delta_{args.npoints}_chi_{chi}",
                (L-1) // 2,
            )
            np.savetxt(
                f"{parent_path}/results/entropy_data/{args.L // 2}_bond_entropy_{args.model}_L_{L}_h_{args.hx_i}-{args.hx_f}_J_{args.hy_i}-{args.hy_f}_delta_{args.npoints}_chi_{chi}",
                entropy_mid,
            )