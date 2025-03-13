import matplotlib.pyplot as plt
import numpy as np
from qs_mps.utils import load_list_of_lists, tensor_shapes
from qs_mps.mps_class import MPS
from scipy.optimize import curve_fit

# TIME EVOLVING BLOCK DECIMATION to find evolutions of ground states of the Z2 Pure Gauge Theory
# changing the transverse field parameters in its dual formulation for the Quench Hamiltonian

import argparse
from qs_mps.mps_class import MPS
from qs_mps.utils import *

import datetime as dt

parser = argparse.ArgumentParser(prog="Time Ev")
parser.add_argument("l", help="Number of ladders in the direct lattice", type=int)
parser.add_argument(
    "npoints",
    help="Trotter steps for the quench dynamics",
    type=int,
)
parser.add_argument(
    "delta",
    help="Width of each time slice during the time evolution. Should be 'small enough'",
    type=float,
)
parser.add_argument(
    "h_i",
    help="Starting value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "h_ev",
    help="Quench value of h (external transverse field on the dual lattice)",
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
    "-o",
    "--obs",
    help="Observable we want to compute. Available are 'el', 'losch'",
    nargs="*",
    type=str,
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
# parser.add_argument(
#     "-lx", "--sites", help="Number of sites in the wilson loop", nargs="*", type=int
# )
# parser.add_argument(
#     "-ly", "--ladders", help="Number of ladders in the wilson loop", nargs="*", type=int
# )
# parser.add_argument(
#     "-d", "--direction", help="Direction of the string", default="hor", type=str
# )
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
    "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
)
parser.add_argument(
    "-mo",
    "--moment",
    help="Moment degree of the Free energy. E.g. Magnetization -> First Moment, Susceptibility -> Second Moment, etc. Available are 1,2,4",
    default=1,
    type=int,
)
parser.add_argument(
    "-tr",
    "--training",
    help="Save all the errors during the variational compression. By default False",
    action="store_true",
)
parser.add_argument(
    "-ex",
    "--exact",
    help="Compare MPS evolution with exact. Reasonable for small systems under 20 dof. By default False",
    action="store_true",
)
parser.add_argument(
    "-bc",
    "--boundcond",
    help="Type of boundary conditions. Available are 'obc', 'pbc'",
    default="pbc",
    type=str,
)
parser.add_argument(
    "-Dmax",
    "--chi_max",
    help="Bond dimension for the initial DMRG",
    default=128,
    type=int,
)
parser.add_argument(
    "-qq",
    "--quantify_quench",
    help="""Before doing the evolution we want to evaluate qualitatively the quench. 
        Compute the overlap of ground states and the energy evaluated in the with the quench hamiltonian.
        By default False""",
    action="store_true",
)
parser.add_argument(
    "-of",
    "--obs_freq",
    help="Frequency of sampling observables during our time evolution. It is expressend in percentage of trotter steps",
    default=0.3, # this means the 30% of the trotter steps we will measure the observables
    type=float,
)
parser.add_argument(
    "-p",
    "--precision",
    help="Precision to load and save tensors and observables. By default True will take the number of decimals in h_i",
    default=3,
    type=int,
)

args = parser.parse_args()

# define the physical dimension
d = int(2 ** (args.l))

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


# # for the wilson loop
# if args.sites == 1:
#     sites = 0
# if args.ladders == 1:
#     ladders = 1

# # define the direction
# if args.direction == "ver":
#     direction = "vertical"
# elif args.direction == "hor":
#     direction = "horizontal"

# define moment
if args.moment == 1:
    moment = "first"
if args.moment == 2:
    moment = "second"
if args.moment == 4:
    moment = "fourth"

# define all observables
if args.obs == []:
    args.obs = ["el", "end", "losch"]

a = np.zeros((1,2))
a[0,0] = 1
aux_qub = a.reshape((1,2,1))

for L in args.Ls:
    # define the sector by looking of the given charges
    if len(args.charges_x) == 0:
        sector = "vacuum_sector"
        charges_x = None
        charges_y = None
    else:
        sector = f"{len(args.charges_x)}_particle(s)_sector"
        charges_x = args.charges_x
        charges_y = args.charges_y

    if args.length != 0:
        charges_x = get_cx(L, args.length)
        charges_y = get_cy(args.l, args.boundcond)
        sector = f"{len(charges_x)}_particle(s)_sector"

    # where to look at for the entropy
    if args.where == -1:
        args.where = L // 2
    elif args.where == -2:
        args.bond = False

    for chi in args.chis:
        lattice_mps = MPS(
                L=L, d=d, model=args.model, chi=args.chi_max, h=args.h_i, bc=args.boundcond
            )

        sector_vac = "vacuum_sector"
        cx_vac = np.nan
        cy_vac = np.nan
        if sector_vac != "vacuum_sector":
            lattice_mps.Z2.add_charges(cx_vac, cy_vac)
            lattice_mps.charges = lattice_mps.Z2.charges
            lattice_mps.Z2._define_sector()
        else:
            lattice_mps.Z2._define_sector()
        try:
            lattice_mps.load_sites(
                path=path_tensor, precision=args.precision, cx=cx_vac, cy=cy_vac
            )
            print("State found!!")
            if args.bond:
                try:
                    entropy = load_list_of_lists(f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_direct_lattice_{args.l}x{L}_{sector_vac}_bc_{args.boundcond}_{cx_vac}-{cy_vac}_h_{args.h_i}_delta_{args.npoints}_chi_{chi}")
                except:
                    lattice_mps.canonical_form(svd_direction="right", trunc_chi=False, trunc_tol=True, schmidt_tol=1e-15)
                    entropy = von_neumann_entropy(lattice_mps.bonds[L//2])
                    print("Entropy of initial state for the middle MPS bond")
                    print(entropy)
            else:
                try:
                    entropy = load_list_of_lists(f"{parent_path}/results/entropy_data/all_bond_entropy_{args.model}_direct_lattice_{args.l}x{L}_{sector_vac}_bc_{args.boundcond}_{cx_vac}-{cy_vac}_h_{args.h_i}_delta_{args.npoints}_chi_{chi}")
                except:
                    lattice_mps.canonical_form(svd_direction="right", trunc_chi=False, trunc_tol=True, schmidt_tol=1e-15)
                    entropy = [von_neumann_entropy(lattice_mps.bonds[i]) for i in range(L-1)]
                    print("Entropy of initial state for all of the MPS bonds")
                    print(entropy)

        except:
            print("State not found! Computing DMRG")
            lattice_mps._random_state(seed=3, type_shape="rectangular", chi=args.chi_max)
            lattice_mps.canonical_form()
            lattice_mps.sites.append(np.random.rand(1,2,1))
            lattice_mps.L = len(lattice_mps.sites)
            energy, entropy, schmidt_vals, t_dmrg = lattice_mps.DMRG(trunc_chi=True, trunc_tol=False, bond=False, long="Z", trans="X")
            lattice_mps.check_canonical(site=1)
            aux_qub = lattice_mps.sites.pop()
            lattice_mps.L -= 1

            lattice_mps.order_param()
            mag = lattice_mps.mpo_first_moment()
            print(f"initial magentization is: {mag}")

            lattice_mps.save_sites(path=path_tensor, precision=args.precision, cx=cx_vac, cy=cy_vac)

        # initialize the variables to save
        errors_tr = [[0, 0]]
        errors = [0]
        entropies_ev = [entropy]

        # ---------------------------------------------------------
        # Trotter Evolution
        # ---------------------------------------------------------
        if sector != "vacuum_sector":
            lattice_mps.Z2.add_charges(charges_x, charges_y)
            lattice_mps.charges = lattice_mps.Z2.charges
            lattice_mps.Z2._define_sector()
        else:
            lattice_mps.Z2._define_sector()

        lattice_mps.chi = chi

        # quantify quench
        if args.quantify_quench:
            print("==============================")
            print("Quantify the quench")
            lattice_mps.sites.append(aux_qub)
            lattice_mps.L = len(lattice_mps.sites)
            lattice_mps.mpo()
            E_init = lattice_mps.mpo_first_moment().real
            aux_qub = lattice_mps.sites.pop()
            lattice_mps.L -= 1
        
        if args.quantify_quench:
            mps_gs_quench = MPS(
                    L=L, d=d, model=args.model, chi=args.chi_max, h=args.h_i, bc=args.boundcond
                )

            if sector != "vacuum_sector":
                mps_gs_quench.Z2.add_charges(charges_x, charges_y)
                mps_gs_quench.charges = mps_gs_quench.Z2.charges
                mps_gs_quench.Z2._define_sector()
            else:
                mps_gs_quench.Z2._define_sector()
            try:
                mps_gs_quench.load_sites(
                    path=path_tensor, precision=args.precision, cx=charges_x, cy=charges_y
                )
                print("State found!!")
                if args.bond:
                    try:
                        entropy = load_list_of_lists(f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}_delta_{args.npoints}_chi_{chi}")
                    except:
                        mps_gs_quench.canonical_form(svd_direction="right", trunc_chi=False, trunc_tol=True, schmidt_tol=1e-15)
                        entropy = von_neumann_entropy(mps_gs_quench.bonds[L//2])
                        print("Entropy of initial state for the middle MPS bond")
                        print(entropy)
                else:
                    try:
                        entropy = load_list_of_lists(f"{parent_path}/results/entropy_data/all_bond_entropy_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}_delta_{args.npoints}_chi_{chi}")
                    except:
                        mps_gs_quench.canonical_form(svd_direction="right", trunc_chi=False, trunc_tol=True, schmidt_tol=1e-15)
                        entropy = [von_neumann_entropy(mps_gs_quench.bonds[i]) for i in range(L-1)]
                        print("Entropy of initial state for all of the MPS bonds")
                        print(entropy)

            except:
                print("State not found! Computing DMRG")
                mps_gs_quench._random_state(seed=3, type_shape="rectangular", chi=args.chi_max)
                mps_gs_quench.canonical_form()
                mps_gs_quench.sites.append(np.random.rand(1,2,1))
                mps_gs_quench.L = len(mps_gs_quench.sites)
                energy, entropy, schmidt_vals, t_dmrg = mps_gs_quench.DMRG(trunc_chi=True, trunc_tol=False, bond=False)
                mps_gs_quench.check_canonical(site=1)
                aux_qub_quench = mps_gs_quench.sites.pop()
                mps_gs_quench.L -= 1

                mps_gs_quench.order_param()
                mag = mps_gs_quench.mpo_first_moment()
                print(f"initial magentization is: {mag}")

                mps_gs_quench.save_sites(path=path_tensor, precision=args.precision, cx=charges_x, cy=charges_y)

            mps_gs_quench.sites.append(aux_qub)
            mps_gs_quench.L = len(mps_gs_quench.sites)
            mps_gs_quench.mpo()
            E_1 = mps_gs_quench.mpo_first_moment().real
            aux_qub = mps_gs_quench.sites.pop()
            mps_gs_quench.L -= 1

            lattice_mps.ancilla_sites = mps_gs_quench.sites.copy()
            fidelity = lattice_mps._compute_norm(site=1, mixed=True)
            lattice_mps.ancilla_sites = []
            print("Energy of H_1 over psi_0: ",E_init, ", Energy of H_1 over psi_1", E_1)
            print(f"Relative Difference (wrt E_1): {(E_1 - E_init)/E_1} \n\n")
            print(f"Fidelity between the two ground states is: {fidelity}")
            break

        
        date_start = dt.datetime.now()
        print(f"\n*** Starting TEBD evolution in {dt.datetime.now()} ***\n")
        # trotter evolution
        (errs,
        entrs,
        svs,
        efields,
        transversal_fields_string,
        losch,
        ex_sp,
        ex_mps,
        mps_sp) = lattice_mps.TEBD_variational_Z2_exact(
            trotter_steps=args.npoints,
            delta=args.delta,
            h_ev=args.h_ev,
            n_sweeps=8,
            conv_tol=1e-12,
            bond=args.bond,
            where=L//2,
            aux_qub=aux_qub,
            cx=charges_x,
            cy=charges_y,
            exact=args.exact,
            obs=args.obs,
            obs_freq=args.obs_freq,
            training=args.training,
            chi_max=args.chi_max
            )

        t_final = dt.datetime.now() - date_start
        print(f"Total time for TEBD evolution of {args.npoints} trotter steps is: {t_final}")
        
        if "el" in args.obs:
            np.save(
                f"{parent_path}/results/electric_field/electric_field_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                efields,
            )

        if "end" in args.obs:
            np.save(
                f"{parent_path}/results/electric_field/electric_field_energy_density_middle_column_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                transversal_fields_string,
            )

        if args.bond == False:
            args.where = "all"
            entrs = np.asarray(entrs)[:,1:]
            entrs_mid = np.asarray(entrs)[:,L//2]
        else:
            entrs = np.asarray(entrs)
    
        if args.training:
            save_list_of_lists(
                f"{parent_path}/results/error_data/errors_tr_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                errs,
            )
            last_errs = []
            for err in errs:
                last_errs.append(err[-1])
            np.save(
                f"{parent_path}/results/error_data/errors_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                last_errs,
            )
        else:
            np.save(
                f"{parent_path}/results/error_data/errors_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                errs,
            )
        
        np.save(
            f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
            entrs,
        )
        save_list_of_lists(
            f"{parent_path}/results/entropy_data/{args.where}_schmidt_vals_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
            svs,
        )
        if args.where == "all":
            
            # entropy_mid = access_txt(
            #     f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
            #     (L - 1) // 2,
            # )
            np.save(
                f"{parent_path}/results/entropy_data/{L // 2}_bond_entropy_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                entrs_mid,
            )
        
        np.save(
                f"{parent_path}/results/overlap/loschmidt_amplitudes_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                losch,
            )
        if args.exact:
            np.save(
                f"{parent_path}/results/exact/overlap/exact_vs_mps_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                ex_mps,
            )
            np.save(
                f"{parent_path}/results/exact/overlap/exact_vs_sparse_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                ex_sp,
            )
            np.save(
                f"{parent_path}/results/exact/overlap/mps_vs_sparse_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                mps_sp,
            )