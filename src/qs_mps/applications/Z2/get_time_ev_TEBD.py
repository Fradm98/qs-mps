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
parser.add_argument(
    "-res",
    "--restart",
    help="Restart a computation starting from a tensor saved temporarily. By default None",
    action="store_true",
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
elif args.path == "ngtcern":
    # parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/shared/projects/1_Z2"
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
    if len(args.charges_x) == 0 and len(args.charges_y) == 0:
        sector = "vacuum_sector"
        charges_x = None
        charges_y = None
    else:
        sector = f"{len(args.charges_x)}_particle(s)_sector"
        charges_x = args.charges_x
        charges_y = args.charges_y

    if args.length != 0:
        charges_x = get_cx(L, args.length, cx=charges_x)
        charges_y = get_cy(args.l, args.boundcond, R=args.length, cy=charges_y)
        sector = f"{len(charges_x)}_particle(s)_sector"

    # where to look at for the entropy
    if args.where == -1:
        args.where = L // 2
    elif args.where == -2:
        args.bond = False

    n_sweeps = 8

    # create a run group for saving observables
    h5file = f"{parent_path}/results/time_data/results_time.hdf5"
    params = dict(L=L, N=args.l, R=args.length, T=args.npoints, bc=args.boundcond, chis=args.chis,
                cx=charges_x, cy=charges_y, delta=args.delta, 
                h_ev=args.h_ev, h_i=args.h_i, of=args.obs_freq)
    
    run_group = create_run_group(h5file, params)

    for chi in args.chis:
        if args.chi_max < chi:
            args.chi_max = chi

        lattice_mps = MPS(
                L=L, d=d, model=args.model, chi=args.chi_max, h=args.h_i, bc=args.boundcond
            )
        if args.restart:
            with h5py.File(h5file, 'r') as f:
                grp = f[run_group]
                entrs = grp[f'entropies/D_{chi}/values'][:]

            trotter_steps = len(entrs) - 1
            last = np.nonzero(entrs)[0][-1]
            args.npoints = (trotter_steps + 1) - last
            if args.npoints == 1:
                print(f"bond dimension {chi} already computed...")
                args.restart = False
                continue

            filename = f"/results/tensors/time_evolved_tensor_sites_Z2_dual_direct_lattice_{args.l}x{L}_bc_{args.boundcond}_2_particle(s)_sector_{charges_x}-{charges_y}_chi_{chi}_h_{args.h_ev:.{args.precision}f}_delta_{args.delta}_trotter_{trotter_steps}.h5"
            lattice_mps.load_sites(path=path_tensor, filename=filename)            
        else:
            last = 0
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
                        schmidt_vals = lattice_mps.bonds[L//2]
                        print("Entropy of initial state for the middle MPS bond")
                        print(entropy)
                else:
                    try:
                        entropy = load_list_of_lists(f"{parent_path}/results/entropy_data/all_bond_entropy_{args.model}_direct_lattice_{args.l}x{L}_{sector_vac}_bc_{args.boundcond}_{cx_vac}-{cy_vac}_h_{args.h_i}_delta_{args.npoints}_chi_{chi}")
                    except:
                        lattice_mps.canonical_form(svd_direction="right", trunc_chi=False, trunc_tol=True, schmidt_tol=1e-15)
                        entropy = [von_neumann_entropy(lattice_mps.bonds[i]) for i in range(L-1)]
                        schmidt_vals = lattice_mps.bonds[L//2]
                        print("Entropy of initial state for all of the MPS bonds")
                        print(entropy)

            except:
                print("State not found! Computing DMRG")
                lattice_mps._random_state(seed=3, type_shape="rectangular", chi=args.chi_max)
                lattice_mps.canonical_form()
                # lattice_mps.sites.append(np.random.rand(1,2,1))
                # lattice_mps.L = len(lattice_mps.sites)
                energy, entropy, schmidt_vals, t_dmrg = lattice_mps.DMRG(trunc_chi=True, trunc_tol=False, bond=False, long="Z", trans="X")
                lattice_mps.check_canonical(site=1)
                # aux_qub = lattice_mps.sites.pop()
                # lattice_mps.L -= 1

                lattice_mps.order_param()
                mag = lattice_mps.mpo_first_moment()
                print(f"initial magentization is: {mag}")

                lattice_mps.save_sites(path=path_tensor, precision=args.precision, cx=cx_vac, cy=cy_vac)
                
                if args.bond:
                    entropy = entropy[L//2]
                    schmidt_vals = np.array(schmidt_vals[L//2])

            # initialize the variables to save
            errors_tr = [[0, 0]]
            errors = [0]
            entropies_ev = [entropy]
            
            # create observables group and save them
            
            if args.training:
                errors = np.zeros((L-1+1)*n_sweeps) # should be L-1 but we have an ancillary qubit on the right
                shape_err = (L - 1 + 1)*n_sweeps
                name_err = f'errors_trunc/D_{chi}/trotter_step_{0:03d}'
                create_observable_group(h5file, run_group, name_err)
                prepare_observable_group(h5file, run_group, name_err, shape=shape_err)
                update_observable(h5file, run_group, name_err, data=errors, attr=0)
            else:
                errors = np.array([0])
                shape_err = args.npoints + 1
                name_err = f'errors_trunc/D_{chi}'
                create_observable_group(h5file, run_group, name_err)
                prepare_observable_group(h5file, run_group, name_err, shape=shape_err)
                update_observable(h5file, run_group, name_err, data=errors, attr=0, assign_all=False)

            if args.bond:
                entropies = np.array([entropy])
                shape_entr = args.npoints + 1
                name_entr = f'entropies/D_{chi}'
                create_observable_group(h5file, run_group, name_entr)
                prepare_observable_group(h5file, run_group, name_entr, shape=shape_entr)
                update_observable(h5file, run_group, name_entr, data=entropies, attr=0, assign_all=False)
            else:
                entropies = np.array(entropy)
                shape_entr = (L - 1)
                name_entr = f'entropies/D_{chi}/trotter_step_{0:03d}'
                create_observable_group(h5file, run_group, name_entr)
                prepare_observable_group(h5file, run_group, name_entr, shape=shape_entr)
                update_observable(h5file, run_group, name_entr, data=entropies, attr=0)

            shape_sm = len(schmidt_vals)
            name_sm = f'schmidt_values/D_{chi}/trotter_step_{0:03d}'
            create_observable_group(h5file, run_group, name_sm)
            prepare_observable_group(h5file, run_group, name_sm, shape=shape_sm)
            update_observable(h5file, run_group, name_sm, data=schmidt_vals, attr=0)

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
        # (errs,
        # entrs,
        # svs,
        # efields,
        # # transversal_fields_string,
        # losch,
        # ex_sp,
        # ex_mps,
        # mps_sp) = 
        lattice_mps.TEBD_variational_Z2_exact(
            trotter_steps=args.npoints,
            delta=args.delta,
            h_ev=args.h_ev,
            n_sweeps=n_sweeps,
            conv_tol=1e-15,
            bond=args.bond,
            where=L//2,
            aux_qub=aux_qub,
            cx=charges_x,
            cy=charges_y,
            exact=args.exact,
            obs=args.obs,
            obs_freq=args.obs_freq,
            training=args.training,
            chi_max=args.chi_max,
            path=path_tensor,
            run_group=run_group,
            save_file=h5file,
            restart=last
            )

        t_final = dt.datetime.now() - date_start
        print(f"Total time for TEBD evolution of {args.npoints} trotter steps is: {t_final}")
        
        # if "el" in args.obs:
        #     np.save(
        #         f"{parent_path}/results/electric_field/electric_field_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #         efields,
        #     )

        # # if "end" in args.obs:
        # #     np.save(
        # #         f"{parent_path}/results/electric_field/electric_field_energy_density_middle_column_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        # #         transversal_fields_string,
        # #     )

        # if args.bond == False:
        #     args.where = "all"
        #     entrs = np.asarray(entrs)[:,1:]
        #     entrs_mid = np.asarray(entrs)[:,L//2]
        # else:
        #     entrs = np.asarray(entrs)
    
        # if args.training:
        #     save_list_of_lists(
        #         f"{parent_path}/results/error_data/errors_tr_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #         errs,
        #     )
        #     last_errs = []
        #     for err in errs:
        #         last_errs.append(err[-1])
        #     np.save(
        #         f"{parent_path}/results/error_data/errors_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #         last_errs,
        #     )
        # else:
        #     np.save(
        #         f"{parent_path}/results/error_data/errors_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #         errs,
        #     )
        
        # np.save(
        #     f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #     entrs,
        # )
        # save_list_of_lists(
        #     f"{parent_path}/results/entropy_data/{args.where}_schmidt_vals_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #     svs,
        # )
        # if args.where == "all":
            
        #     # entropy_mid = access_txt(
        #     #     f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #     #     (L - 1) // 2,
        #     # )
        #     np.save(
        #         f"{parent_path}/results/entropy_data/{L // 2}_bond_entropy_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #         entrs_mid,
        #     )
        
        # np.save(
        #         f"{parent_path}/results/overlap/loschmidt_amplitudes_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #         losch,
        #     )
        # if args.exact:
        #     np.save(
        #         f"{parent_path}/results/exact/overlap/exact_vs_mps_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #         ex_mps,
        #     )
        #     np.save(
        #         f"{parent_path}/results/exact/overlap/exact_vs_sparse_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #         ex_sp,
        #     )
        #     np.save(
        #         f"{parent_path}/results/exact/overlap/mps_vs_sparse_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_R_{args.length}_off-{charges_y}_h_{args.h_i}-{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
        #         mps_sp,
        #     )

el_field_trott_162 = np.array([[        np.nan,  0.85275735,         np.nan,  0.84392189,         np.nan,
         0.84380346,         np.nan,  0.84521995,         np.nan,  0.84452911,
                np.nan,  0.84362721,         np.nan,  0.84394122,         np.nan,
         0.84419485,         np.nan,  0.83853297,         np.nan,  0.76748244,
                np.nan,  0.04108998,         np.nan, -0.29344166,         np.nan,
        -0.41335215,         np.nan, -0.43569351,         np.nan, -0.440578  ,
                np.nan, -0.44628763,         np.nan, -0.45394944,         np.nan,
        -0.4579473 ,         np.nan, -0.45360104,         np.nan, -0.44516122,
                np.nan, -0.44553026,         np.nan, -0.45393572,         np.nan,
        -0.45790119,         np.nan, -0.45371965,         np.nan, -0.44573678,
                np.nan, -0.44055471,         np.nan, -0.43573396,         np.nan,
        -0.41336138,         np.nan, -0.29352083,         np.nan,  0.04117925,
                np.nan,  0.76752845,         np.nan,  0.8385436 ,         np.nan,
         0.8441874 ,         np.nan,  0.8439447 ,         np.nan,  0.84375721,
                np.nan,  0.84444034,         np.nan,  0.84458047,         np.nan,
         0.8440738 ,         np.nan,  0.84676133,         np.nan,  0.89709843,
                np.nan],
       [ 0.92109495,         np.nan,  0.84831748,         np.nan,  0.8438095 ,
                np.nan,  0.84441927,         np.nan,  0.84486165,         np.nan,
         0.84399572,         np.nan,  0.84372384,         np.nan,  0.84400346,
                np.nan,  0.84135481,         np.nan,  0.80294606,         np.nan,
         0.47381738,         np.nan,  0.51631792,         np.nan,  0.59259887,
                np.nan,  0.60854485,         np.nan,  0.61488456,         np.nan,
         0.61956571,         np.nan,  0.62319499,         np.nan,  0.62115128,
                np.nan,  0.61318516,         np.nan,  0.60012279,         np.nan,
         0.58907398,         np.nan,  0.60110435,         np.nan,  0.61310449,
                np.nan,  0.62101866,         np.nan,  0.62324316,         np.nan,
         0.62005862,         np.nan,  0.61476129,         np.nan,  0.60857566,
                np.nan,  0.5925422 ,         np.nan,  0.51629041,         np.nan,
         0.47384687,         np.nan,  0.80295498,         np.nan,  0.84135104,
                np.nan,  0.8440055 ,         np.nan,  0.8437624 ,         np.nan,
         0.84401071,         np.nan,  0.84456467,         np.nan,  0.84423262,
                np.nan,  0.84414616,         np.nan,  0.84877668,         np.nan,
         0.89709843],
       [        np.nan,  0.85275619,         np.nan,  0.8439142 ,         np.nan,
         0.84380903,         np.nan,  0.84533988,         np.nan,  0.84462582,
                np.nan,  0.8429497 ,         np.nan,  0.84317779,         np.nan,
         0.84494557,         np.nan,  0.84072281,         np.nan,  0.79812827,
                np.nan,  0.50255419,         np.nan,  0.42082873,         np.nan,
         0.42724557,         np.nan,  0.45585302,         np.nan,  0.47534029,
                np.nan,  0.48351949,         np.nan,  0.48537357,         np.nan,
         0.4854825 ,         np.nan,  0.48736265,         np.nan,  0.48860273,
                np.nan,  0.4890066 ,         np.nan,  0.48804703,         np.nan,
         0.48569323,         np.nan,  0.48529786,         np.nan,  0.48354035,
                np.nan,  0.47520339,         np.nan,  0.45585417,         np.nan,
         0.42727529,         np.nan,  0.4208029 ,         np.nan,  0.50255279,
                np.nan,  0.79811837,         np.nan,  0.84072272,         np.nan,
         0.84494212,         np.nan,  0.84315868,         np.nan,  0.84304177,
                np.nan,  0.84469751,         np.nan,  0.84483905,         np.nan,
         0.84381388,         np.nan,  0.84429688,         np.nan,  0.8515415 ,
                np.nan],
       [ 0.92109355,         np.nan,  0.84830842,         np.nan,  0.84380773,
                np.nan,  0.84454643,         np.nan,  0.84506976,         np.nan,
         0.84341433,         np.nan,  0.84232234,         np.nan,  0.84397995,
                np.nan,  0.84418724,         np.nan,  0.83181961,         np.nan,
         0.7614702 ,         np.nan,  0.6832831 ,         np.nan,  0.64626697,
                np.nan,  0.63975145,         np.nan,  0.64439149,         np.nan,
         0.64602604,         np.nan,  0.64657823,         np.nan,  0.64387015,
                np.nan,  0.6388209 ,         np.nan,  0.63029539,         np.nan,
         0.62446247,         np.nan,  0.63119266,         np.nan,  0.63913829,
                np.nan,  0.64403418,         np.nan,  0.64668844,         np.nan,
         0.64596101,         np.nan,  0.64436713,         np.nan,  0.63965859,
                np.nan,  0.64646021,         np.nan,  0.68321781,         np.nan,
         0.7614621 ,         np.nan,  0.83182279,         np.nan,  0.84418499,
                np.nan,  0.84395892,         np.nan,  0.84233826,         np.nan,
         0.84357636,         np.nan,  0.84496889,         np.nan,  0.84418612,
                np.nan,  0.84378946,         np.nan,  0.84694881,         np.nan,
         0.8930748 ],
       [        np.nan,  0.85275386,         np.nan,  0.84390255,         np.nan,
         0.84383027,         np.nan,  0.8455248 ,         np.nan,  0.84461863,
                np.nan,  0.8420038 ,         np.nan,  0.8429615 ,         np.nan,
         0.84596905,         np.nan,  0.84256021,         np.nan,  0.82857895,
                np.nan,  0.75176504,         np.nan,  0.66157979,         np.nan,
         0.59166474,         np.nan,  0.54976059,         np.nan,  0.52887708,
                np.nan,  0.52033408,         np.nan,  0.51886616,         np.nan,
         0.52060622,         np.nan,  0.52343436,         np.nan,  0.52547644,
                np.nan,  0.52566558,         np.nan,  0.52388593,         np.nan,
         0.52108219,         np.nan,  0.51883687,         np.nan,  0.5204442 ,
                np.nan,  0.52898637,         np.nan,  0.54971153,         np.nan,
         0.5918746 ,         np.nan,  0.6616513 ,         np.nan,  0.75176809,
                np.nan,  0.82858151,         np.nan,  0.84255863,         np.nan,
         0.84596469,         np.nan,  0.84291927,         np.nan,  0.84208832,
                np.nan,  0.84485554,         np.nan,  0.84510019,         np.nan,
         0.84352072,         np.nan,  0.84423306,         np.nan,  0.84954778,
                np.nan],
       [ 0.92109212,         np.nan,  0.84830328,         np.nan,  0.84381952,
                np.nan,  0.84462667,         np.nan,  0.84503805,         np.nan,
         0.84302554,         np.nan,  0.84256247,         np.nan,  0.84474988,
                np.nan,  0.8442106 ,         np.nan,  0.83859654,         np.nan,
         0.81469792,         np.nan,  0.76782845,         np.nan,  0.72014359,
                np.nan,  0.68595089,         np.nan,  0.66819355,         np.nan,
         0.66098141,         np.nan,  0.65738603,         np.nan,  0.6555811 ,
                np.nan,  0.6525013 ,         np.nan,  0.64035865,         np.nan,
         0.63159524,         np.nan,  0.63976382,         np.nan,  0.65249249,
                np.nan,  0.65573186,         np.nan,  0.65744592,         np.nan,
         0.66114359,         np.nan,  0.66833851,         np.nan,  0.68605362,
                np.nan,  0.72027776,         np.nan,  0.76780151,         np.nan,
         0.8146885 ,         np.nan,  0.83859612,         np.nan,  0.84420404,
                np.nan,  0.84472716,         np.nan,  0.842578  ,         np.nan,
         0.84318728,         np.nan,  0.84493742,         np.nan,  0.84426642,
                np.nan,  0.84380113,         np.nan,  0.84694365,         np.nan,
         0.89307347],
       [        np.nan,  0.85275387,         np.nan,  0.84390255,         np.nan,
         0.84383025,         np.nan,  0.84552484,         np.nan,  0.84461865,
                np.nan,  0.84200345,         np.nan,  0.84296159,         np.nan,
         0.84597231,         np.nan,  0.8425622 ,         np.nan,  0.828569  ,
                np.nan,  0.75170112,         np.nan,  0.66153873,         np.nan,
         0.59168787,         np.nan,  0.54975168,         np.nan,  0.52958195,
                np.nan,  0.52037308,         np.nan,  0.51837096,         np.nan,
         0.52010849,         np.nan,  0.52320024,         np.nan,  0.52299467,
                np.nan,  0.52257555,         np.nan,  0.52219241,         np.nan,
         0.51980731,         np.nan,  0.51847934,         np.nan,  0.52052979,
                np.nan,  0.52975631,         np.nan,  0.54997733,         np.nan,
         0.59165089,         np.nan,  0.66152003,         np.nan,  0.75166194,
                np.nan,  0.82857037,         np.nan,  0.8425543 ,         np.nan,
         0.84596549,         np.nan,  0.842941  ,         np.nan,  0.84209544,
                np.nan,  0.84469018,         np.nan,  0.8450244 ,         np.nan,
         0.84383505,         np.nan,  0.84428525,         np.nan,  0.85153926,
                np.nan],
       [ 0.92109357,         np.nan,  0.84830843,         np.nan,  0.84380771,
                np.nan,  0.84454646,         np.nan,  0.84506986,         np.nan,
         0.84341399,         np.nan,  0.84232221,         np.nan,  0.84398458,
                np.nan,  0.84418983,         np.nan,  0.83179573,         np.nan,
         0.76138002,         np.nan,  0.68317673,         np.nan,  0.6461473 ,
                np.nan,  0.63945753,         np.nan,  0.64469387,         np.nan,
         0.64696183,         np.nan,  0.64683831,         np.nan,  0.64494141,
                np.nan,  0.63997494,         np.nan,  0.63067853,         np.nan,
         0.62362199,         np.nan,  0.63093404,         np.nan,  0.63983993,
                np.nan,  0.64443198,         np.nan,  0.64699689,         np.nan,
         0.64675477,         np.nan,  0.64477392,         np.nan,  0.63952028,
                np.nan,  0.64620347,         np.nan,  0.6831539 ,         np.nan,
         0.76134185,         np.nan,  0.83179554,         np.nan,  0.84418386,
                np.nan,  0.84398342,         np.nan,  0.84235956,         np.nan,
         0.8434291 ,         np.nan,  0.84477262,         np.nan,  0.84436002,
                np.nan,  0.8441444 ,         np.nan,  0.8487678 ,         np.nan,
         0.89709726],
       [        np.nan,  0.85275621,         np.nan,  0.84391419,         np.nan,
         0.84380901,         np.nan,  0.84533992,         np.nan,  0.84462584,
                np.nan,  0.84294938,         np.nan,  0.84317913,         np.nan,
         0.84495026,         np.nan,  0.84070303,         np.nan,  0.79802793,
                np.nan,  0.50242292,         np.nan,  0.42071031,         np.nan,
         0.42691679,         np.nan,  0.45614564,         np.nan,  0.47535463,
                np.nan,  0.48366239,         np.nan,  0.48583737,         np.nan,
         0.4873675 ,         np.nan,  0.48979141,         np.nan,  0.49105418,
                np.nan,  0.4908226 ,         np.nan,  0.48958378,         np.nan,
         0.48716827,         np.nan,  0.48599025,         np.nan,  0.48365934,
                np.nan,  0.47537612,         np.nan,  0.45601205,         np.nan,
         0.4269647 ,         np.nan,  0.42070475,         np.nan,  0.5024806 ,
                np.nan,  0.7980653 ,         np.nan,  0.84071407,         np.nan,
         0.84494079,         np.nan,  0.8431814 ,         np.nan,  0.8430795 ,
                np.nan,  0.84453695,         np.nan,  0.8447006 ,         np.nan,
         0.84407937,         np.nan,  0.8467536 ,         np.nan,  0.89709726,
                np.nan],
       [ 0.92109495,         np.nan,  0.84831747,         np.nan,  0.8438095 ,
                np.nan,  0.84441927,         np.nan,  0.84486165,         np.nan,
         0.8439958 ,         np.nan,  0.84372486,         np.nan,  0.8440038 ,
                np.nan,  0.84132841,         np.nan,  0.80281161,         np.nan,
         0.47373335,         np.nan,  0.51631021,         np.nan,  0.59232756,
                np.nan,  0.60827847,         np.nan,  0.61501383,         np.nan,
         0.61996272,         np.nan,  0.62315203,         np.nan,  0.62206969,
                np.nan,  0.61377923,         np.nan,  0.60004058,         np.nan,
         0.58899836,         np.nan,  0.60113514,         np.nan,  0.61377866,
                np.nan,  0.62202054,         np.nan,  0.62308986,         np.nan,
         0.62005766,         np.nan,  0.61496833,         np.nan,  0.60837715,
                np.nan,  0.59254341,         np.nan,  0.51634311,         np.nan,
         0.47380312,         np.nan,  0.8028714 ,         np.nan,  0.84133543,
                np.nan,  0.84400034,         np.nan,  0.84380177,         np.nan,
         0.84405534,         np.nan,  0.84439507,         np.nan,  0.84426582,
                np.nan,  0.84661411,         np.nan,  0.89239704,         np.nan,
         1.        ]])