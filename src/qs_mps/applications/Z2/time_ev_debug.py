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

parser = argparse.ArgumentParser(prog="Time Ev")
parser.add_argument("l", help="Number of ladders in the direct lattice", type=int)
parser.add_argument(
    "npoints",
    help="Trotter steps for the quench dynamics",
    type=int,
)
parser.add_argument(
    "delta", help="Width of each time slice during the time evolution. Should be 'small enough'", type=float
)
parser.add_argument(
    "h_i", help="Starting value of h (external transverse field on the dual lattice)", type=float
)
parser.add_argument(
    "h_ev", help="Quench value of h (external transverse field on the dual lattice)", type=float
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("-L", "--Ls", help="Number of rungs per ladder", nargs="+", type=int)
parser.add_argument("-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument("-o", "--obs", help="Observable we want to compute. Available are 'wl', 'el', 'thooft', 'mag', 'eed'", nargs="*", type=str)
parser.add_argument("-cx", "--charges_x", help="a list of the first index of the charges", nargs="*", type=int)
parser.add_argument("-cy", "--charges_y", help="a list of the second index of the charges", nargs="*", type=int)
parser.add_argument("-lx", "--sites", help="Number of sites in the wilson loop", nargs="*", type=int)
parser.add_argument("-ly", "--ladders", help="Number of ladders in the wilson loop", nargs="*", type=int)
parser.add_argument(
    "-d", "--direction", help="Direction of the string", default="hor", type=str
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
    "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
)
parser.add_argument(
    "-mo", "--moment", help="Moment degree of the Free energy. E.g. Magnetization -> First Moment, Susceptibility -> Second Moment, etc. Available are 1,2,4", default=1, type=int
)
parser.add_argument(
    "-tr",
    "--training",
    help="Save all the errors during the variational compression. By default False",
    action="store_true",
)

args = parser.parse_args()

# define the physical dimension
d = int(2**(args.l))

# define the precision to load the mps
# precision = get_precision(args.h_i)
precision = 2

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/1_Z2"
    path_tensor = "D:/code/projects/1_Z2"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/Users/fradm98/Desktop/projects/1_Z2"
elif args.path == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/Users/fradm/Desktop/projects/1_Z2"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")


# for the wilson loop
if args.sites == 1:
    sites = 0
if args.ladders == 1:
    ladders = 1

# define the direction
if args.direction == "ver":
    direction = "vertical"
elif args.direction == "hor":
    direction = "horizontal"    

# define moment
if args.moment == 1:
    moment = "first"
if args.moment == 2:
    moment = "second"
if args.moment == 4:
    moment = "fourth"

# define all observables
if args.obs == []:
    args.obs = ["wl","el","thooft","mag"]


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

    # where to look at for the entropy
    if args.where == -1:
        args.where = L // 2
    elif args.where == -2:
        args.bond = False

    for chi in args.chis:
        W = []
        El = []
        S = []
        M = []
        C = []
        Ed = []

        # initialize the tensor
        lattice_mps = MPS(L=L, d=d, model=args.model, chi=chi, h=args.h_i)
        lattice_mps.L = lattice_mps.L - 1
        
        # load the mps
        lattice_mps.load_sites(path=path_tensor, precision=precision, cx=charges_x, cy=charges_y)
        if sector != "vacuum_sector":
            lattice_mps.Z2.add_charges(charges_x, charges_y)
        
        # initialize the variables to save
        errors_tr = [[0, 0]]
        errors = [0]
        entropies = [[0]]
        schmidt_vals = np.array([0]*chi)
        schmidt_vals[0] = 1
        schmidt_vals = [schmidt_vals]
        # ---------------------------------------------------------
        # Initial Observables
        # ---------------------------------------------------------
        if "wl" in args.obs:
            print(f"wilson loop before trotter - L:{L}, chi:{chi}")
            lattice_mps.Z2.wilson_Z2_dual(mpo_sites=args.sites, ls=args.ladders) #list(range(s))
            lattice_mps.w = lattice_mps.Z2.mpo.copy()
            if args.moment == 1:
                W.append(lattice_mps.mpo_first_moment().real)
            elif args.moment == 2:
                W.append(lattice_mps.mpo_second_moment().real)
            elif args.moment == 4:
                W.append(lattice_mps.mpo_fourth_moment().real)

        if  "el" in args.obs:
            print(f"electric field before trotter - L:{L}, chi:{chi}")
            E_h = np.zeros((2*args.l+1,2*L-1))
            E_h[:] = np.nan
            E_h = lattice_mps.electric_field_Z2(E_h)
            El.append(E_h)
        
        if "thooft" in args.obs:
            print(f"'t Hooft string before trotter - L:{L}, chi:{chi}")
            lattice_mps.Z2.thooft(site=args.sites, l=args.ladders, direction=direction)
            lattice_mps.w = lattice_mps.Z2.mpo.copy()
            S.append(lattice_mps.mpo_first_moment().real)

        if "mag" in args.obs:
            print(f"Magnetization before trotter - L:{L}, chi:{chi}")
            lattice_mps.order_param()
            if args.moment == 1:
                M.append(lattice_mps.mpo_first_moment().real/(len(lattice_mps.Z2.latt.plaquettes())-(2*(L-3)+2*(args.l))))
            elif args.moment == 2:
                M.append(lattice_mps.mpo_second_moment().real/(len(lattice_mps.Z2.latt.plaquettes())-(2*(L-3)+2*(args.l)))**2)
            elif args.moment == 4:
                M.append(lattice_mps.mpo_fourth_moment().real/(len(lattice_mps.Z2.latt.plaquettes())-(2*(L-3)+2*(args.l)))**4)

        if "corr" in args.obs:
            s = (charges_x[0]+charges_x[1]) //2
            lad = np.min(charges_y)
            print(f"connected correlator before trotter - L:{L}, chi:{chi}")
            corr = lattice_mps.connected_correlator(site=s, lad=lad)
            C.append(corr)
        
        if "eed" in args.obs:
            s = (charges_x[0]+charges_x[1]) //2
            print(f"electric energy density before trotter - L:{L}, chi:{chi}")
            ed = lattice_mps.electric_energy_density_Z2(site=s)
            Ed.append(ed)

        # ---------------------------------------------------------
        # Trotter Evolution
        # ---------------------------------------------------------        
        lattice_mps.enlarge_chi()
        lattice_mps.ancilla_sites = lattice_mps.sites.copy()
        for t in range(args.npoints):
            
            lattice_mps, error, entropy, schmidt_values = lattice_mps.TEBD_variational_Z2_trotter_step(trotter_step=t, delta=args.delta, h_ev=args.h_ev, where=args.where, bond=args.bond)
            
            # ---------------------------------------------------------
            # Trotter Observables
            # ---------------------------------------------------------  
            if "wl" in args.obs:
                print(f"wilson loop for trotter step:{t}, L:{L}, chi:{chi}")
                lattice_mps.Z2.wilson_Z2_dual(mpo_sites=args.sites, ls=args.ladders) #list(range(s))
                lattice_mps.w = lattice_mps.Z2.mpo.copy()
                if args.moment == 1:
                    W.append(lattice_mps.mpo_first_moment().real)
                elif args.moment == 2:
                    W.append(lattice_mps.mpo_second_moment().real)
                elif args.moment == 4:
                    W.append(lattice_mps.mpo_fourth_moment().real)

            if  "el" in args.obs:
                print(f"electric field for trotter step:{t}, L:{L}, chi:{chi}")
                E_h = np.zeros((2*args.l+1,2*L-1))
                E_h[:] = np.nan
                E_h = lattice_mps.electric_field_Z2(E_h)
                El.append(E_h)
            
            if "thooft" in args.obs:
                print(f"'t Hooft string for trotter step:{t}, L:{L}, chi:{chi}")
                lattice_mps.Z2.thooft(site=args.sites, l=args.ladders, direction=direction)
                lattice_mps.w = lattice_mps.Z2.mpo.copy()
                S.append(lattice_mps.mpo_first_moment().real)

            if "mag" in args.obs:
                print(f"Magnetization for trotter step:{t}, L:{L}, chi:{chi}")
                lattice_mps.order_param()
                if args.moment == 1:
                    M.append(lattice_mps.mpo_first_moment().real/(len(lattice_mps.Z2.latt.plaquettes())-(2*(L-3)+2*(args.l))))
                elif args.moment == 2:
                    M.append(lattice_mps.mpo_second_moment().real/(len(lattice_mps.Z2.latt.plaquettes())-(2*(L-3)+2*(args.l)))**2)
                elif args.moment == 4:
                    M.append(lattice_mps.mpo_fourth_moment().real/(len(lattice_mps.Z2.latt.plaquettes())-(2*(L-3)+2*(args.l)))**4)

            if "corr" in args.obs:
                print(f"connected correlator for trotter step:{t}, L:{L}, chi:{chi}")
                corr = lattice_mps.connected_correlator(site=s, lad=lad)
                C.append(corr)
            
            if "eed" in args.obs:
                s = (charges_x[0]+charges_x[1]) //2
                print(f"electric energy density before trotter - L:{L}, chi:{chi}")
                ed = lattice_mps.electric_energy_density_Z2(site=s)
                Ed.append(ed)

            print(f"\nError at trotter step: {t} is: {error}\n")
            errors_tr.append(error)
            errors.append(error[-1])
            entropies.append(entropy)
            schmidt_vals.append(schmidt_values)

        if "wl" in args.obs:
            np.save(
                        f"{parent_path}/results/wilson_loops/wilson_loop_{moment}_moment_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                        W,
                    )
        if "el" in args.obs:
            np.save(
                        f"{parent_path}/results/electric_field/electric_field_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                        El,
                    )
        if "thooft" in args.obs:
            np.save(
                        f"{parent_path}/results/thooft/thooft_string_{moment}_moment_{args.sites[0]}-{args.ladders[0]}_{direction}_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                        S,
                    )
        if "mag" in args.obs:
            np.save(
                        f"{parent_path}/results/mag_data/dual_mag_{moment}_moment_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                        M,
                    )
        if "corr" in args.obs:
            np.save(
                        f"{parent_path}/results/mag_data/connected_correlator_s_{s}_l_{lad}_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                        C,
                    )
        if "eed" in args.obs:
            np.save(
                        f"{parent_path}/results/electric_field/electric_energy_density_s_{s}_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}.npy",
                        Ed,
                    )
            
        if args.bond == False:
            args.where = "all"

        if args.training:
            save_list_of_lists(
                f"{parent_path}/results/error_data/errors_tr_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{charges_x}-{charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}",
                errors_tr,
            )
            np.savetxt(
                f"{parent_path}/results/error_data/errors_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{charges_x}-{charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}",
                errors,
            )
        else:
            np.savetxt(
                f"{parent_path}/results/error_data/errors_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{charges_x}-{charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}",
                errors,
            )
            
        save_list_of_lists(
            f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{charges_x}-{charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}",
            entropies,
        )
        save_list_of_lists(
            f"{parent_path}/results/entropy_data/{args.where}_schmidt_vals_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{charges_x}-{charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}",
            schmidt_vals,
        )
        if args.where == "all":
            entropy_mid = access_txt(
                f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{charges_x}-{charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}",
                (L-1) // 2,
            )
            np.savetxt(
                f"{parent_path}/results/entropy_data/{L // 2}_bond_entropy_quench_dynamics_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{charges_x}-{charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}",
                entropy_mid,
            )
