# import packages
import argparse
from qs_mps.mps_class import MPS
from qs_mps.utils import *

parser = argparse.ArgumentParser(prog="observables_Z2_mps")
parser.add_argument("l", help="Number of ladders in the direct lattice", type=int)
parser.add_argument("L", help="Number of rungs per ladder", type=int)
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "h_i", help="Starting value of h (external transverse field on the dual lattice)", type=float
)
parser.add_argument(
    "h_f", help="Final value of h (external transverse field on the dual lattice)", type=float
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("o", help="Observable we want to compute. Available are 'wl', 'el', 'thooft'", type=str)
parser.add_argument("chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument("-cx", "--charges_x", help="a list of the first index of the charges", nargs="*", type=int)
parser.add_argument("-cy", "--charges_y", help="a list of the second index of the charges", nargs="*", type=int)
parser.add_argument("-s", "--sites", help="Number of sites in the wilson loop", nargs="*", type=int)
parser.add_argument("-r", "--ladders", help="Number of ladders in the wilson loop", nargs="*", type=int)
parser.add_argument(
    "-d", "--direction", help="Direction of the string", default="hor", type=str
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
)
parser.add_argument(
    "-mo", "--moment", help="Moment degree of the Free energy. E.g. Magnetization -> First Moment, Susceptibility -> Second Moment, etc. Available are 1,2,4", default=1, type=int
)

args = parser.parse_args()

# define the physical dimension
d = int(2**(args.l))

# define the interval of equally spaced values of external field
interval = np.linspace(args.h_i, args.h_f, args.npoints)

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

num = (args.h_f - args.h_i) / args.npoints
precision = get_precision(num)

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

# define the sector by looking of the given charges
if len(args.charges_x) == 0:
    sector = "vacuum_sector"
    args.charges_x = None
    args.charges_y = None
else:
    for i in range(1,args.l*args.L):
        if len(args.charges_x) == i:
            sector = f"{i}_particle(s)_sector"

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
    W = []
    E = []
    S = []
    M = []
    for h in interval:
        lattice_mps = MPS(L=args.L, d=d, model=args.model, chi=chi, h=h)
        lattice_mps.L = lattice_mps.L - 1

        lattice_mps.load_sites(path=path_tensor, precision=precision, cx=args.charges_x, cy=args.charges_y)
        if sector != "vacuum_sector":
            lattice_mps.Z2.add_charges(args.charges_x, args.charges_y)
        
        if args.o == "wl":
            print(f"wilson loop for h:{h:.{precision}f}")
            lattice_mps.Z2.wilson_Z2_dual(mpo_sites=[sites], ls=[ladders]) #list(range(s))
            lattice_mps.w = lattice_mps.Z2.mpo.copy()
            W.append(lattice_mps.mpo_first_moment().real)

        elif args.o == "el":
            print(f"electric field for h:{h:.{precision}f}")
            E_h = np.zeros((2*args.l+1,2*args.L-1))
            E_h[:] = np.nan
            E_h = lattice_mps.electric_field_Z2(E_h)
            E.append(E_h)
        
        elif args.o == "thooft":
            print(f"'t Hooft string for h:{h:.{precision}f}")
            lattice_mps.Z2.thooft(site=args.sites, l=args.ladders, direction=direction)
            lattice_mps.w = lattice_mps.Z2.mpo.copy()
            S.append(lattice_mps.mpo_first_moment().real)

        elif args.o == "mag":
            print(f"Magnetization for h:{h:.{precision}f}")
            lattice_mps.order_param()
            if args.moment == 1:
                M.append(lattice_mps.mpo_first_moment().real/(len(lattice_mps.Z2.latt.plaquettes()) - 2*(args.L-1)-2*(args.l-2)))
            elif args.moment == 2:
                M.append(lattice_mps.mpo_second_moment().real/((len(lattice_mps.Z2.latt.plaquettes()) - 2*(args.L-1)-2*(args.l-2))**2))
            elif args.moment == 4:
                M.append(lattice_mps.mpo_fourth_moment().real/((len(lattice_mps.Z2.latt.plaquettes()) - 2*(args.L-1)-2*(args.l-2))**4))


    if args.o == "wl":
        np.savetxt(
                    f"{parent_path}/results/wilson_loops/wilson_loop_{args.model}_direct_lattice_{args.l}x{args.L-1}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                    W,
                )
    if args.o == "el":
        np.save(
                    f"{parent_path}/results/electric_field/electric_field_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                    E,
                )
    if args.o == "thooft":
        np.save(
                    f"{parent_path}/results/thooft/thooft_string_{moment}_moment_{args.sites[0]}-{args.ladders[0]}_{direction}_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                    S,
                )
    if args.o == "mag":
        np.save(
                    f"{parent_path}/results/mag_data/dual_mag_{moment}_moment_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                    M,
                )