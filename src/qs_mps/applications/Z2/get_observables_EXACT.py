# import packages
import argparse
from qs_mps.applications.Z2.exact_hamiltonian import H_Z2_gauss
from qs_mps.utils import *
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(prog="observables_Z2_exact")
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
parser.add_argument("o", help="Observable we want to compute. Available are 'wl', 'el'", type=str)
parser.add_argument("-cx", "--charges_x", help="a list of the first index of the charges", nargs="*", type=int)
parser.add_argument("-cy", "--charges_y", help="a list of the second index of the charges", nargs="*", type=int)
parser.add_argument("-s","--sites", help="Indices of sites in the wilson loop. Start from 0, left", nargs="*", type=int)
parser.add_argument("-v","--ladders", help="Indices of ladders in the wilson loop. Start from 1, above", nargs="*", type=int)
parser.add_argument(
    "-m", "--model", help="Model to simulate. By default Z2_dual", default="Z2_dual", type=str
)
parser.add_argument(
    "-U", "--gauss", help="Gauss constraint parameter. By default 1e+3", default=1e+3, type=float
)

args = parser.parse_args()

# define the interval of equally spaced values of external field
interval = np.linspace(args.h_i, args.h_f, args.npoints)

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/1_Z2"
    path_eigvec = "D:/code/projects/1_Z2"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2"
    path_eigvec = "/Users/fradm98/Desktop/projects/1_Z2"
elif args.path == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
    path_eigvec = "/Users/fradm/Desktop/projects/1_Z2"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")

num = (args.h_f - args.h_i) / args.npoints
precision = get_precision(num)


# define the sector by looking of the given charges
if len(args.charges_x) == 0:
    sector = "vacuum_sector"
else:
    for i in range(1,args.l*args.L):
        if len(args.charges_x) == i:
            sector = f"{i}_particle(s)_sector"
# ---------------------------------------------------------
# Wilson Loop
# ---------------------------------------------------------

W = []
E = []
E_sum = []

for h in interval:

    Z2_exact = H_Z2_gauss(L=args.L, l=args.l, model=args.model, lamb=h, U=1e+3)
    # print(Z2_exact.latt._lattice_drawer.draw_lattice())
    psi = np.load(f"{path_eigvec}/results/eigenvectors/ground_state_direct_lattice_{args.l-1}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_U_{args.gauss}_h_{h:.{precision}f}.npy")

    if args.o == "wl":
        W.append(Z2_exact.wilson_loop(psi, args.sites, args.ladders))
    
    if args.o == "el":
        print(f"electric field for h:{h:.{precision}f}")
        E_h = np.zeros((2*args.l-1,2*args.L-1))
        E_h[:] = np.nan
        E_h = Z2_exact.electric_field(psi, E_h)
        E.append(E_h)

        if sector != "vacuum_sector":
            if args.charges_x[0] == args.charges_x[1]:
                # vertical charges
                sum_el = sum(E_h[args.charges_y[0]*2+1:args.charges_y[1]*2, args.charges_x[0]*2])
            elif args.charges_y[0] == args.charges_y[1]:
                # horizontal charges
                sum_el = sum(E_h[args.charges_y[0]*2,args.charges_x[0]*2+1, args.charges_x[1]*2])
            E_sum.append(sum_el)

if args.o == "wl":
    np.savetxt(
                f"{parent_path}/results/exact/wilson_loops/wilson_loop_{args.model}_direct_lattice_{args.l-1}x{args.L-1}_{sector}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}",
                W,
            )
if args.o == "el":
    np.save(
                f"{parent_path}/results/exact/electric_field/electric_field_{args.model}_direct_lattice_{args.l-1}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}.npy",
                E,
            )
    if sector != "vacuum_sector":
        np.save(
                f"{parent_path}/results/exact/electric_field/sum_of_electric_field_{args.model}_direct_lattice_{args.l-1}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}.npy",
                E_sum,
            )
    
    