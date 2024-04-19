import numpy as np
import argparse
from qs_mps.sparse_hamiltonians_and_operators import exact_evolution_sparse
from qs_mps.utils import get_precision, save_list_of_lists, access_txt
from qs_mps.applications.Z2.ground_state_multiprocessing import ground_state_Z2_exact, ground_state_Z2_exact_test
from qs_mps.applications.Z2.exact_hamiltonian import H_Z2_gauss

# EXACT DIAGONALIZATION to find ground states of the Z2 Pure Gauge Theory 
# changing the plaquette (magnetic) parameters in its direct formulation

parser = argparse.ArgumentParser(prog="gs_search_Z2_exact")
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
parser.add_argument("-cx", "--charges_x", help="a list of the first index of the charges", nargs="*", type=int)
parser.add_argument("-cy", "--charges_y", help="a list of the second index of the charges", nargs="*", type=int)
parser.add_argument(
    "-m", "--model", help="Model to simulate. By default Z2_dual", default="Z2_dual", type=str
)
parser.add_argument(
    "-v",
    "--save",
    help="Save the ground states. By default True",
    action="store_false",
)
parser.add_argument(
    "-p",
    "--sparse",
    help="Use Sparse method for exact diagonalization. By default True",
    action="store_false",
)
parser.add_argument(
    "-w",
    "--spectrum",
    help="The spectrum of the Exact Hamiltonian. By default (-1) computes only the ground state eigenvalue. Can compute the spectrum up to the degrees of freedom of the dual lattice",
    default=-1,
    type=int,
)
parser.add_argument(
    "-U", "--gauss", help="Gauss constraint parameter. By default 1e+3", default=1e+3, type=float
)

args = parser.parse_args()

# define the interval
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

# define the precision
num = (args.h_f - args.h_i) / args.npoints
precision = get_precision(num)

# define the initial guess state
dof_direct = (2*args.l*args.L - args.l - args.L)
# constant without sparse method (if we start from h=0)
c = (1/np.sqrt(2))**dof_direct

print("finding guess for the sparse computation...")
v0 = np.full(2**dof_direct, c)
v0 = np.array(v0, dtype=complex)
print(v0)

# choose how many eigenvalues to compute
if args.spectrum == -1:
    spectrum = "gs"
elif args.spectrum == 0:
    spectrum = "all"

    
# define the sector by looking of the given charges
if len(args.charges_x) == 0:
    sector = "vacuum_sector"
    args.charges_x = None
    args.charges_y = None
else:
    for i in range(1,args.l*args.L):
        if len(args.charges_x) == i:
            sector = f"{i}_particle(s)_sector"

# ---------------------------------------------------------
# Exact Diagonalization
# ---------------------------------------------------------
args_lattice = {
    "L": args.L,
    "l": args.l,
    "model": args.model,
    "path": path_eigvec,
    "save": args.save,
    # "v0": v0,
    "sparse": args.sparse,
    "precision": precision,
    "spectrum": spectrum,
    "U": args.gauss,
    "sector": sector,
    "charges_x": args.charges_x,
    "charges_y": args.charges_y,
}

energy = []

for h in interval:
    print(f"computing ground state for param: {h:.{precision}f}")
    Z2 = H_Z2_gauss(l=args.l, L=args.L, model=args.model, lamb=h, U=args.gauss)
    if len(args.charges_x) > 0:
        Z2.add_charges(rows=args.charges_x, columns=args.charges_y)
        Z2._define_sector()
    e, v = Z2.diagonalize(v0=v0, sparse=args.sparse, save=args.save, path=path_eigvec, cx=args.charges_x, cy=args.charges_y, precision=precision) # , path=args.path, precision=precision, spectrum=spectrum, cx=args.charges_x, cy=args.charges_y
    energy.append(e[0])
    v0 = v[:,0]
    print("groud state:")
    print(v0)

# energy = ground_state_Z2_exact_test(
#     args_lattice=args_lattice, param=interval
# )

if spectrum == "all":
    save_list_of_lists(
        f"{parent_path}/results/exact/energy_data/energies_{args.model}_direct_lattice_{args.l-1}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}",
        energy,
    )
    energy_gs = access_txt(
            f"{parent_path}/results/exact/energy_data/energies_{args.model}_direct_lattice_{args.l-1}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}",
            0,
        )
    np.savetxt(f"{parent_path}/results/exact/energy_data/energies_{args.model}_direct_lattice_{args.l-1}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}",
        energy_gs,
    )
else:
    np.savetxt(f"{parent_path}/results/exact/energy_data/energies_{args.model}_direct_lattice_{args.l-1}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}",
        energy,
    )