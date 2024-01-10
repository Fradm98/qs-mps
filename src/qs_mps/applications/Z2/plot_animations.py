from qs_mps.utils import *
import matplotlib.pyplot as plt
import argparse

# default parameters of the plot layout
plt.rcParams["text.usetex"] = True  # use latex
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

parser = argparse.ArgumentParser(prog="plot animations")
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
parser.add_argument("-lx","--sites", help="Number of sites in the wilson loop", type=int)
parser.add_argument("-ly","--ladders", help="Number of ladders in the wilson loop", type=int)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
)
parser.add_argument(
    "-U", "--gauss", help="Gauss constraint parameter", default=1e+3, type=float
)
parser.add_argument(
    "-sh",
    "--show",
    help="Show the animation. By default False",
    action="store_true",
)
parser.add_argument(
    "-v",
    "--save",
    help="Save the animation. By default True",
    action="store_false",
)

args = parser.parse_args()

# define the interval of equally spaced values of external field
interval = np.linspace(args.h_i, args.h_f, args.npoints)

# define the precision
num = (args.h_f - args.h_i) / args.npoints
precision = get_precision(num)

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


# define the sector by looking of the given charges
if len(args.charges_x) == 0:
    sector = "vacuum_sector"
else:
    for i in range(1,args.l*args.L):
        if len(args.charges_x) == i:
            sector = f"{i}_particle(s)_sector"

if args.o == 'el':
    path_file = f"electric_field_{args.model}_direct_lattice_{args.l-1}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}"
    data = np.load(
        f"{parent_path}/results/exact/electric_field/{path_file}.npy")


movie = anim(frames=args.npoints, interval=200, data=data, params=interval, show=args.show, charges_x=args.charges_x, charges_y=args.charges_y, precision=precision)
if args.save:
    movie.save(f"{parent_path}/figures/animations/animation_{path_file}.mp4")