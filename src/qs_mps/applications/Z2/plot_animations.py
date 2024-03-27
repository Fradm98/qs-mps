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
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values or number of trotter steps",
    type=int,
)
parser.add_argument(
    "h_i", help="Starting value of h (external transverse field on the dual lattice)", type=float
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("o", help="Observable we want to compute. Available are 'wl', 'el'", type=str)
parser.add_argument("-L", "--Ls", help="Number of rungs per ladder", nargs="+", type=int)
parser.add_argument("-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument("-cx", "--charges_x", help="a list of the first index of the charges", nargs="*", type=int)
parser.add_argument("-cy", "--charges_y", help="a list of the second index of the charges", nargs="*", type=int)
parser.add_argument(
    "-f", "--h_f", help="Final value of h (external transverse field on the dual lattice)", type=float
)
parser.add_argument(
    "-d", "--delta", help="Width of each time slice during the time evolution. Should be 'small enough'", type=float
)
parser.add_argument(
    "-ev", "--h_ev", help="Quench value of h (external transverse field on the dual lattice)", type=float
)
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
parser.add_argument(
    "-e",
    "--exact",
    help="Exact computation. By default False",
    action="store_true",
)
parser.add_argument(
    "-t",
    "--time",
    help="Animation of real time evolution of the observable. By default False",
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

# define the interval of equally spaced values of external field or trotter steps
if args.time:
    interval = range(args.npoints)
    precision = get_precision(args.h_i)
else:
    if args.interval == "lin":
        interval = np.linspace(args.h_i, args.h_f, args.npoints)
    elif args.interval == "log":
        interval = np.logspace(args.h_i, args.h_f, args.npoints)

    # define the precision
    num = (args.h_f - args.h_i) / args.npoints
    precision = get_precision(num)

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = "G:/My Drive/projects/1_Z2"
    path_state = "D:/code/projects/1_Z2"
elif args.path == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2"
    path_state = "/Users/fradm98/Desktop/projects/1_Z2"
elif args.path == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
    path_state = "/Users/fradm/Desktop/projects/1_Z2"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")




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

    for chi in args.chis:
        if args.o == 'el':
            
            if args.exact:
                path_file = f"electric_field_{args.model}_direct_lattice_{args.l-1}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}"
                dataname = f"{parent_path}/results/exact/electric_field/{path_file}.npy"
                savename = f"{parent_path}/figures/exact/animations/animation_{path_file}.mp4"
            elif args.time:
                path_file = f"electric_field_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}_chi_{chi}"
                dataname = f"{parent_path}/results/electric_field/{path_file}.npy"
                savename = f"{parent_path}/figures/animations/animation_{path_file}.mp4"
            else:
                path_file = f"electric_field_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}"
                dataname = f"{parent_path}/results/electric_field/{path_file}.npy"
                savename = f"{parent_path}/figures/animations/animation_{path_file}.mp4"
            
            data = np.load(dataname)


        movie = anim(frames=args.npoints, interval=200, data=data, params=interval, show=args.show, charges_x=args.charges_x, charges_y=args.charges_y, precision=precision, time=args.time)
        if args.save:
            movie.save(savename)