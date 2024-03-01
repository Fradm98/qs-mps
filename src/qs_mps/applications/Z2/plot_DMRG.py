import matplotlib.pyplot as plt
import numpy as np
from qs_mps.utils import plot_results_DMRG, plot_colormaps_evolution, get_precision
import argparse

# default parameters of the plot layout
plt.rcParams["text.usetex"] = True  # use latex
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True


parser = argparse.ArgumentParser(prog="Plot Ground States")
parser.add_argument("l", help="Number of ladders in the direct lattice", type=int)
parser.add_argument("L", help="Number of rungs per ladder", type=int)
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "h_i", help="Starting value of h (external transverse field)", type=float
)
parser.add_argument(
    "h_f", help="Final value of h (external transverse field)", type=float
)
parser.add_argument(
    "loc",
    help="From which computer you want to access the drive. Useful for the path spec. Available are: 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument(
    "what",
    help="Results we want to plot. Available are: 'energy', 'entropy', 'wilson_loop', 'thooft', 'energy_tr', 'entropy_tot'",
    type=str,
)
parser.add_argument("chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument("-cx", "--charges_x", help="a list of the first index of the charges", nargs="*", type=int)
parser.add_argument("-cy", "--charges_y", help="a list of the second index of the charges", nargs="*", type=int)
parser.add_argument("-s","--sites", help="Indices of sites in the wilson loop. Start from 0, left", nargs="*", type=int)
parser.add_argument("-r","--ladders", help="Indices of ladders in the wilson loop. Start from 1, above", nargs="*", type=int)
parser.add_argument(
    "-dir", "--direction", help="Direction of the string", default="hor", type=str
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
)
parser.add_argument(
    "-mo", "--moment", help="Moment of the order parameter. Available are 1,2, and 4. By default 1", default=1, type=int
)
parser.add_argument(
    "-t", "--time", help="Final time of the evolution", default=10, type=float
)
parser.add_argument(
    "-v",
    "--save",
    help="Save plot, by default True",
    action="store_false",
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
    "-k", "--marker", help="Marker to use for the Plot", default="+", type=str
)
parser.add_argument("-z", "--m_size", help="marker size", default=20, type=int)
parser.add_argument(
    "-li",
    "--linewidth",
    help="Linewidth of the marker",
    default=1,
    type=float,
)
parser.add_argument(
    "-a",
    "--alpha",
    help="Alpha of the marker",
    default=1,
    type=float,
)
parser.add_argument(
    "-n",
    "--n_points",
    help="Percentage of number of points we want to plot, from 0 to 1",
    default=1,
    type=float,
)
parser.add_argument(
    "-c", "--cmap", help="colormap for the plots", default="viridis", type=str
)
parser.add_argument(
    "-i",
    "--interpolation",
    help="interpolation for the colormap plots",
    default="antialiased",
    type=str,
)
parser.add_argument(
    "-d",
    "--dim",
    help="Dimension to plot, by defalut plots two dimensional plots. If True 3D plots, by default False",
    action="store_true",
)
parser.add_argument(
    "-e",
    "--exact",
    help="Exact results to compare the mps ones. If True we compare - be default False",
    action="store_true",
)

args = parser.parse_args()
if args.where == -1:
    args.where = args.L // 2
elif args.where == -2:
    args.bond = False
    args.where = "all"

# define interval and precision
interval = np.linspace(args.h_i, args.h_f, args.npoints).tolist()
num = (args.h_f - args.h_i) / args.npoints
precision = get_precision(num)

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

# define the moment
if args.moment == 1:
    moment = "first"
elif args.moment == 2:
    moment = "second"
elif args.moment == 4:
    moment = "fourth"


plot_val = [
    "energy",
    "entropy",
    "wilson_loop",
    "thooft"
]
plot_cmap = ["energy_tr", "entropy_tot"]

if args.loc == "pc":
    path_computer = "G:/My Drive/projects/1_Z2/"
elif args.loc == "mac":
    path_computer = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2/"
elif args.loc == "marcos":
    path_computer = "/Users/fradm/Google Drive/My Drive/projects/1_Z2/"
else:
    raise SyntaxError("insert a valid location: 'pc', 'mac', 'marcos'")


if args.what == "energy":
    title = f"Energy: lattice = ${args.l}$x${args.L-1}$ ; vacuum sector"
    fname_what = "energies"
    fname_ex_what = "energies"
    path = path_computer + f"results/energy_data"
    path_ex = path_computer + f"results/exact/energy_data"
    path_save = path_computer + f"figures/energy/"
    ylabel = "$\\langle E\\rangle$"


elif args.what == "entropy":
    title = (
        f" ${args.where}-th$ Bond Entanglement Entropy: lattice = ${args.l}$x${args.L-1}$ ; {sector}"
        # + f" $h \in ({args.h_i},{args.h_f})$"
    )
    fname_what = f"{args.where}_bond_entropy"
    fname_ex_what = f"{args.where}_bond_exact_entropy"
    path = path_computer + f"results/entropy_data"
    path_ex = path_computer + f"results/exact/entropy_data"
    path_save = path_computer + f"figures/entropy/"
    ylabel = "entanglement von neumann entropy $(S_{\chi})$"

elif args.what == "wilson_loop":
    title = f"Wilson Loop: lattice = ${args.l}$x${args.L-1}$"
    fname_what = "wilson_loop"
    fname_ex_what = "wilson_loop"
    path = path_computer + f"results/wilson_loops"
    path_ex = path_computer + f"results/exact/wilson_loops"
    path_save = path_computer + f"figures/wilson_loops/"
    ylabel = "$\\langle W\\rangle$"

elif args.what == "thooft":
    title = f"'t Hooft {direction} string: lattice = ${args.l}$x${args.L-1}$ ;" + " $\mu^z$ for " + f"L={args.sites[0]}, l={args.ladders[0]}" + " dual"
    fname_what = f"thooft_string_{moment}_moment_{args.sites[0]}-{args.ladders[0]}_{direction}"
    fname_ex_what = f"thooft_string_{moment}_moment_{args.sites[0]}-{args.ladders[0]}_{direction}"
    path = path_computer + f"results/thooft"
    path_ex = path_computer + f"results/exact/thooft"
    path_save = path_computer + f"figures/thooft/"
    ylabel = "$\\langle \sigma^x \sigma^x \\rangle$"

elif args.what == "entropy_tot":
    title = (
        f"All Bonds Entanglement Entropy: $lattice = {args.l}x{args.L}$ ; $gap = {args.npoints}$ ;"
        + f" $h \in ({args.h_i},{args.h_f})$"
    )
    fname_what = f"{args.where}_bond_entropy"
    path = path_computer + f"results/entropy_data"
    path_save = path_computer + f"figures/entropy/"
    xlabel = "bonds"
    x_ticks = range(args.L - 1)
    labels = range(1, args.L - 1 + 1)
    steps = len(x_ticks) // 5
    xticks = x_ticks[::steps]
    steps = len(labels) // 5
    xlabels = labels[::steps]
    y_ticks = np.arange(args.trotter_steps + 1)
    # labels = delta * y_ticks
    steps = len(y_ticks) // 5
    yticks = y_ticks[::steps]
    steps = len(labels) // 5
    ylabels = labels[::steps]
    X, Y = np.meshgrid(x_ticks, y_ticks)
    view_init = False

elif args.what == "e_string":
    title = f"Sum of the Electric String: lattice = ${args.l}$x${args.L-1}$ ; charges {args.charges_x}-{args.charges_y}"
    fname_what = "sum_of_electric_field"
    fname_ex_what = "sum_of_electric_field"
    path = path_computer + f"results/electric_field"
    path_ex = path_computer + f"results/exact/electric_field"
    path_save = path_computer + f"figures/electric_field/"
    ylabel = "$\\sum_i^R\\sigma_i^x$"

elif args.what == "mag_loc":
    title = (
        f"Local Magnetization Evolution: $L = {args.L}$ ;"
        + " $h_{ev}=$ "
        + f"${args.h_ev}$"
    )
    fname_what = f"mag_mps_loc"
    fname_ex_what = "mag_exact_loc"
    path = path_computer + f"results/mag_data"
    path_ex = path_computer + f"results/exact/mag_data"
    path_save = path_computer + f"figures/magnetization/"
    xlabel = "sites"
    x_ticks = range(args.L)
    labels = range(1, args.L + 1)
    steps = len(x_ticks) // 5
    xticks = x_ticks[::steps]
    steps = len(labels) // 5
    xlabels = labels[::steps]
    y_ticks = np.arange(args.trotter_steps + 1)
    # labels = delta * y_ticks
    steps = len(y_ticks) // 5
    yticks = y_ticks[::steps]
    steps = len(labels) // 5
    ylabels = labels[::steps]
    X, Y = np.meshgrid(x_ticks, y_ticks)
    view_init = True

else:
    raise SyntaxError(
        "insert a valid result to plot: 'energy', 'entropy', 'wilson_loop', 'energy_tr', 'entropy_tot', 'e_string'"
    )

if args.what in plot_val:
    fname = f"{fname_what}_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}"
    fname_ex = f"{fname_ex_what}_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}"
    fname_save = f"{args.what}_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}"
    plot_results_DMRG(
        title,
        for_array=args.chis,
        interval=interval,
        fname=fname,
        path=path,
        fname_save=fname_save,
        path_save=path_save,
        ylabel=ylabel,
        save=args.save,
        exact=args.exact,
        fname_ex=fname_ex,
        path_ex=path_ex,
        marker=args.marker,
        m_size=args.m_size,
        linewidth=args.linewidth,
        alpha=args.alpha,
        n_points=args.n_points,
        cmap=args.cmap,
        precision=precision
    )

if args.what in plot_cmap:
    for chi in args.chis:
        title_fin = title + f" ; $\chi = {chi}$"
        fname = f"{fname_what}_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}"
        fname_save = f"{args.what}_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}"
        plot_colormaps_evolution(
            title=title_fin,
            fname=fname,
            path=path,
            fname_save=fname_save,
            path_save=path_save,
            xlabel=xlabel,
            xticks=xticks,
            xlabels=xlabels,
            yticks=yticks,
            ylabels=ylabels,
            X=X,
            Y=Y,
            save=args.save,
            cmap=args.cmap,
            interpolation=args.interpolation,
            d=args.dim,
            view_init=view_init,
        )
