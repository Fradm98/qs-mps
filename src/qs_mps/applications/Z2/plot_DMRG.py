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
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values or trotter steps",
    type=int,
)
parser.add_argument(
    "h_i", help="Starting value of h (external transverse field)", type=float
)
parser.add_argument(
    "loc",
    help="From which computer you want to access the drive. Useful for the path spec. Available are: 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument(
    "obs",
    help="Results we want to plot. Available are: 'energy', 'error', 'entropy', 'wilson_loop', 'thooft', 'mag', 'energy_tr', 'error_tr', 'entropy_tot'",
    type=str,
)
parser.add_argument("-L", "--Ls", help="Number of rungs per ladder", nargs="+", type=int)
parser.add_argument("-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument("-cx", "--charges_x", help="a list of the first index of the charges", nargs="*", type=int)
parser.add_argument("-cy", "--charges_y", help="a list of the second index of the charges", nargs="*", type=int)
parser.add_argument("-lx", "--sites", help="Number of sites in the wilson loop", nargs="*", type=int)
parser.add_argument("-ly", "--ladders", help="Number of ladders in the wilson loop", nargs="*", type=int)
parser.add_argument(
    "-f", "--h_f", help="Final value of h (external transverse field on the dual lattice)", type=float
)
parser.add_argument(
    "-del", "--delta", help="Width of each time slice during the time evolution. Should be 'small enough'", type=float
)
parser.add_argument(
    "-ev", "--h_ev", help="Quench value of h (external transverse field on the dual lattice)", type=float
)

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
    "-t", "--time", help="Plot observables of time evolution", action="store_true"
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
    "-int",
    "--interval",
    help="Type of interval spacing. Available are 'log', 'lin'",
    default="lin",
    type=str
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

# define the interval of equally spaced values of external field or trotter steps
if args.time:
    interval = range(args.npoints+1)
    precision = get_precision(args.h_i)
    if args.interval == "lin":
        xscale = "linear"
    elif args.interval == "log":
        xscale = "log"
else:
    if args.interval == "lin":
        interval = np.linspace(args.h_i, args.h_f, args.npoints)
        xscale = "linear"
    elif args.interval == "log":
        interval = np.logspace(args.h_i, args.h_f, args.npoints)
        xscale = "log"

    # define the precision
    num = (args.h_f - args.h_i) / args.npoints
    precision = get_precision(num)

# define the direction
if args.direction == "ver":
    direction = "vertical"
elif args.direction == "hor":
    direction = "horizontal"  


# define the moment
if args.moment == 1:
    moment = "first"
elif args.moment == 2:
    moment = "second"
elif args.moment == 4:
    moment = "fourth"


plot_val = [
    "energy",
    "error",
    "entropy",
    "wilson_loop",
    "thooft",
    "mag"
]
plot_cmap = ["energy_tr", "error_tr", "entropy_tot"]

if args.loc == "pc":
    parent_path = "G:/My Drive/projects/1_Z2/"
elif args.loc == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2/"
elif args.loc == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2/"
else:
    raise SyntaxError("insert a valid location: 'pc', 'mac', 'marcos'")

yscale = "linear"

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
        args.bond = "all"


    if args.obs == "energy":
        title = f"Energy: lattice = ${args.l}$x${L-1}$ ; {sector}"
        fname_obs = "energies"
        fname_ex_obs = "energies"
        path = parent_path + f"results/energy_data"
        path_ex = parent_path + f"results/exact/energy_data"
        path_save = parent_path + f"figures/energy/"
        ylabel = "$\\langle E\\rangle$"
        txt = True

    elif args.obs == "error":
        title = f"Error: lattice = ${args.l}$x${L-1}$ ; {sector}"
        if args.time:
            fname_obs = "errors_quench_dynamics"
            fname_ex_obs = "errors_quench_dynamics"
        else:
            fname_obs = "errors"
            fname_ex_obs = "errors"
        path = parent_path + f"results/error_data"
        path_ex = parent_path + f"results/exact/error_data"
        path_save = parent_path + f"figures/error/"
        ylabel = "$\\left|\\left|\\left| \psi\\right>- \\left| \hat{O}\phi\\right>\\right|\\right|^2$"
        yscale = "log"
        txt = True

    elif args.obs == "entropy":
        title = (
            f" ${args.where}-th$ Bond Entanglement Entropy: lattice = ${args.l}$x${L-1}$ ; {sector}"
            # + f" $h \in ({args.h_i},{args.h_f})$"
        )
        if args.time:
            fname_obs = f"{args.where}_bond_entropy_quench_dynamics"
            fname_ex_obs = f"{args.where}_bond_entropy_quench_dynamics"
        else:
            fname_obs = f"{args.where}_bond_entropy"
            fname_ex_obs = f"{args.where}_bond_entropy"

        path = parent_path + f"results/entropy_data"
        path_ex = parent_path + f"results/exact/entropy_data"
        path_save = parent_path + f"figures/entropy/"
        ylabel = "entanglement von neumann entropy $(S_{\chi})$"
        txt = True

    elif args.obs == "wilson_loop":
        title = f"Wilson Loop: lattice = ${args.l}$x${L-1}$ ; {sector}"
        fname_obs = f"wilson_loop_{moment}_moment"
        fname_ex_obs = f"wilson_loop_{moment}_moment"
        path = parent_path + f"results/wilson_loops"
        path_ex = parent_path + f"results/exact/wilson_loops"
        path_save = parent_path + f"figures/wilson_loops/"
        ylabel = "$\\langle W\\rangle$"
        txt = False

    elif args.obs == "thooft":
        title = f"'t Hooft {direction} string: lattice = ${args.l}$x${L-1}$ ; {sector}" + " $\mu^z$ for " + f"L={args.sites[0]}, l={args.ladders[0]}" + " dual"
        fname_obs = f"thooft_string_{moment}_moment_{args.sites[0]}-{args.ladders[0]}_{direction}"
        fname_ex_obs = f"thooft_string_{moment}_moment_{args.sites[0]}-{args.ladders[0]}_{direction}"
        path = parent_path + f"results/thooft"
        path_ex = parent_path + f"results/exact/thooft"
        path_save = parent_path + f"figures/thooft/"
        ylabel = "$\\langle \sigma^x \dots \sigma^x \\rangle$"
        txt = False
    
    elif args.obs == "mag":
        title = f"Dual Magnetization: lattice = ${args.l}$x${L-1}$ ; {sector}" 
        fname_obs = f"dual_mag_{moment}_moment"
        fname_ex_obs = f"dual_mag_{moment}_moment"
        path = parent_path + f"results/mag_data"
        path_ex = parent_path + f"results/exact/mag_data"
        path_save = parent_path + f"figures/magnetization/"
        ylabel = "$\\langle \sigma^x \dots \sigma^x \\rangle$"
        txt = False

    elif args.obs == "entropy_tot":
        title = (
            f"All Bonds Entanglement Entropy: $lattice = {args.l}x{L}$ ; {sector}"
            + f" $h \in ({args.h_i},{args.h_f})$"
        )
        fname_obs = f"{args.where}_bond_entropy"
        path = parent_path + f"results/entropy_data"
        path_save = parent_path + f"figures/entropy/"
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

    else:
        raise SyntaxError(
            "insert a valid result to plot: 'energy', 'error', 'entropy', 'wilson_loop', 'thooft', 'mag', 'energy_tr', 'error_tr', 'entropy_tot'"
        )

    if args.obs in plot_val:
        if args.time:
            fname = f"{fname_obs}_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}"
            fname_ex = f"{fname_ex_obs}_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}"
            fname_save = f"{args.obs}_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_i_{args.h_i}_h_ev_{args.h_ev}_delta_{args.delta}_trotter_steps_{args.npoints}"
        else:
            fname = f"{fname_obs}_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}"
            fname_ex = f"{fname_ex_obs}_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}"
            fname_save = f"{args.obs}_{args.model}_direct_lattice_{args.l}x{L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}"
        plot_results_DMRG(
            title,
            for_array=args.chis,
            interval=interval,
            fname=fname,
            path=path,
            fname_save=fname_save,
            path_save=path_save,
            ylabel=ylabel,
            yscale=yscale,
            xscale=xscale,
            save=args.save,
            exact=args.exact,
            fname_ex=fname_ex,
            path_ex=path_ex,
            delta=args.delta,
            time=args.time,
            marker=args.marker,
            m_size=args.m_size,
            linewidth=args.linewidth,
            alpha=args.alpha,
            n_points=args.n_points,
            cmap=args.cmap,
            txt=txt,
            precision=precision
        )

    if args.obs in plot_cmap:
        for chi in args.chis:
            title_fin = title + f" ; $\chi = {chi}$"
            fname = f"{fname_obs}_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}"
            fname_save = f"{args.obs}_{args.model}_L_{args.L}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}"
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
