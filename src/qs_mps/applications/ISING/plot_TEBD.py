import matplotlib.pyplot as plt
import numpy as np
from qs_mps.utils import plot_results_TEBD, plot_colormaps_evolution
import argparse

# default parameters of the plot layout
plt.rcParams["text.usetex"] = True  # use latex
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True


parser = argparse.ArgumentParser(prog="Plot Time Evolution")
parser.add_argument("L", help="Spin chain length", type=int)
parser.add_argument(
    "trotter_steps",
    help="It will give you the accuracy of the trotter evolution for fixed t",
    type=int,
)
parser.add_argument(
    "h_ev", help="It will give you the magnitude of the quench", type=float
)
parser.add_argument(
    "what",
    help="Results we want to plot. Available are: 'mag_tot', 'mag_z', 'mag_x', 'entropy', 'err_mag_tot', 'err_mag_z', 'err_mag_x', 'err_entropy', 'entropy_tot', 'mag_loc' ",
    type=str,
)
parser.add_argument(
    "loc",
    help="From which computer you want to access the drive. Useful for the path spec. Available are: 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument("chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument(
    "-f",
    "--flip",
    help="Flip the middle site or not. By defalut True",
    action="store_false",
)
parser.add_argument(
    "-q",
    "--quench",
    help="Type of quench. Available are 'flip', 'global'",
    default="global",
    type=str,
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Ising", type=str
)
parser.add_argument(
    "-t", "--time", help="Final time of the evolution", default=10, type=float
)
parser.add_argument(
    "-s",
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
    "-l",
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

# interval = np.linspace(args.h_i, args.h_f, args.npoints).tolist()
delta = args.time / args.trotter_steps

plot_val = [
    "mag_tot",
    "mag_z",
    "mag_x",
    "entropy",
    "err_mag_tot",
    "err_mag_z",
    "err_mag_x",
    "err_entropy",
]
plot_cmap = ["mag_loc", "entropy_tot"]

if args.loc == "pc":
    path_computer = "G:/My Drive/projects/0_ISING/"
elif args.loc == "mac":
    path_computer = "/Users/fradm98/Google Drive/My Drive/projects/0_ISING/"
elif args.loc == "marcos":
    path_computer = "/Users/fradm/Google Drive/My Drive/projects/0_ISING/"
else:
    raise SyntaxError("insert a valid location: 'pc', 'mac', 'marcos'")

if args.what == "mag_tot":
    title = f"Total Magnetization: $L = {args.L}$ ;" + " $h_{ev}=$ " + f"${args.h_ev}$"
    fname_what = "mag_mps_tot"
    fname_ex_what = "mag_exact_tot"
    path = path_computer + f"results/mag_data"
    path_ex = path_computer + f"results/exact/mag_data"
    path_save = path_computer + f"figures/magnetization/"
    ylabel = "$\\sum_{i=1}^L \\sigma_i^z$"

elif args.what == "mag_z":
    title = (
        f"Magnetization Z in the middle site: $L = {args.L}$ ;"
        + " $h_{ev}=$ "
        + f"${args.h_ev}$"
    )
    fname_what = "mag_mps_loc_Z"
    fname_ex_what = "mag_exact_loc_Z"
    path = path_computer + f"results/mag_data"
    path_ex = path_computer + f"results/exact/mag_data"
    path_save = path_computer + f"figures/magnetization/"
    ylabel = "$\\sigma_{L/2}^z$"

elif args.what == "mag_x":
    title = (
        f"Magnetization X in the middle site: $L = {args.L}$ ;"
        + " $h_{ev}=$ "
        + f"${args.h_ev}$"
    )
    fname_what = "mag_mps_loc_X"
    fname_ex_what = "mag_exact_loc_X"
    path = path_computer + f"results/mag_data"
    path_ex = path_computer + f"results/exact/mag_data"
    path_save = path_computer + f"figures/magnetization/"
    ylabel = "$\\sigma_{L/2}^x$"

elif args.what == "entropy":
    title = (
        f" ${args.where}-th$ Bond Entanglement Entropy: $L = {args.L}$ ;"
        + " $h_{ev}=$"
        + f" ${args.h_ev}$"
    )
    fname_what = f"{args.where}_bond_entropy"
    fname_ex_what = f"{args.where}_bond_exact_entropy"
    path = path_computer + f"results/entropy"
    path_ex = path_computer + f"results/exact/entropy"
    path_save = path_computer + f"figures/entropy/"
    ylabel = "entanglement von neumann entropy $(S_{\\chi})$"

elif args.what == "entropy_tot":
    title = (
        f"All Bonds Entanglement Entropy: $L = {args.L}$ ;"
        + " $h_{ev}=$"
        + f" ${args.h_ev}$"
    )
    fname_what = f"{args.where}_bond_entropy"
    path = path_computer + f"results/entropy"
    path_save = path_computer + f"figures/entropy/"
    xlabel = "bonds"
    x_ticks = range(args.L - 1)
    labels = range(1, args.L - 1 + 1)
    steps = len(x_ticks) // 5
    xticks = x_ticks[::steps]
    steps = len(labels) // 5
    xlabels = labels[::steps]
    y_ticks = np.arange(args.trotter_steps + 1)
    labels = delta * y_ticks
    steps = len(y_ticks) // 5
    yticks = y_ticks[::steps]
    steps = len(labels) // 5
    ylabels = labels[::steps]
    # x_grid = list(x_ticks).copy
    # y_grid = list(y_ticks).copy()
    # y_grid.pop(-1)
    # X,Y = np.meshgrid(np.asarray(x_grid), np.asarray(y_grid))
    X, Y = np.meshgrid(x_ticks, y_ticks)
    view_init = False

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
    labels = delta * y_ticks
    steps = len(y_ticks) // 5
    yticks = y_ticks[::steps]
    steps = len(labels) // 5
    ylabels = labels[::steps]
    X, Y = np.meshgrid(x_ticks, y_ticks)
    view_init = True

else:
    raise SyntaxError(
        "insert a valid result to plot: 'mag_tot', 'mag_z', 'mag_x', 'entropy', 'err_mag_tot', 'err_mag_z', 'err_mag_x', 'err_entropy'"
    )

if args.what in plot_val:
    fname_ex = f"{fname_ex_what}_{args.model}_L_{args.L}_midflip_{args.flip}_quench_{args.quench}_delta_{delta}_h_ev_{args.h_ev}"
    fname = f"{fname_what}_{args.model}_L_{args.L}_midflip_{args.flip}_quench_{args.quench}_delta_{delta}"
    second_part = f"_h_ev_{args.h_ev}"
    fname_save = f"{args.what}_{args.model}_L_{args.L}_midflip_{args.flip}_quench_{args.quench}_delta_{delta}"
    second_part = f"_h_ev_{args.h_ev}"
    plot_results_TEBD(
        title,
        for_array=args.chis,
        trotter_steps=args.trotter_steps,
        delta=delta,
        fname=fname,
        second_part=second_part,
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
    )

if args.what in plot_cmap:
    for chi in args.chis:
        title_fin = title + f" ; $\\chi = {chi}$"
        fname = f"{fname_what}_{args.model}_L_{args.L}_midflip_{args.flip}_quench_{args.quench}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}"
        fname_save = f"{args.what}_{args.model}_L_{args.L}_midflip_{args.flip}_quench_{args.quench}_delta_{delta}_chi_{chi}_h_ev_{args.h_ev}"
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
