import matplotlib.pyplot as plt
import numpy as np
from qs_mps.utils import plot_results_evolution
import argparse

# default parameters of the plot layout
plt.rcParams["text.usetex"] = True # use latex
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
    "what", help="Results we want to plot. Available are: 'mag_tot', 'mag_z', 'mag_x', 'error_compr', 'entropy', 'err_mag_tot', 'err_mag_z', 'err_mag_x', 'err_entropy'", type=str
)
parser.add_argument(
    "loc", help="From which computer you want to access the drive. Useful for the path spec. Available are: 'pc', 'mac', 'marcos'", type=str
)
parser.add_argument("chis", help="Simulated bond dimensions", nargs="+", type=int)
parser.add_argument(
    "-f", "--flip", help="Flip the middle site or not", action="store_false"
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Ising", type=str
)
parser.add_argument(
    "-t", "--time", help="Final time of the evolution", default=10, type=float
)
parser.add_argument(
    "-e",
    "--exact",
    help="Plot exact results for comparison. Doable only for small Ising chains",
    action="store_true",
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
parser.add_argument(
    "-z", "--m_size", help="marker size", default=20, type=int
)
parser.add_argument(
    "-l", "--linewidth", help="Linewidth of the marker", default=1, type=float,
)
parser.add_argument(
    "-a", "--alpha", help="Alpha of the marker", default=1, type=float,
)
parser.add_argument(
    "-n",
    "--n_points",
    help="Percentage of number of points we want to plot, from 0 to 1",
    default=1,
    type=float,
)

args = parser.parse_args()
delta = args.time / args.trotter_steps
if args.where == -1:
    args.where = (args.L // 2)
elif args.where == -2:
    args.bond = False
    args.where = "all"

if args.loc == 'pc':
    path_computer = "G:/Google Drive/My Drive/projects/0_ISING/"
elif args.loc == 'mac':
    path_computer = "/Users/fradm98/Google Drive/My Drive/projects/0_ISING/"
elif args.loc == 'marcos':
    path_computer = "/Users/fradm/Google Drive/My Drive/projects/0_ISING/"
else:
    raise SyntaxError("insert a valid location: 'pc', 'mac', 'marcos'")

if args.what == "mag_tot":
    title = f"Total Magnetization: $L = {args.L}$ ; $\delta = {delta}$ ;" + " $h_{ev}=$ " + f"${args.h_ev}$"
    fname_what = "mag_mps_tot"
    fname_ex_what = "mag_exact_tot"
    path = path_computer + f"results/mag_data"
    path_ex = path_computer + f"results/exact/mag_data"
    path_save = path_computer + f"figures/magnetization/"
    ylabel = "$\sum_{i=1}^L \sigma_i^z$"

elif args.what == "mag_z":
    title = f"Magnetization Z in the middle site: $L = {args.L}$ ; $\delta = {delta}$ ;" + " $h_{ev}=$ "+ f"${args.h_ev}$"
    fname_what = "mag_mps_loc_Z"
    fname_ex_what = "mag_exact_loc_Z"
    path = path_computer + f"results/mag_data"
    path_ex = path_computer + f"results/exact/mag_data"
    path_save = path_computer + f"figures/magnetization/"
    ylabel = "$\sigma_{L/2}^z$"

elif args.what == "mag_x":
    title = f"Magnetization X in the middle site: $L = {args.L}$ ; $\delta = {delta}$ ;" + " $h_{ev}=$ " + f"${args.h_ev}$"
    fname_what = "mag_mps_loc_X"
    fname_ex_what = "mag_exact_loc_X"
    path = path_computer + f"results/mag_data"
    path_ex = path_computer + f"results/exact/mag_data"
    path_save = path_computer + f"figures/magnetization/"
    ylabel = "$\sigma_{L/2}^x$"

elif args.what == "entropy":
    title = f" ${args.where}-th$ Bond Entanglement Entropy: $L = {args.L}$ ; $\delta = {delta}$ ;" + " $h_{ev}=$ " + f"${args.h_ev}$"
    fname_what = f"{args.where}_bond_entropy"
    fname_ex_what = f"{args.where}_bond_exact_entropy"
    path = path_computer + f"results/bonds_data"
    path_ex = path_computer + f"results/exact/entropy"
    path_save = path_computer + f"figures/entropy/"
else:
    raise SyntaxError("insert a valid result to plot: 'mag_tot', 'mag_z', 'mag_x', 'error_compr', 'entropy', 'err_mag_tot', 'err_mag_z', 'err_mag_x', 'err_entropy'")


fname = f"{fname_what}_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}"
fname_ex = f"{fname_ex_what}_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}"
fname_save = f"{args.what}_{args.model}_L_{args.L}_flip_{args.flip}_delta_{delta}_h_ev_{args.h_ev}"


plot_results_evolution(title, for_array=args.chis, trotter_steps=args.trotter_steps, delta=delta, fname=fname, path=path, fname_ex=fname_ex, path_ex=path_ex, fname_save=fname_save, path_save=path_save, ylabel=ylabel, exact=args.exact, save=args.save, marker=args.marker, m_size=args.m_size, linewidth=args.linewidth, alpha=args.alpha, n_points=args.n_points)

