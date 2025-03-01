# import packages
import argparse
from qs_mps.mps_class import MPS
from qs_mps.utils import *

parser = argparse.ArgumentParser(prog="observables_Z2_mps")
parser.add_argument("l", help="Number of ladders in the direct lattice", type=int)
parser.add_argument(
    "npoints",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "h_i",
    help="Starting value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "h_f",
    help="Final value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument(
    "-o",
    "--obs",
    help="Observable we want to compute. Available are 'wl', 'wl_av', 'el', 'thooft', 'mag', 'corr', 'en', 'end', 'md'",
    nargs="+",
    type=str,
)
parser.add_argument(
    "-L", "--Ls", help="Number of rungs per ladder", nargs="+", type=int
)
parser.add_argument(
    "-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int
)
parser.add_argument(
    "-R", "--Rs", help="Strings we want to analyze", nargs="*", type=int
)
parser.add_argument(
    "-cx",
    "--charges_x",
    help="a list of the first index of the charges",
    nargs="*",
    type=int,
)
parser.add_argument(
    "-cy",
    "--charges_y",
    help="a list of the second index of the charges",
    nargs="*",
    type=int,
)
parser.add_argument("-lx", "--sites", help="coordinates of sites", nargs="*", type=int)
parser.add_argument(
    "-ly", "--ladders", help="coordinates of ladders", nargs="*", type=int
)
parser.add_argument(
    "-d", "--direction", help="Direction of the string", default="hor", type=str
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
)
parser.add_argument(
    "-mo",
    "--moment",
    help="Moment degree of the Free energy. E.g. Magnetization -> First Moment, Susceptibility -> Second Moment, etc. Available are 1,2,4",
    default=1,
    type=int,
)
parser.add_argument(
    "-i",
    "--interval",
    help="Type of interval spacing. Available are 'log', 'lin'",
    default="lin",
    type=str,
)
parser.add_argument(
    "-bc",
    "--boundcond",
    help="Type of boundary conditions. Available are 'obc', 'pbc'",
    default="pbc",
    type=str,
)
parser.add_argument(
    "-p",
    "--prec",
    help="Precision of the coupling constant",
    default=None,
    type=int,
)

args = parser.parse_args()

# define the physical dimension
d = int(2 ** (args.l))

# define the interval of equally spaced values of external field
if args.interval == "lin":
    interval = np.linspace(args.h_i, args.h_f, args.npoints)
    num = (interval[-1] - interval[0]) / args.npoints
    if args.prec == None:
        precision = get_precision(num)
    else:
        precision = args.prec
elif args.interval == "log":
    interval = np.logspace(args.h_i, args.h_f, args.npoints)
    if args.prec == None:
        precision = int(np.max([np.abs(args.h_f), np.abs(args.h_i)]))
    else:
        precision = args.prec

# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = f"C:/Users/HP/Desktop/projects/1_Z2"
    # parent_path = "G:/My Drive/projects/1_Z2"
    path_tensor = "D:/code/projects/1_Z2"
elif args.path == "mac":
    # parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/Users/fradm98/Desktop/projects/1_Z2"
    parent_path = path_tensor
elif args.path == "marcos":
    # parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/Users/fradm/Desktop/projects/1_Z2"
    parent_path = path_tensor
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")


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


# ---------------------------------------------------------
# Observables
# ---------------------------------------------------------
for L in args.Ls:
    # define the sector by looking of the given charges
    if len(args.charges_x) == 0 and len(args.Rs) == 0:
        sector = "vacuum_sector"
        charges_x = None
        charges_y = None
    else:
        sector = f"{len(args.charges_x)}_particle(s)_sector"
        charges_x = args.charges_x
        charges_y = args.charges_y
    

    for chi in args.chis:
        W = []
        W_av = []
        E = []
        En = []
        End = []
        S = []
        M = []
        Md = []
        C = []
        Pot = []
        for h in interval:
            lattice_mps = MPS(
                L=L, d=d, model=args.model, chi=chi, h=h, bc=args.boundcond
            )

            if sector != "vacuum_sector":
                lattice_mps.Z2.add_charges(charges_x, charges_y)
                lattice_mps.charges = lattice_mps.Z2.charges
                lattice_mps.Z2._define_sector()
            else:
                lattice_mps.Z2._define_sector()
            lattice_mps.load_sites(
                path=path_tensor, precision=precision, cx=charges_x, cy=charges_y
            )

            if "wl" in args.obs:
                print(
                    f"wilson loop for h:{h:.{precision}f}, direct lattice lxL:{args.l}x{L}, bc: {args.boundcond}, chi:{chi}"
                )
                lattice_mps.Z2.wilson_Z2_dual(
                    mpo_sites=args.sites, ls=args.ladders
                )  # list(range(s))
                lattice_mps.w = lattice_mps.Z2.mpo.copy()
                if args.moment == 1:
                    # print(lattice_mps.mpo_first_moment().real)
                    W.append(lattice_mps.mpo_first_moment().real)
                elif args.moment == 2:
                    # print(lattice_mps.mpo_second_moment().real)
                    W.append(lattice_mps.mpo_second_moment().real)
                elif args.moment == 4:
                    # print(lattice_mps.mpo_fourth_moment().real)
                    W.append(lattice_mps.mpo_fourth_moment().real)

            if "wl_av" in args.obs:
                print(
                    f"wilson loop average for h:{h:.{precision}f}, direct lattice lxL:{args.l}x{L}, bc: {args.boundcond}, chi:{chi}"
                )
                wav = []
                for Ly in range(args.l):
                    for Lx in range(L - 1):
                        lattice_mps.Z2.wilson_Z2_dual(
                            mpo_sites=[Lx], ls=[Ly]
                        )  # list(range(s))
                        lattice_mps.w = lattice_mps.Z2.mpo.copy()
                        if args.moment == 1:
                            # print(lattice_mps.mpo_first_moment().real)
                            wav.append(lattice_mps.mpo_first_moment().real)
                        elif args.moment == 2:
                            # print(lattice_mps.mpo_second_moment().real)
                            wav.append(lattice_mps.mpo_second_moment().real)
                        elif args.moment == 4:
                            # print(lattice_mps.mpo_fourth_moment().real)
                            wav.append(lattice_mps.mpo_fourth_moment().real)
                N = len(wav)
                wav = 1 / N * np.sum(wav)
                W_av.append(wav)

            if "el" in args.obs:
                print(
                    f"electric field for h:{h:.{precision}f}, direct lattice lxL:{args.l}x{L}, bc: {args.boundcond}, chi:{chi}"
                )
                if lattice_mps.bc == "obc":
                    E_h = np.zeros((2 * args.l + 1, 2 * L + 1))
                if lattice_mps.bc == "pbc":
                    E_h = np.zeros((2 * args.l, 2 * L + 1))
                    # a = np.zeros((1,2))
                    # a[0,0] = 1
                    # extra_ancillary_site = a.reshape((1,2,1))
                    # lattice_mps.sites.append(extra_ancillary_site)
                    # lattice_mps.L = len(lattice_mps.sites)

                E_h[:] = np.nan
                E_h = lattice_mps.electric_field_Z2(E_h)
                E.append(E_h)

            if "thooft" in args.obs:
                print(
                    f"'t Hooft string for h:{h:.{precision}f}, direct lattice lxL:{args.l}x{L}, bc: {args.boundcond}, chi:{chi}"
                )
                lattice_mps.Z2.thooft(
                    site=args.sites, l=args.ladders, direction=direction
                )
                lattice_mps.w = lattice_mps.Z2.mpo.copy()
                S.append(lattice_mps.mpo_first_moment().real)

            if "mag" in args.obs:
                print(
                    f"Magnetization for h:{h:.{precision}f}, direct lattice lxL:{args.l}x{L}, bc: {args.boundcond}, chi:{chi}"
                )
                lattice_mps.order_param()
                if args.moment == 1:
                    #  print(lattice_mps.mpo_first_moment().real, (len(lattice_mps.Z2.latt.plaquettes())-(2*(L-3)+2*(args.l))))
                    M.append(
                        lattice_mps.mpo_first_moment().real
                        / (
                            len(lattice_mps.Z2.latt.plaquettes())
                            - (2 * (L - 3) + 2 * (args.l))
                        )
                    )
                elif args.moment == 2:
                    M.append(
                        lattice_mps.mpo_second_moment().real
                        / (
                            len(lattice_mps.Z2.latt.plaquettes())
                            - (2 * (L - 3) + 2 * (args.l))
                        )
                        ** 2
                    )
                elif args.moment == 4:
                    M.append(
                        lattice_mps.mpo_fourth_moment().real
                        / (
                            len(lattice_mps.Z2.latt.plaquettes())
                            - (2 * (L - 3) + 2 * (args.l))
                        )
                        ** 4
                    )

            if "md" in args.obs:
                Md.append(lattice_mps.local_magnetization_Z2_dual())

            if "corr" in args.obs:
                print(
                    f"Correlator for h:{h:.{precision}f}, direct lattice lxL:{args.l}x{L}, bc: {args.boundcond}, chi:{chi}"
                )
                c = lattice_mps.connected_correlator(
                    site=args.sites[0], lad=args.ladders[0]
                )
                C.append(c)

            if "en" in args.obs:
                if lattice_mps.bc == "pbc":
                    a = np.zeros((1,2))
                    a[0,0] = 1
                    extra_ancillary_site = a.reshape((1,2,1))
                    lattice_mps.sites.append(extra_ancillary_site)
                    lattice_mps.L = len(lattice_mps.sites)

                print(
                    f"Ground state energy for h:{h:.{precision}f}, direct lattice lxL:{args.l}x{L}, bc: {args.boundcond}, chi:{chi}"
                )
                lattice_mps.Z2.mpo_Z2_ladder_generalized_pbc()
                lattice_mps.w = lattice_mps.Z2.mpo.copy()
                En.append(lattice_mps.mpo_first_moment().real)

            if "end" in args.obs:
                # print(f"Energy density for h:{h:.{precision}f}, direct lattice lxL:{args.l}x{L}, bc: {args.boundcond}, chi:{chi}, sector: {sector}")
                End.append(
                    lattice_mps.mpo_Z2_column_electric_energy_density(site=L // 2)
                )
            if "pot" in args.obs:
                potr = []
                for R in args.Rs:
                    sector = f"{len(args.charges_x)}_particle(s)_sector"
                    charges_x = get_cx(L,R)
                    charges_y = get_cy(args.l, args.boundcond)

                    lattice_mps = MPS(
                    L=L, d=d, model=args.model, chi=chi, h=h, bc=args.boundcond
                    )

                    if sector != "vacuum_sector":
                        lattice_mps.Z2.add_charges(charges_x, charges_y)
                        lattice_mps.charges = lattice_mps.Z2.charges
                        lattice_mps.Z2._define_sector()
                    else:
                        lattice_mps.Z2._define_sector()
                    lattice_mps.load_sites(
                        path=path_tensor, precision=precision, cx=charges_x, cy=charges_y
                    )
                    if lattice_mps.bc == "pbc":
                        a = np.zeros((1,2))
                        a[0,0] = 1
                        extra_ancillary_site = a.reshape((1,2,1))
                        lattice_mps.sites.append(extra_ancillary_site)
                        lattice_mps.L = len(lattice_mps.sites)

                
                    print("ground state of the charges hamiltonian in the two-particles sector")
                    lattice_mps.Z2.mpo_Z2_ladder_generalized_pbc()
                    lattice_mps.w = lattice_mps.Z2.mpo.copy()
                    energy0 = lattice_mps.mpo_first_moment().real
                    print(energy0)

                    print("ground state of the vacuum hamiltonian in the two-particles sector")
                    lattice_mps.Z2.charges = lattice_mps.Z2._define_charges()
                    lattice_mps.Z2.mpo_Z2_ladder_generalized_pbc()
                    lattice_mps.w = lattice_mps.Z2.mpo.copy()
                    energy2p = lattice_mps.mpo_first_moment().real
                    print(energy2p)

                    potr.append(energy2p-energy0)
                Pot.append(potr)


        if "wl" in args.obs:
            np.save(
                f"{parent_path}/results/wilson_loops/wilson_loop_{moment}_moment_{args.sites}-{args.ladders}_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                W,
            )
        if "wl_av" in args.obs:
            np.save(
                f"{parent_path}/results/wilson_loops/wilson_loop_average_{moment}_moment_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                W_av,
            )
        if "el" in args.obs:
            np.save(
                f"{parent_path}/results/electric_field/electric_field_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                E,
            )
        if "thooft" in args.obs:
            np.save(
                f"{parent_path}/results/thooft/thooft_string_{moment}_moment_{args.sites[0]}-{args.ladders[0]}_{direction}_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                S,
            )
        if "mag" in args.obs:
            np.save(
                f"{parent_path}/results/mag_data/direct_mag_{moment}_moment_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                M,
            )
        
        if "md" in args.obs:
            np.save(
                f"{parent_path}/results/mag_data/dual_mag_{moment}_moment_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                Md,
            )

        if "corr" in args.obs:
            C = np.array_split(C, args.npoints)
            np.save(
                f"{parent_path}/results/mag_data/connected_correlator_s_{args.sites[0]}_l_{args.ladders[0]}_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                C,
            )
        if "en" in args.obs:
            np.save(
                f"{parent_path}/results/energy_data/energy_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                En,
            )
        if "end" in args.obs:
            np.save(
                f"{parent_path}/results/energy_data/electric_energy_density_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                End,
            )
        if "pot" in args.obs:
            np.save(
                f"{parent_path}/results/energy_data/string_potential_vacuum_ham_on_charge_state_{args.model}_direct_lattice_{args.l}x{L}_{sector}_bc_{args.boundcond}_{args.Rs}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                Pot,
            )