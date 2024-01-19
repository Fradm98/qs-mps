import argparse
import json
import numpy as np
from qs_mps.utils import get_precision
from qs_mps.mps_class import MPS

parser_ext = argparse.ArgumentParser(prog="observables_Z2_multi_run")
parser_ext.add_argument("p", help="path of the json files. Available paths are 'pc', 'mac', 'marcos'", type=str)
args_ext =parser_ext.parse_args()

def parse_args():
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

    return parser.parse_args()

def save_config(run_number, args):
    config = {
        'run': run_number,
        'arguments': vars(args)
    }

    with open(f'run_{run_number}_obs.json', 'w') as config_file:
        json.dump(config, config_file, indent=2)

if args_ext.p == "pc":
    parent_path = "G:/My Drive/projects/1_Z2"
elif args_ext.p == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2"
elif args_ext.p == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")

def load_config(run_number):
    with open(f'{parent_path}/json_files/run_{run_number}_obs.json', 'r') as config_file:
        config = json.load(config_file)
        return argparse.Namespace(**config['arguments'])

def main():

    # Uncomment the next line if you want to resume from the last run
    # while os.path.exists(f'run_{run_number}.json'):
    for run_number in range(1,5):
        print(f"\nRun {run_number}:")
        
        # Save arguments to a configuration file
        # args = parse_args()
        # save_config(run_number, args)

        # Load arguments
        args = load_config(run_number)

        # Your main script logic here using the loaded arguments
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

        # ---------------------------------------------------------
        # Observables
        # ---------------------------------------------------------
        for chi in args.chis:
            W = []
            E = []
            S = []
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

                if args.o == "el":
                    print(f"electric field for h:{h:.{precision}f}")
                    E_h = np.zeros((2*args.l+1,2*args.L-1))
                    E_h[:] = np.nan
                    E_h = lattice_mps.electric_field_Z2(E_h)
                    E.append(E_h)
                
                if args.o == "thooft":
                    print(f"'t Hooft string for h:{h:.{precision}f}")
                    lattice_mps.Z2.thooft(site=args.sites, l=args.ladders, direction=direction)
                    lattice_mps.w = lattice_mps.Z2.mpo.copy()
                    S.append(lattice_mps.mpo_first_moment().real)


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
            if args.o == "hooft":
                np.save(
                            f"{parent_path}/results/thooft/thooft_string_{args.sites[0]}-{args.ladders[0]}_{direction}_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy",
                            S,
                        )

if __name__ == '__main__':
    main()
