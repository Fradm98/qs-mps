import argparse
import json
import numpy as np
from qs_mps.utils import get_precision, save_list_of_lists, access_txt
from qs_mps.applications.Z2.ground_state_multiprocessing import ground_state_Z2
from qs_mps.mps_class import MPS
import os
import concurrent.futures

parser_ext = argparse.ArgumentParser(prog="gs_search_Z2_multi_run")
parser_ext.add_argument("p", help="path of the json files. Available paths are 'pc', 'mac', 'marcos'", type=str)
args_ext =parser_ext.parse_args()
# def parse_args():
#     parser = argparse.ArgumentParser(prog="gs_search_Z2")
#     parser.add_argument("l", help="Number of ladders in the direct lattice", type=int)
#     parser.add_argument("L", help="Number of rungs per ladder", type=int)
#     parser.add_argument(
#         "npoints",
#         help="Number of points in an interval of transverse field values",
#         type=int,
#     )
#     parser.add_argument(
#         "h_i", help="Starting value of h (external transverse field on the dual lattice)", type=float
#     )
#     parser.add_argument(
#         "h_f", help="Final value of h (external transverse field on the dual lattice)", type=float
#     )
#     parser.add_argument(
#         "path",
#         help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
#         type=str,
#     )
#     parser.add_argument("chis", help="Simulated bond dimensions", nargs="+", type=int)
#     parser.add_argument("-cx", "--charges_x", help="a list of the first index of the charges", nargs="*", type=int)
#     parser.add_argument("-cy", "--charges_y", help="a list of the second index of the charges", nargs="*", type=int)

#     parser.add_argument(
#         "-ty", "--type_shape", help="Type of shape of the bond dimension. Available are: 'trapezoidal', 'pyramidal', 'rectangular'", default="trapezoidal", type=str
#     )
#     parser.add_argument(
#         "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
#     )
#     parser.add_argument(
#         "-mu", "--multpr", help="If True computes ground states with multiprocessing. By default True", action="store_false"
#     )
#     parser.add_argument(
#         "-s",
#         "--number_sweeps",
#         help="Number of sweeps during the compression algorithm for each trotter step",
#         default=8,
#         type=int,
#     )
#     parser.add_argument(
#         "-cv",
#         "--conv_tol",
#         help="Convergence tolerance of the compression algorithm",
#         default=1e-10,
#         type=float,
#     )
#     parser.add_argument(
#         "-b",
#         "--bond",
#         help="Save the schmidt values for one bond. If False save for each bond. By default True",
#         action="store_false",
#     )
#     parser.add_argument(
#         "-w",
#         "--where",
#         help="Bond where we want to observe the Schmidt values, should be between 1 and (L-1)",
#         default=-1,
#         type=int,
#     )
#     parser.add_argument(
#         "-v",
#         "--save",
#         help="Save the tensors. By default True",
#         action="store_false",
#     )
#     parser.add_argument(
#         "-tr",
#         "--training",
#         help="Save all the energies also the ones during the variational optimization. By default True",
#         action="store_false",
#     )

#     return parser.parse_args()

# def save_config(run_number, args):
#     config = {
#         'run': run_number,
#         'arguments': vars(args)
#     }

#     with open(f'run_{run_number}_gs.json', 'w') as config_file:
#         json.dump(config, config_file, indent=2)

if args_ext.p == "pc":
    parent_path = "G:/My Drive/projects/1_Z2"
elif args_ext.p == "mac":
    parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2"
elif args_ext.p == "marcos":
    parent_path = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")

def load_config(run_number):
    with open(f'{parent_path}/json_files/run_{run_number}_gs.json', 'r') as config_file:
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

        # define the sector by looking of the given charges
        # print(f"charge x: {type(args.charges_x)}")
        if args.charges_x == []:
            sector = "vacuum_sector"
            args.charges_x = None
            args.charges_y = None
        else:
            for i in range(1,args.l*args.L):
                if len(args.charges_x) == i:
                    sector = f"{i}_particle(s)_sector"

        # where to look at for the entropy
        if args.where == -1:
            args.where = args.L // 2
        elif args.where == -2:
            args.bond = False

        # ---------------------------------------------------------
        # DMRG
        # ---------------------------------------------------------
        for chi in args.chis:  # L // 2 + 1
            args_mps = {
                "L": args.L,
                "d": d,
                "chi": chi,
                "type_shape": args.type_shape,
                "model": args.model,
                "trunc_tol": False,
                "trunc_chi": True,
                "where": args.where,
                "bond": args.bond,
                "path": path_tensor,
                "save": args.save,
                "precision": precision,
                "sector": sector,
                "charges_x": args.charges_x,
                "charges_y": args.charges_y,
            }
            if __name__ == "__main__":
                energy_chi, entropy_chi = ground_state_Z2(
                    args_mps=args_mps, multpr=args.multpr, param=interval
                )

                # if args.bond == False:
                #     args.where = "all"

                # if args.training:
                #     save_list_of_lists(
                #         f"{parent_path}/results/energy_data/energies_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                #         energy_chi,
                #     )
                #     energy_gs = access_txt(
                #             f"{parent_path}/results/energy_data/energies_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                #             -1,
                #         )
                #     np.savetxt(
                #         f"{parent_path}/results/energy_data/energies_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                #         energy_gs,
                #     )
                # else:
                #     np.savetxt(
                #         f"{parent_path}/results/energy_data/energies_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                #         energy_chi,
                #     )
                    
                # save_list_of_lists(
                #     f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                #     entropy_chi,
                # )
                # if args.where == "all":
                #     entropy_mid = access_txt(
                #         f"{parent_path}/results/entropy_data/{args.where}_bond_entropy_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                #         (args.L-1) // 2,
                #     )
                #     np.savetxt(
                #         f"{parent_path}/results/entropy_data/{args.L // 2}_bond_entropy_{args.model}_direct_lattice_{args.l}x{args.L-1}_{sector}_{args.charges_x}-{args.charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                #         entropy_mid,
                #     )

if __name__ == '__main__':
    main()
