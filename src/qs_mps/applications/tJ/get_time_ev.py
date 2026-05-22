import argparse
import numpy as np
import datetime as dt
from qs_mps.mps_class import MPS
from qs_mps.utils import save_list_of_lists, access_txt, tensor_shapes
from qs_mps.applications.tJ.ed_ham import mpo_ev_trotter_i_ip1_pipeline, mpo_ev_trotter_i_ip2_pipeline, neel_prod_state

# DENSITY MATRIX RENORMALIZATION GROUP to find ground states of the Z2 Pure Gauge Theory
# changing the transverse field parameters in its dual formulation

parser = argparse.ArgumentParser(prog="gs_search_Heis")
parser.add_argument(
    "trott",
    help="Number of points in an interval of transverse field values",
    type=int,
)
parser.add_argument(
    "tf",
    help="Final value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "Jz",
    help="Starting value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "J_perp",
    help="Final value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "t_up",
    help="Final value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "t_down",
    help="Final value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "tp_up",
    help="Final value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "tp_down",
    help="Final value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "V",
    help="Final value of h (external transverse field on the dual lattice)",
    type=float,
)
parser.add_argument(
    "path",
    help="Path to the drive depending on the device used. Available are 'pc', 'mac', 'marcos'",
    type=str,
)
parser.add_argument(
    "-L", "--Ls", help="Number of rungs per ladder", nargs="+", type=int
)
parser.add_argument(
    "-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int
)
parser.add_argument(
    "-df",
    "--defects",
    help="a list of the first index of the charges",
    default=0,
    type=int,
)
parser.add_argument(
    "-ty",
    "--type_shape",
    help="Type of shape of the bond dimension. Available are: 'trapezoidal', 'pyramidal', 'rectangular'",
    default="rectangular",
    type=str,
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="heis", type=str
)
parser.add_argument(
    "-mu",
    "--multpr",
    help="If True computes ground states with multiprocessing. By default False",
    action="store_true",
)
parser.add_argument(
    "-s",
    "--number_sweeps",
    help="Number of sweeps during the compression algorithm for each trotter step",
    default=10,
    type=int,
)
parser.add_argument(
    "-cv",
    "--conv_tol",
    help="Convergence tolerance of the compression algorithm",
    default=1e-12,
    type=float,
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
    help="Bond where we want to observe the Schmidt values, should be between 1 and (L)",
    default=-1,
    type=int,
)
parser.add_argument(
    "-v",
    "--save",
    help="Save the tensors. By default True",
    action="store_false",
)
parser.add_argument(
    "-bc",
    "--boundcond",
    help="Type of boundary conditions. Available are 'obc', 'pbc'",
    default="obc",
    type=str,
)
parser.add_argument(
    "-log",
    "--logging",
    help="Name to log the output of the computation",
    default="output.out",
    type=str,
)
parser.add_argument(
    "-cc",
    "--chargeconv",
    help="Type of Charge convension for obc. Available are 'h', 'v'. By default 'h'",
    default="h",
    type=str,
)
parser.add_argument(
    "-p",
    "--precision",
    help="Precision to load and save tensors and observables. By default True 3",
    default=3,
    type=int,
)
parser.add_argument(
    "-j",
    "--J",
    help="Value of the J coupling in the hamiltonian. By default 1",
    default=1,
    type=float,
)
parser.add_argument(
    "-e",
    "--eps",
    help="Value of the epsilon coupling to penalize holes creation. By default 0",
    default=0,
    type=float,
)
parser.add_argument(
    "-exc",
    "--excited",
    help="First excited state. By default False",
    action="store_true",
)

args = parser.parse_args()

# # Redirect stdout and stderr to the log file
# sys.stdout = open(f'results/logs/{args.logging}', 'w')
# sys.stderr = sys.stdout


# take the path and precision to save files
# if we want to save the tensors we save them locally because they occupy a lot of memory
if args.path == "pc":
    parent_path = f"C:/Users/HP/Desktop/projects/Fidelities_with_TN"
    # parent_path = "G:/My Drive/projects/Fidelities_with_TN"
    path_tensor = "D:/code/projects/6_TJ"
elif args.path == "mac":
    # parent_path = "/Users/fradm98/Google Drive/My Drive/projects/Fidelities_with_TN"
    path_tensor = "/Users/fradm98/Desktop/projects/Fidelities_with_TN"
    parent_path = path_tensor
elif args.path == "marcos":
    # parent_path = "/Users/fradm/Google Drive/My Drive/projects/Fidelities_with_TN"
    path_tensor = "/Users/fradm/Desktop/projects/Fidelities_with_TN"
    parent_path = path_tensor
elif args.path == "ngt":
    path_tensor = "/eos/user/f/fdimarca/projects/6_TJ"
    parent_path = path_tensor
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")


# ---------------------------------------------------------
# TEBD
# ---------------------------------------------------------

def initial_state(defect: int, *args):
    hole_tn = np.array([[[0],[1],[0]]])
    if defect == 0:
        mps_chain = MPS(L=args["L"],d=args["d"],model=args["model"],chi=args["chi"],J=args["Jz"],h=args["J_perp"],k=(args["t"],args["tp"]))
        mps_chain.load_sites(path=args["path"],precision=args["precision"])
        mps_chain.sites[args["L"]//2] = hole_tn.copy()
        mps_chain.sites[args["L"]//2-1] = hole_tn.copy()
        mps_chain.enlarge_chi(noise_std=1e-7)
        mps_chain.canonical_form(svd_direction="left")
        mps_chain.canonical_form(svd_direction="right")
        init_state = mps_chain.sites.copy()
    if defect == 2:
        mps_chain = MPS(L=args["L"],d=args["d"],model=args["model"], chi=1)
        neel_state = neel_prod_state(args["half_chain_length"])
        init_state = neel_state + [hole_tn] + [hole_tn] + neel_state
        mps_chain.sites = init_state.copy()
    return mps_chain, init_state

def main():
    delta = args.tf / args.trott
    
    hole_tn = np.array([0,1,0]).reshape((1,3,1))
    
    for L in args.Ls:
        half_chain_length = L // 2 - args.defects//2
        
        args_mps = {
            "L": L,
            "d": args.d,
            "chi": chi,
            "model": args.model,
            "path": path_tensor,
            "precision": args.precision,
            "Jz": args.Jz,
            "J_perp": args.J_perp,
            "t": args.t,
            "tp": args.tp,
            "half_chain_length": half_chain_length
        }
        mps_chain, init_state = initial_state(args.defects, args_mps)

        mpo_i_ip1_eo, mpo_i_ip1_oe = mpo_ev_trotter_i_ip1_pipeline(L, args.Jz, args.J_perp, args.t_up, args.t_down, args.V, delta)
        mpo_i_ip2_delta_half, mpo_i_ip2_delta = mpo_ev_trotter_i_ip2_pipeline(L, args.tp_up, args.tp_down, delta)
        # tensor_shapes(mpo_i_ip1_eo)
        # tensor_shapes(mpo_i_ip1_oe)
        # tensor_shapes(mpo_i_ip2_delta_half)
        # tensor_shapes(mpo_i_ip2_delta)
        mps_chain.w_dag = {
            "i,i+1 eo interaction delta/2": mpo_i_ip1_eo.copy(), 
            "i,i+2 1st interaction delta/2": mpo_i_ip2_delta_half.copy(), 
            "i,i+2 interaction delta": mpo_i_ip2_delta.copy(), 
            "i,i+2 2nd interaction delta/2": mpo_i_ip2_delta_half.copy(), 
            "i,i+1 oe interaction delta/2": mpo_i_ip1_oe.copy(),
                           }
        print(len(mps_chain.w_dag))

        for chi in args.chis:
            date_start = dt.datetime.now()
            print("--------------")
            print(f"chi: {chi}")
            print("--------------")
            (
            errs,
            entrs,
            svs,
            local_magnetization,
            ovlps,
            chi_sat,
                    ) = mps_chain.TEBD_variational_tJ(trotter_steps=args.trott, 
                                                      where=mps_chain.L//2, 
                                                      chi_max=chi, 
                                                      obs=['lh'], 
                                                      obs_freq=1)
            
            np.save(f"{path_tensor}/results/error_data/time_ev_errors_L_{L}_tj_model_delta_{delta}_chi_{chi}.npy", errs)
            np.save(f"{path_tensor}/results/entropy_data/time_ev_entropy_L_{L}_tj_model_delta_{delta}_chi_{chi}.npy", entrs)
            np.save(f"{path_tensor}/results/mag_data/time_ev_hole_occup_L_{L}_tj_model_delta_{delta}_chi_{chi}.npy", local_magnetization)
            np.save(f"{path_tensor}/results/svs_data/time_ev_svs_L_{L}_tj_model_delta_{delta}_chi_{chi}.npy", svs)
            
            mps_chain.sites = init_state.copy()
            t_final = dt.datetime.now() - date_start
            print(f"time of the whole search for chi={chi} is: {t_final}")

if __name__ == "__main__":
    main()
    