import numpy as np
import argparse
import scipy.sparse.linalg as spla
from scipy.sparse import csc_array
from scipy.sparse.linalg import LinearOperator

from qs_mps.applications.Z2.exact_hamiltonian import *
from qs_mps.sparse_hamiltonians_and_operators import diagonalization
from qs_mps.mps_class import MPS

parser = argparse.ArgumentParser(prog="gs_search_Z2")
parser.add_argument("N", help="Number of ladders in the direct lattice", type=int)
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
    "-L", "--long", help="Number of rungs per ladder", type=int)
parser.add_argument(
    "-D", "--chis", help="Simulated bond dimensions", nargs="+", type=int
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
parser.add_argument(
    "-R",
    "--length",
    help="String length in the two particle sector. By default 0 means we are in the vacuum",
    default=0,
    type=int,
)
parser.add_argument(
    "-s",
    "--sites",
    help="Number of sites we want to iterate in the transfer matrix. By default 10",
    default=10,
    type=int,
)
parser.add_argument(
    "-m", "--model", help="Model to simulate", default="Z2_dual", type=str
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
    "--precision",
    help="Precision to load and save tensors and observables. By default True 3",
    default=3,
    type=int,
)
parser.add_argument(
    "-v",
    "--save",
    help="Save the tensors. By default True",
    action="store_false",
)

args = parser.parse_args()

# # Redirect stdout and stderr to the log file
# sys.stdout = open(f'results/logs/{args.logging}', 'w')
# sys.stderr = sys.stdout


# define the physical dimension
d = int(2 ** (args.N))

# define the interval of equally spaced values of external field
if args.interval == "lin":
    interval = np.linspace(args.h_i, args.h_f, args.npoints)
    # num = (interval[-1] - interval[0]) / args.npoints
    # precision = get_precision(num)
    
elif args.interval == "log":
    interval = np.logspace(args.h_i, args.h_f, args.npoints)
    # precision = int(np.max([np.abs(args.h_f), np.abs(args.h_i)]))

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
    path_tensor = "/Users/fradm/Desktop/projects/1_Z2"
    parent_path = path_tensor
else:
    raise SyntaxError("Path not valid. Choose among 'pc', 'mac', 'marcos'")


def multi_site_mps_transfer_matrix(sites, mps_tensor: MPS=None, mps_tm: np.ndarray=None, linop: bool=False):
    tensors_idxs = [mps_tensor.L//2-sites//2+i for i in range(sites)]

    if mps_tm is None:
        mps_tm = ncon([mps_tensor.sites[tensors_idxs[0]].conjugate(), mps_tensor.sites[tensors_idxs[0]]], [[-1,1,-3],[-2,1,-4]])
    else:
        mps_tm = mps_tm.reshape((mps_tensor.sites[tensors_idxs[0]].shape[0],mps_tensor.sites[tensors_idxs[0]].shape[0],mps_tensor.sites[tensors_idxs[-1]].shape[2],mps_tensor.sites[tensors_idxs[-1]].shape[2]))
        if (sites % 2) == 0:
            mps_tm = ncon([mps_tensor.sites[tensors_idxs[0]].conjugate(), mps_tm], [[-1,-2,1], [1,-3,-4,-5]])
            mps_tm = ncon([mps_tensor.sites[tensors_idxs[0]], mps_tm], [[-2,1,2],[-1,1,2,-3,-4]])
        elif (sites % 2) == 1:
            mps_tm = ncon([mps_tm, mps_tensor.sites[tensors_idxs[-1]].conjugate()], [[-1,-2,1,-5],[1,-4,-3]])
            mps_tm = ncon([mps_tm, mps_tensor.sites[tensors_idxs[-1]]], [[-1,-2,-3,1,2],[2,1,-4]])

    transfer_matrix = mps_tm.reshape((mps_tensor.sites[tensors_idxs[0]].shape[0]**2,mps_tensor.sites[tensors_idxs[-1]].shape[2]**2))
    if linop:
        return mps_tm
    return transfer_matrix

def get_tm_eigs(mps_tm):
    D = mps_tm.shape[0]
    rng = np.random.default_rng(42)  # optional for reproducibility
    v0 = rng.random(D*D) + 1j * rng.random(D*D)
    v0 /= np.linalg.norm(v0)

    def matvec(v):
        vec_eff = ncon([mps_tm, v.reshape(D, D)], [[-1,-2,1,2],[1,2]]).reshape(D*D)
        return vec_eff

    A = LinearOperator(
        (D*D, D*D),
        matvec=matvec,
        dtype=np.complex128,
    )

    e = spla.eigsh(A, k=2, v0=v0, which="LM", return_eigenvectors=False)
    return e



for chi in args.chis:
    if args.charges_x == [] and args.charges_y == []:
        sector = "vacuum_sector"
        charges_x = np.nan
        charges_y = np.nan
    else:
        sector = f"{len(args.charges_x)}_particle(s)_sector"
        charges_x = args.charges_x
        charges_y = args.charges_y

    if args.length != 0:
        charges_x = get_cx(args.long, args.length)
        charges_y = get_cy(args.N, args.boundcond, args.charges_y, R=args.length)
        sector = f"{len(charges_x)}_particle(s)_sector"

    linop = True
    interval = interval.tolist()

    if args.length == 0:
        sector = "vacuum_sector"
        try:
            e0_mps = np.load(f"{path_tensor}/results/energy_data/energy_{args.model}_direct_lattice_{args.N}x{args.long}_{sector}_bc_{args.boundcond}_nan-nan_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy")
            e1_mps = np.load(f"{path_tensor}/results/energy_data/first_excited_energy_{args.model}_direct_lattice_{args.N}x{args.long}_{sector}_bc_{args.boundcond}_nan-nan_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy")
        except:
            e0_mps = np.load(f"{path_tensor}/results/energy_data/energy_{args.model}_direct_lattice_{args.N}x{args.long}_{sector}_bc_{args.boundcond}_None-None_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy")
            e1_mps = np.load(f"{path_tensor}/results/energy_data/first_excited_energy_{args.model}_direct_lattice_{args.N}x{args.long}_{sector}_bc_{args.boundcond}_None-None_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy")
    else:
        sector = "2_particle(s)_sector"
        e0_mps = np.load(f"{path_tensor}/results/energy_data/energy_{args.model}_direct_lattice_{args.N}x{args.long}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy")
        e1_mps = np.load(f"{path_tensor}/results/energy_data/first_excited_energy_{args.model}_direct_lattice_{args.N}x{args.long}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}.npy")
        
    velocities = []
    for g in interval:
        lattice = MPS(L=args.long, d=d, model=args.model, chi=chi, h=g, bc=args.boundcond)
        if args.length != 0:
            lattice.Z2.add_charges(charges_x,charges_y)
            lattice.Z2._define_sector()
        lattice.load_sites(path_tensor, precision=args.precision, cx=charges_x, cy=charges_y)

        energies = []
        tm = None
        for sites in range(1,args.sites+1):
            print(f"computing a {sites} site(s) transfer matrix...")
            tm = multi_site_mps_transfer_matrix(sites, mps_tensor=lattice, mps_tm=tm, linop=linop)
            print(f"transfer matrix found. Shape is {tm.shape}")
            if linop:
                e1 = get_tm_eigs(tm)
            else:
                e1, v1 = diagonalization(tm, sparse=True, k=2, which='LA')
            energies.append(e1)
        
        if linop:
            energies = [np.sort(e1)[::-1] for e1 in energies]

        corr_lens = np.array([-(i+1)/np.log(np.abs(np.asarray(energies)[i,1])) for i in range(len(energies))])

        idx = interval.index(g)
        vs = (e1_mps - e0_mps)[idx] * corr_lens
        velocities.append(vs)

    velocities = np.asarray(velocities).T

    if args.save:
        np.save(
                f"{parent_path}/results/energy_data/speed_of_sound_tm_sites_{args.sites}_{args.model}_direct_lattice_{args.N}x{args.long}_{sector}_bc_{args.boundcond}_{charges_x}-{charges_y}_h_{args.h_i}-{args.h_f}_delta_{args.npoints}_chi_{chi}",
                velocities,
        )