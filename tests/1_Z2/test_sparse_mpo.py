from scipy.sparse import csc_array, identity
from ncon import ncon
from qs_mps.exact_Ising_ground_state_and_time_evolution import sparse_pauli_x, sparse_pauli_z
import numpy as np

def mpo_skeleton(l):
    I = identity(2**l)
    O = csc_array((2**l,2**l))
    skeleton = np.tile(O, (2+l,2+l))
    pass


