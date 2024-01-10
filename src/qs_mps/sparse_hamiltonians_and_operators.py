import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, expm_multiply
from .utils import get_middle_chain_schmidt_values, von_neumann_entropy, mps_to_vector


# -----------------------------------------------
# Sparse Pauli matrices
# -----------------------------------------------
def sparse_non_diag_paulis_indices(n, N):
    """
    Returns a tuple (row_indices, col_indices) containing the row and col indices of the non_zero elements
    of the tensor product of a non diagonal pauli matrix (x, y) acting over a single qubit in a Hilbert
    space of N qubits

    """
    if 0 <= n < N:
        block_length = 2 ** (N - n - 1)
        nblocks = 2**n
        ndiag_elements = block_length * nblocks
        k = np.arange(ndiag_elements, dtype=int)
        red_row_col_ind = (k % block_length) + 2 * (k // block_length) * block_length
        upper_diag_row_indices = red_row_col_ind
        upper_diag_col_indices = block_length + red_row_col_ind
        row_indices = np.concatenate((upper_diag_row_indices, upper_diag_col_indices))
        col_indices = np.concatenate((upper_diag_col_indices, upper_diag_row_indices))
        return row_indices, col_indices
    else:
        raise ValueError("Index n must fulfill 0 <= n < N")


def sparse_pauli_x(n, L, row_indices_cache=None, col_indices_cache=None):
    """
    Returns a CSC sparse matrix representation of the pauli_x matrix acting over qubit n in a Hilbert space of L qubits
    0 <= n < L

    """
    if 0 <= n < L:
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, L)
        data = np.ones_like(row_indices_cache)
        result = sparse.csc_array(
            (data, (row_indices_cache, col_indices_cache)), shape=(2**L, 2**L)
        )  # , dtype=complex
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < L")


def sparse_pauli_y(n, L, row_indices_cache=None, col_indices_cache=None):
    """
    Returns a CSC sparse matrix representation of the pauli_y matrix acting over qubit n in a Hilbert space of L qubits
    0 <= n < L

    """
    if 0 <= n < L:
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, L)
        data = -1j * np.ones_like(row_indices_cache)
        data[len(data) // 2 : :] = 1j
        result = sparse.csc_array(
            (data, (row_indices_cache, col_indices_cache)),
            shape=(2**L, 2**L),
            dtype=complex,
        )
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < L")


def sparse_pauli_z(n, L):
    """
    Returns a CSC sparse matrix representation of the pauli_z matrix acting over qubit n in a Hilbert space of L qubits
    0 <= n < L

    """
    if 0 <= n < L:
        block_length = 2 ** (L - n)
        nblocks = 2**n
        block = np.ones(block_length, dtype=int)
        block[block_length // 2 : :] = -1
        diag = np.tile(block, nblocks)
        row_col_indices = np.arange(2**L, dtype=int)
        result = sparse.csc_array(
            (diag, (row_col_indices, row_col_indices)),
            shape=(2**L, 2**L),
            dtype=complex,
        )
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < L")


# -----------------------------------------------
# Sparse Ising Hamiltonian
# -----------------------------------------------
def sparse_ising_hamiltonian(J, h_t, h_l, L):
    """
    Returns a sparse representation of the Hamiltonian of the 1D Heisemberg model in a chain of length L
    with periodic boundary conditions (hbar = 1)
    J < 0: Antiferromagnetic case (Unique ground state of total angular momentum S=0)
    J > 0: Ferromagnetic case (L+1-fold degeneracy of the ground state of angular momentum L/2) -> Dicke states for even L
    """
    hamiltonian_l = sparse.csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_t = sparse.csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_int = sparse.csc_array((2**L, 2**L), dtype=complex)

    # First sum over the terms containing sigma_x, sigma_y because the non-zero element indices are the same
    # so that this improves performance
    if h_t != 0:
        for n in range(L):
            n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
            n_pauli_x = sparse_pauli_x(n, L, n_row_indices, n_col_indices)
            hamiltonian_t += n_pauli_x
    if h_l != 0:
        for n in range(L):
            n_pauli_z = sparse_pauli_z(n, L)
            hamiltonian_l += n_pauli_z

    # Sum over sigma_z terms

    for n in range(L - 1):
        n_pauli_z = sparse_pauli_z(n, L)
        np1_pauli_z = sparse_pauli_z(n + 1, L)
        hamiltonian_int += n_pauli_z @ np1_pauli_z

    return -J * hamiltonian_int - h_t * hamiltonian_t - h_l * hamiltonian_l


# ---------------------------------------------------------------------------------------
# Sparse Ground state
# ---------------------------------------------------------------------------------------
def sparse_ising_ground_state(
    L: int, h_t: float, h_l: float = 1e-7, J: float = 1, k: int = 1
) -> sparse.csc_array:
    """
    exact_initial_state

    This function is computing the initial state given by an Ising Hamiltonian.

    L: int - chain size
    h_t: float - initial transverse field parameter
    h_l: float - initial longitudinal field parameter
    k: int - number of eigenvalues we want to compute. By default 1

    """
    print("Finding the Hamiltonian...")
    H = sparse_ising_hamiltonian(J=J, h_t=h_t, h_l=h_l, L=L)
    print("Hamiltonian found")
    e, v = eigsh(H, k=k, which="SA")
    print(f"first {k} eigenvalue(s) SA (Smallest (algebraic) eigenvalues): {e}")
    psi = v[:, 0]
    return psi


# ---------------------------------------------------------------------------------------
# Sparse U Evolution
# ---------------------------------------------------------------------------------------
def U_evolution_sparse(
    psi_init: sparse.csc_array,
    H_ev: sparse.csc_array,
    trotter: int,
    time: float,
):
    """
    U_evolution

    This function applies a time evolution operator to some initial state.
    The evolution operator uses the Ising hamiltonian with some tunable parameters.

    psi_init: csc_array - initial state to be evolved
    H_ev: csc_array - Time evolution Hamiltonian
    trotter: int - the trotter step we are during the evolution
    time: float - indicates the final time we want to reach

    """
    delta = time / trotter
    H_ev = -1j * delta * H_ev
    psi_ev = expm_multiply(H_ev, psi_init)
    return psi_ev


# ---------------------------------------------------------------------------------------
# Sparse exact Evolution
# ---------------------------------------------------------------------------------------
def exact_evolution_sparse(
    L: int,
    h_t: float,
    h_ev: float,
    time: float,
    trotter_steps: int,
    h_l: float = 1e-7,
    flip: bool = False,
    where: int = -1,
    bond: bool = True,
    model: str = None,
):
    """
    exact_evolution

    This function evolves an initial state for trotter steps times.
    We extract at each time step the local and total magnetization.

    L: int - chain size
    trotter_step: int - indicates the specific trotter step we are evolving
    time: float - indicates the final time we want to reach
    h_t: float - initial transverse field parameter
    h_l: float - initial longitudinal field parameter
    h_ev: float - evolution transverse field parameter
    where: int - in which bond we perform the Schmidt decompositon
    bond: bool - if we compute the entropy for all the bonds. By default False

    """
    if bond:
        entropy = [0]
    else:
        entropy = [0] * (L - 3)

    # psi_exact = sparse_ising_ground_state(L=L, h_t=h_t, h_l=h_l, k=6).reshape(2**L, 1)
    # for now only all spin up
    # psi_exact = np.zeros((2**L, 1))
    # psi_exact[0] = 1
    init_state = np.zeros((1, 2, 1))
    init_state[0, 0, 0] = 1
    mps = [init_state for _ in range(L)]
    psi_exact = mps_to_vector(mps).reshape(2**L, 1)
    if flip:
        flip = sparse_pauli_x(n=L // 2, L=L)
        psi_exact = sparse.csc_array(flip @ psi_exact)
    # local Z
    mag_loc_Z_op = [sparse_pauli_z(n=i, L=L) for i in range(L)]
    # local X
    mag_loc_X_op = sparse_pauli_x(n=L // 2, L=L)
    # total
    mag_tot_op = sparse_ising_hamiltonian(L=L, h_t=0, h_l=-1, J=0)

    mag_exact_loc_Z = []
    mag_exact_loc_X = []
    mag_exact_tot = []

    # local Z
    mag_exact = []
    for i in range(L):
        mag_exact.append(
            (psi_exact.T.conjugate() @ mag_loc_Z_op[i] @ psi_exact)[0, 0].real
        )
    mag_exact_loc_Z.append(mag_exact)

    # local X
    mag_exact_loc_X.append(
        (psi_exact.T.conjugate() @ mag_loc_X_op @ psi_exact)[0, 0].real
    )

    # total
    mag_exact_tot.append((psi_exact.T.conjugate() @ mag_tot_op @ psi_exact)[0, 0].real)

    H_ev = sparse_ising_hamiltonian(L=L, J=1, h_l=h_l, h_t=h_ev)

    entropy_tot = []
    entropy_tot.append(entropy)

    psi_new = psi_exact
    for trott in range(trotter_steps):
        print(f"-------- Trotter step {trott} ---------")
        # exact
        psi_new = U_evolution_sparse(
            psi_init=psi_new, H_ev=H_ev, trotter=trotter_steps, time=time
        )
        # entropy
        sing_vals = get_middle_chain_schmidt_values(psi_new, where=where, bond=bond)
        entropy = []
        for s in sing_vals:
            ent = von_neumann_entropy(s)
            entropy.append(ent)
        entropy_tot.append(entropy)
        # local Z
        mag_exact = []
        for i in range(L):
            mag_exact.append(
                (psi_new.T.conjugate() @ mag_loc_Z_op[i] @ psi_new)[0, 0].real
            )
        mag_exact_loc_Z.append(mag_exact)

        # local X
        mag_exact_loc_X.append(
            (psi_new.T.conjugate() @ mag_loc_X_op @ psi_new)[0, 0].real
        )

        # total
        mag_exact_tot.append((psi_new.T.conjugate() @ mag_tot_op @ psi_new)[0, 0].real)
    return psi_new, mag_exact_loc_Z, mag_exact_loc_X, mag_exact_tot, entropy_tot


# psi_new, mag_exact_loc, mag_exact_loc_X, mag_exact_tot, entropy_tot = exact_evolution_sparse(L=15, h_t=0, h_ev=0.3, time=10, trotter_steps=5, flip=True, where=7)
