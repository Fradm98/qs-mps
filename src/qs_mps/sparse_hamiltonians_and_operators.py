import numpy as np

import scipy.linalg as la
import scipy.sparse.linalg as spla

from .utils import get_middle_chain_schmidt_values, von_neumann_entropy, mps_to_vector
from scipy.sparse import identity, csc_array, csc_matrix


# -----------------------------------------------
# Sparse Non-Diagonal Pauli Indices
# -----------------------------------------------
def sparse_non_diag_paulis_indices(n: int, N: int):
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


# ---------------------------------------------------------------------------------------
# Sparse Pauli X
# ---------------------------------------------------------------------------------------
def sparse_pauli_x(
    n: int,
    L: int,
    row_indices_cache: np.ndarray = None,
    col_indices_cache: np.ndarray = None,
):
    """
    Returns a CSC sparse matrix representation of the pauli_x matrix acting over qubit n in a Hilbert space of L qubits
    0 <= n < L

    """
    if 0 <= n < L:
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, L)
        data = np.ones_like(row_indices_cache)
        result = csc_array(
            (data, (row_indices_cache, col_indices_cache)), shape=(2**L, 2**L)
        )  # , dtype=complex
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < L")


# ---------------------------------------------------------------------------------------
# Sparse Pauli Y
# ---------------------------------------------------------------------------------------
def sparse_pauli_y(
    n: int,
    L: int,
    row_indices_cache: np.ndarray = None,
    col_indices_cache: np.ndarray = None,
):
    """
    Returns a CSC sparse matrix representation of the pauli_y matrix acting over qubit n in a Hilbert space of L qubits
    0 <= n < L

    """
    if 0 <= n < L:
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, L)
        data = -1j * np.ones_like(row_indices_cache)
        data[len(data) // 2 : :] = 1j
        result = csc_array(
            (data, (row_indices_cache, col_indices_cache)),
            shape=(2**L, 2**L),
            dtype=complex,
        )
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < L")


# ---------------------------------------------------------------------------------------
# Sparse Pauli Z
# ---------------------------------------------------------------------------------------
def sparse_pauli_z(n: int, L: int):
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
        result = csc_array(
            (diag, (row_col_indices, row_col_indices)),
            shape=(2**L, 2**L),
            dtype=complex,
        )
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < L")


# ---------------------------------------------------------------------------------------
# Sparse Magnetization
# ---------------------------------------------------------------------------------------
def sparse_magnetization(L, op="X", staggered: bool = False):
    if op == "X":
        ope = sparse_pauli_x
    elif op == "Z":
        ope = sparse_pauli_z

    O = csc_array((2**L, 2**L), dtype=complex)
    m = O
    c = [1 for _ in range(L)]
    if staggered:
        c = [(-1) ** (i // 2) for i in range(L)]
    if op == "X":
        for i in range(L):
            n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(i, L)
            m += c[i] * ope(i, L, n_row_indices, n_col_indices)
    elif op == "Z":
        for i in range(L):
            m += ope(i,L)
    return m / L


# -----------------------------------------------
# Sparse Ising Hamiltonian
# -----------------------------------------------
def sparse_ising_hamiltonian(J: float, h_t: float, h_l: float, L: int, long: str = "X", deg: int=1):
    """
    Returns a sparse representation of the Hamiltonian of the 1D Ising model in a chain of length L
    with open boundary conditions
    J < 0: Antiferromagnetic case (Unique ground state of total angular momentum S=0)
    J > 0: Ferromagnetic case (2-fold degeneracy of the ground state of angular momentum S=L/2)
    """
    hamiltonian_l = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_t = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_int = csc_array((2**L, 2**L), dtype=complex)

    if long == "X":
        # First sum over the terms containing sigma_x, sigma_y because the non-zero element indices are the same
        # so that this improves performance
        if h_t != 0:
            for n in range(L):
                n_pauli_z = sparse_pauli_z(n, L)
                hamiltonian_t += n_pauli_z
        if h_l != 0:
            for n in range(L):
                n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
                n_pauli_x = sparse_pauli_x(n, L, n_row_indices, n_col_indices)
                hamiltonian_l += n_pauli_x

        # Sum over sigma_z terms
        for n in range(L - 1):
            n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
            np1_row_indices, np1_col_indices = sparse_non_diag_paulis_indices(n + 1, L)
            n_pauli_x = sparse_pauli_x(n, L, n_row_indices, n_col_indices)
            np1_pauli_x = sparse_pauli_x(n + 1, L, np1_row_indices, np1_col_indices)
            hamiltonian_int += n_pauli_x @ np1_pauli_x

    if long == "Z":
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

        # Interaction
        for n in range(L - 1):
            n_pauli_z = sparse_pauli_z(n, L)
            np1_pauli_z = sparse_pauli_z(n + 1, L)
            hamiltonian_int += n_pauli_z @ np1_pauli_z

    if deg == 1:
        return -J * hamiltonian_int - h_t * hamiltonian_t - h_l * hamiltonian_l
    if deg == 2:
        return -J * hamiltonian_int - h_t * hamiltonian_t - h_l * (hamiltonian_l - L*identity(2**L))


# -----------------------------------------------
# Sparse ANNNI Hamiltonian
# -----------------------------------------------
def sparse_ANNNI_hamiltonian(
    J: float,
    h_t: float,
    h_ll: float,
    L: int,
    eps: float = 1e-5,
    long: str = "X",
    deg_method: int = 1,
):
    """
    sparse_ANNNI_hamiltonian

    This function gives a representation of the 1D Axial Next Nearest Neighbor Interaction model.
    The next nearest neighbor interaction (h_ll) is competing with the nearest neighbor interaction (J)
    We use eps to break the degeneracy for small transverse field h_t.

    """
    hamiltonian_ll = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_deg = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_t = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_int = csc_array((2**L, 2**L), dtype=complex)
    if long == "X":
        if h_t != 0:
            # transverse field
            for n in range(L):
                n_pauli_z = sparse_pauli_z(n, L)
                hamiltonian_t += n_pauli_z
        if h_ll != 0:
            # next nearest neighbor interaction
            for n in range(L - 2):
                n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
                np2_row_indices, np2_col_indices = sparse_non_diag_paulis_indices(
                    n + 2, L
                )
                n_pauli_x = sparse_pauli_x(n, L, n_row_indices, n_col_indices)
                np2_pauli_x = sparse_pauli_x(n + 2, L, np2_row_indices, np2_col_indices)
                hamiltonian_ll += n_pauli_x @ np2_pauli_x

        # nearest neighbor interaction
        for n in range(L - 1):
            n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
            np1_row_indices, np1_col_indices = sparse_non_diag_paulis_indices(n + 1, L)
            n_pauli_x = sparse_pauli_x(n, L, n_row_indices, n_col_indices)
            np1_pauli_x = sparse_pauli_x(n + 1, L, np1_row_indices, np1_col_indices)
            hamiltonian_int += n_pauli_x @ np1_pauli_x
        if eps != 0:
            # add a term to break the double degeneracy of the ground state
            if deg_method == 0:
                hamiltonian_deg = 0
            elif deg_method == 1:
                hamiltonian_deg = sparse_pauli_x(n=0, L=L) - identity(2**L)
            elif deg_method == 2:
                for n in range(L):
                    hamiltonian_deg += (1 + (-1) ** (n // 2)) * sparse_pauli_x(
                        n=n, L=L
                    ) - 2 * identity(2**L)
            elif deg_method == 3:
                for n in range(L):
                    hamiltonian_deg += (-1) ** (n // 2) * sparse_pauli_x(
                        n=n, L=L
                    ) - identity(2**L)
            else:
                raise ValueError("Choose a proper degeneracy method")
    return (
        -J * hamiltonian_int
        + h_ll * hamiltonian_ll
        - h_t * hamiltonian_t
        - eps * hamiltonian_deg
    )


# -----------------------------------------------
# Sparse Cluster Hamiltonian
# -----------------------------------------------
def sparse_cluster_hamiltonian(
    J: float, h_t: float, L: int, eps: float = 1e-5, long: str = "X"
):
    """
    Returns a sparse representation of the Hamiltonian of the 1D Cluster model in a chain of length L
    with open boundary conditions
    J < 0: Antiferromagnetic case (Unique ground state of total angular momentum S=0)
    J > 0: Ferromagnetic case (L+1-fold degeneracy of the ground state of angular momentum L/2) -> Dicke states for even L
    """
    hamiltonian_t = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_deg = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_int = csc_array((2**L, 2**L), dtype=complex)

    if long == "X":
        # First sum over the terms containing sigma_x, sigma_y because the non-zero element indices are the same
        # so that this improves performance
        if h_t != 0:
            for n in range(L):
                n_pauli_z = sparse_pauli_z(n, L)
                hamiltonian_t += n_pauli_z
        if eps != 0:
            for n in range(L):
                n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
                n_pauli_x = sparse_pauli_x(n, L, n_row_indices, n_col_indices)
                hamiltonian_deg += n_pauli_x

        # Interaction
        for n in range(L - 2):
            n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
            np2_row_indices, np2_col_indices = sparse_non_diag_paulis_indices(n + 2, L)
            n_pauli_x = sparse_pauli_x(n, L, n_row_indices, n_col_indices)
            np1_pauli_z = sparse_pauli_z(n + 1, L)
            np2_pauli_x = sparse_pauli_x(n + 2, L, np2_row_indices, np2_col_indices)
            hamiltonian_int += n_pauli_x @ np1_pauli_z @ np2_pauli_x

    return -J * hamiltonian_int - h_t * hamiltonian_t - eps * hamiltonian_deg


# -----------------------------------------------
# Sparse Cluster-XY Hamiltonian
# -----------------------------------------------
def sparse_cluster_xy_hamiltonian(
    J: float,
    h_t: float,
    h_x: float,
    h_y: float,
    L: int,
    eps: float = 1e-5,
    long: str = "X",
):
    """
    Returns a sparse representation of the Hamiltonian of the 1D Cluster-XY model in a chain of length L
    with open boundary conditions
    J < 0: Antiferromagnetic case (Unique ground state of total angular momentum S=0)
    J > 0: Ferromagnetic case (L+1-fold degeneracy of the ground state of angular momentum L/2) -> Dicke states for even L
    """
    hamiltonian_t = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_x = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_y = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_deg = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_int = csc_array((2**L, 2**L), dtype=complex)

    if long == "X":
        # First sum over the terms containing sigma_x, sigma_y because the non-zero element indices are the same
        # so that this improves performance
        if h_t != 0:
            for n in range(L):
                n_pauli_z = sparse_pauli_z(n, L)
                hamiltonian_t += n_pauli_z
        if h_x != 0:
            for n in range(L - 1):
                n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
                n_pauli_x = sparse_pauli_x(n, L, n_row_indices, n_col_indices)
                hamiltonian_x += n_pauli_x
        if h_y != 0:
            for n in range(L - 1):
                n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
                n_pauli_y = sparse_pauli_y(n, L, n_row_indices, n_col_indices)
                hamiltonian_y += n_pauli_y
        if eps != 0:
            for n in range(L):
                n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
                n_pauli_x = sparse_pauli_x(n, L, n_row_indices, n_col_indices)
                hamiltonian_deg += n_pauli_x

        # Interaction
        for n in range(L - 2):
            n_row_indices, n_col_indices = sparse_non_diag_paulis_indices(n, L)
            np2_row_indices, np2_col_indices = sparse_non_diag_paulis_indices(n + 2, L)
            n_pauli_x = sparse_pauli_x(n, L, n_row_indices, n_col_indices)
            np1_pauli_z = sparse_pauli_z(n + 1, L)
            np2_pauli_x = sparse_pauli_x(n + 2, L, np2_row_indices, np2_col_indices)
            hamiltonian_int += n_pauli_x @ np1_pauli_z @ np2_pauli_x

    return (
        -J * hamiltonian_int
        - h_t * hamiltonian_t
        - eps * hamiltonian_deg
        + h_x * hamiltonian_x
        + h_y * hamiltonian_y
    )


def sparse_Z2_electric_dual_ham(l: int, L: int, cx: list=None, cy: list=None):
    dof = l*L + 1
    O = csc_array((2**dof, 2**dof), dtype=complex)
    H_sigmas = O
    H_taus = O

    # charges
    charges = np.ones((l,L+1))
    if cx is not None:
        charges[cy[0],cx[0]] = -1
        charges[cy[1],cx[1]] = -1

    prod_charges = np.prod(charges, axis=1).tolist()

    # first column of sigmas are local
    for j in range(l):
        H_sigmas += sparse_pauli_z(n=j, L=dof)
    
    # horizontal zz interactions, we exclude the last column of sigmas
    for i in range(L-1):
        for j in range(l):
            H_sigmas += sparse_pauli_z(n=j+i*l, L=dof) @ sparse_pauli_z(n=j+(i+1)*l, L=dof)

    # horizontal zz interactions, last column of sigmas
    for j in range(l):
        # H_sigmas += sparse_pauli_z(n=j*L+L-1, L=dof) @ sparse_pauli_z(n=l*L, L=dof)
        H_sigmas += np.prod(prod_charges[:(j+1)]) * sparse_pauli_z(n=j+(L-1)*l, L=dof) @ sparse_pauli_z(n=l*L, L=dof)

    # vertical zz interactions 
    for i in range(L):
        for j in range(l):
            # coeff = charge_func(i,j,l,cx,cy)
            # H_taus += coeff * sparse_pauli_z(n=j*L+i, L=dof) @ sparse_pauli_z(n=((j+1)%l)*L+i, L=dof)
            H_taus += np.prod(charges[(j+1)%l,:(i+1)]) * sparse_pauli_z(n=j+i*l, L=dof) @ sparse_pauli_z(n=((j+1)%l)+i*l, L=dof)

    return H_sigmas + H_taus

def sparse_Z2_magnetic_dual_ham(l: int, L: int):
    dof = l*L + 1
    O = csc_array((2**dof, 2**dof), dtype=complex)
    H_plaquettes = O

    # plaquette terms
    for i in range(l*L):
        H_plaquettes += sparse_pauli_x(n=i, L=dof)
    return H_plaquettes

def sparse_Z2_dual_ham(l,L,g, cx: list=None, cy: list=None):
    # electric vertical and horizontal terms
    H_el = sparse_Z2_electric_dual_ham(l,L,cx,cy)

    # magnetic plaquette terms
    H_mag = sparse_Z2_magnetic_dual_ham(l,L)

    return - g * H_el - 1/g * H_mag

def electric_field(l: int, L: int, statevector: np.ndarray, cx: list=[], cy: list=[]):
    dof = l*L + 1
    exp_sigmas = []
    exp_taus = []

    # charges
    charges = np.ones((l,L+1))
    if len(cx) != 0:
        charges[cy[0],cx[0]] = -1
        charges[cy[1],cx[1]] = -1
    prod_charges = [1] + np.prod(charges[1:], axis=1).tolist()

    # first column of sigma are local
    for j in range(l):
        # H_sigmas += sparse_pauli_z(n=j*L, L=dof)
        exp_sigmas.append(statevector.T.conjugate() @ sparse_pauli_z(n=j*L, L=dof) @ statevector)
    
    # horizontal zz interactions, we exclude the last column of sigmas
    for j in range(l):
        for i in range(L-1):
            # H_sigmas += sparse_pauli_z(n=i+j*L, L=dof) @ sparse_pauli_z(n=(i+1)+j*L, L=dof)
            exp_sigmas.append(statevector.T.conjugate() @ sparse_pauli_z(n=i+j*L, L=dof) @ sparse_pauli_z(n=(i+1)+j*L, L=dof) @ statevector)

    # horizontal zz interactions, last column of sigmas
    for j in range(l):
        # H_sigmas += np.prod(prod_charges[:(j+1)]) * sparse_pauli_z(n=j*L+L-1, L=dof) @ sparse_pauli_z(n=l*L, L=dof)
        exp_sigmas.append(np.prod(prod_charges[:(j+1)]) * (statevector.T.conjugate() @ sparse_pauli_z(n=j*L+L-1, L=dof) @ sparse_pauli_z(n=l*L, L=dof) @ statevector))

    # vertical zz interactions 
    for i in range(L):
        for j in range(l):
            # H_taus += np.prod(charges[(j+1)%l,:(i+1)]) * sparse_pauli_z(n=j*L+i, L=dof) @ sparse_pauli_z(n=((j+1)%l)*L+i, L=dof)
            exp_taus.append(np.prod(charges[(j+1)%l,:(i+1)]) * (statevector.T.conjugate() @ sparse_pauli_z(n=j*L+i, L=dof) @ sparse_pauli_z(n=((j+1)%l)*L+i, L=dof) @ statevector))
    
    exp_sigmas_l = np.asarray(exp_sigmas[:l]).reshape((l,1))
    exp_sigmas_b = np.asarray(exp_sigmas[l:-l]).reshape((l,(L+1)-2))
    exp_sigmas_r = np.asarray(exp_sigmas[-l:]).reshape((l,1))
    exp_ver = [[exp_sigmas_l[i]]+[exp_sigmas_b[i]]+[exp_sigmas_r[i]] for i in range(l)]
    exp_ver_swaped = []
    for row in exp_ver:
        for elem in row:
            if len(elem) == 1:
                exp_ver_swaped += (elem.real).tolist()
            else:
                exp_ver_swaped += (elem.real).tolist()
    exp_hor = np.array(exp_taus).reshape((L,l))
    exp_hor_swaped = exp_hor.swapaxes(0,1)
    exp_hor_swaped[[0,-1]] = exp_hor_swaped[[-1,0]]
    exp_val = np.zeros((2*l,2*L+1))
    exp_val[:] = np.nan
    exp_val[1::2,::2] = np.asarray(exp_ver_swaped).reshape((l,L+1))
    exp_val[::2,1::2] = np.asarray(exp_hor_swaped).reshape((l,L))
    
    return exp_val

# ---------------------------------------------------------------------------------------
# Diagonalization
# ---------------------------------------------------------------------------------------
def diagonalization(
    H: csc_matrix, sparse: bool, v0: np.ndarray = None, k: int = 1, which: str = "SA", maxiter: int = None
):
    if sparse:
        e, v = spla.eigsh(H, k=k, which=which, v0=v0, maxiter=maxiter)
    else:
        try:
            e, v = la.eigh(H.toarray())
        except:
            e, v = la.eigh(H)
    return e, v


# ---------------------------------------------------------------------------------------
# Sparse Ground state
# ---------------------------------------------------------------------------------------
def sparse_ising_ground_state(
    L: int, h_t: float, h_l: float = 1e-7, J: float = 1, k: int = 1
) -> csc_array:
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
    e, v = diagonalization(H, sparse=True, k=k)
    print(f"first {k} eigenvalue(s) SA (Smallest (algebraic) eigenvalues): {e}")
    psi = v[:, 0]
    return psi


# ---------------------------------------------------------------------------------------
# Sparse U Evolution
# ---------------------------------------------------------------------------------------
def U_evolution_sparse(
    psi_init: csc_array,
    H_ev: csc_array,
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
    psi_ev = spla.expm_multiply(H_ev, psi_init)
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
        psi_exact = csc_array(flip @ psi_exact)
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

def evolution_operator(delta, hamiltonian):
    return spla.expm(-1j*delta*hamiltonian)

def trott_1(delta, coupling_loc, coupling_int, H_loc, H_int):
    U_loc = evolution_operator(delta=delta*(coupling_loc), hamiltonian=H_loc)    
    U_int = evolution_operator(delta=delta*(coupling_int), hamiltonian=H_int)
    return U_loc @ U_int
    
def trott_2(delta, coupling_loc, coupling_int, H_loc, H_int):
    U_loc = evolution_operator(delta=(delta/2)*(coupling_loc), hamiltonian=H_loc)    
    U_int = evolution_operator(delta=delta*(coupling_int), hamiltonian=H_int)
    return U_loc @ U_int @ U_loc

def trott_Z2_dual(l, L, cx, cy, delta, coupling, ord: int=1):
    H_ev_el = sparse_Z2_electric_dual_ham(l, L, cx, cy) # zz dual interaction term
    H_ev_mag = sparse_Z2_magnetic_dual_ham(l, L) # x dual local term
    if ord == 1:
        U_ev = trott_1(delta, coupling_loc=1/coupling, coupling_int=coupling, H_loc=H_ev_mag, H_int=H_ev_el)
    if ord == 2:
        U_ev = trott_2(delta, coupling_loc=1/coupling, coupling_int=coupling, H_loc=H_ev_mag, H_int=H_ev_el)
    return U_ev

def trott_ising(L, J, h, delta, ord: int=1):
    H_int = sparse_ising_hamiltonian(J=J, h_t=0, h_l=0, L=L, long="Z") # zz interaction term
    H_loc = sparse_ising_hamiltonian(J=0, h_t=h, h_l=0, L=L, long="Z") # x local term
    if ord == 1:
        U_ev = trott_1(delta, coupling_loc=h, coupling_int=J, H_loc=H_loc, H_int=H_int)
    if ord == 2:
        U_ev = trott_2(delta, coupling_loc=h, coupling_int=J, H_loc=H_loc, H_int=H_int)
    return U_ev