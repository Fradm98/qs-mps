import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse.linalg import expm, eigsh, expm_multiply
from scipy.sparse import csr_matrix, csc_matrix, csc_array, kron as spkron
import os
from ncon import ncon
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Union


# ---------------------------------------------------------------------------------------
# Tensor shapes
# ---------------------------------------------------------------------------------------
def tensor_shapes(lists):
    """
    tensor_shapes

    This function returns the shapes of the tensors
    present in a given list.

    lists: list - list of tensors, e.g. the mps

    """
    shapes = [array.shape for array in lists]
    for i in range(len(lists)):
        print(lists[i].shape)

    return shapes


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
"""
Saving and loading tools
"""
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Get labels
# ---------------------------------------------------------------------------------------
def get_labels(shapes):
    """
    get_labels

    This function computes the labels to split a flat list
    in arrays (still flat) corresponding to the original tensors.
    In order to do so it is sufficient to know how many elements
    where present in the original tensors. This information is given
    by the product of the shape of each tensor which is saved in shapes.
    Once we know the elements in one tensor we sum the product of
    shapes to get the correct index in the list.

    shapes: list - list of shapes of the tensors of, e.g., a mps.
    """
    labels = []
    label = 0
    for shape in shapes:
        label += np.prod(shape)
        labels.append(label)
    return labels


# ---------------------------------------------------------------------------------------
# Renaming
# ---------------------------------------------------------------------------------------
def renaming(folder, hs):
    """
    renaming

    This function can be used to rename files in a certain
    folder. If the file is initially saved by including in its name
    a float number which is not automatically approximated correctly,
    we can rename it by giving that array of floating numbers.

    folder: string - path or folder where the files have to be loaded
    hs: np.ndarray - array of floating numbers which were not saved correctly

    """
    # e.g. of folder
    # folder = "/data/fdimarca/tensor_network/bonds_data/"
    # array which is saved in the files with extra zeros
    # hs = np.arange(0.8,1.2,0.01)

    # Iterate
    for file in os.listdir(folder):
        # Taking the old name
        oldName = os.path.join(folder, file)
        # We split the filename where the point is
        n = os.path.splitext(file)[0]
        for h in hs:
            # We take the precision we are interested in saving
            h = f"{h:.2f}"
            # We split the filename where the point is
            h = os.path.splitext(h)[1]
            if h in oldName:
                b = n + h
                newName = os.path.join(folder, b)
                # print(newName)
                # Rename the file
                os.rename(oldName, newName)


# ---------------------------------------------------------------------------------------
# Save list of lists
# ---------------------------------------------------------------------------------------
def save_list_of_lists(file_path, list):
    """
    save_list_of_lists

    This function allows you to save the list of list at the file path specified

    file_path: string - file path
    list: list - list to save

    """
    with open(file_path, "w") as file:
        for sublist in list:
            line = " ".join(map(str, sublist))
            file.write(line + "\n")


# ---------------------------------------------------------------------------------------
# Load list of lists
# ---------------------------------------------------------------------------------------
def load_list_of_lists(file_path):
    """
    load_list_of_lists

    This function allows you to load the list of list from the file path specified

    file_path: string - file path

    """
    loaded_data = []

    # Open the file in read mode and read the data
    with open(file_path, "r") as file:
        for line in file:
            # Split the line into elements (assuming space-separated values)
            elements = line.strip().split()

            # Append the sublist to the loaded_data list
            loaded_data.append(elements)
    return loaded_data


# ---------------------------------------------------------------------------------------
# Access txt
# ---------------------------------------------------------------------------------------
def access_txt(file_path: str, column_index: int):
    """
    access_txt

    This function accesses to .txt files that have
    an equal number of space separated values for each row.
    We can access to one specific column of the .txt file.

    file_path: str - file path
    column_index: int - index of the column we want to retrieve

    """
    # Initialize an empty grid to store the values
    grid = []

    # Open and read the file
    with open(file_path, "r") as file:
        for line in file:
            # Split each line into a list of values, assuming values are separated by spaces
            values = line.strip().split()
            row = [float(value) for value in values]
            # Append the row to the grid
            grid.append(row)

    # Assuming 'grid' contains your data as a list of lists
    # Get the number of rows and columns in the grid
    m = len(grid)  # Number of rows
    n = len(
        grid[0]
    )  # Number of columns (assuming all rows have the same number of columns)

    # Initialize empty lists for each column
    columns = [[] for _ in range(n)]

    # Extract values column-wise
    for row in grid:
        for j in range(n):
            columns[j].append(row[j])

    # Now, 'columns' is a list of lists where columns[j] contains the values of the j-th column.
    column = columns[column_index]
    return column


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
"""
Critical exponent tools
"""
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------------------
def variance(first_m, sm):
    """
    variance

    This function computes the variance of the distribution function
    of a local order parameter, e.g. magnetization, energy.

    first_m: float - it is the first momentum of the, e.g., magnetization
    sm: float - it is the second momentum of the, e.g., magnetization

    """
    return np.abs(sm - first_m**2)


# ---------------------------------------------------------------------------------------
# Binder's Cumulant
# ---------------------------------------------------------------------------------------
def binders_cumul(fourth_m, sm):
    """
    binders_cumul

    This function computes the binders cumulant of the distribution
    function of a local order parameter, e.g. magnetization, energy.

    fourth_m: float - it is the fourth momentum of the, e.g., magnetization
    sm: float - it is the second momentum of the, e.g., the magnetization

    """
    return 1 - fourth_m / (3 * sm**2)


# ---------------------------------------------------------------------------------------
# k values
# ---------------------------------------------------------------------------------------
def k_values(L):
    """
    k_values

    This function computes the k values present in the summation
    of the ground state formula for the PBC of the 1D transverse field Ising model.

    L: int - the number of spins present in the 1D chain

    """
    ks = []
    add = 2 * np.pi / L
    k = -(L - 1) * np.pi / L
    ks.append(k)
    for _ in range(L - 1):
        ks.append(ks[-1] + add)

    return ks


# ---------------------------------------------------------------------------------------
# Ground state
# ---------------------------------------------------------------------------------------
def ground_state(L):
    """
    ground_state

    This function computes the analytical solution of the ground state
    for the 1D transverse field Ising model with Periodic Boundary Conditions (PBC).

    L: int - the number of spins present in the 1D chain

    """
    ks = k_values(L)
    e_0 = []
    for k in ks:
        e_0.append(np.sqrt(2 + 2 * np.cos(k)))

    return -sum(e_0)


# ---------------------------------------------------------------------------------------
# Von Neumann Entropy
# ---------------------------------------------------------------------------------------
def von_neumann_entropy(s):
    """
    von_neumann_entropy

    This function computes the entanglement entropy
    given the Schmidt values of a system.

    s: np.ndarray - array of Schmidt values of a system

    """
    return -np.sum((s**2) * np.log2(s**2))


# ---------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------
def fitting(xs, results, guess):
    """
    fitting

    This function gives you back the parameters and the variance
    in fitting some given data with a function you can define in fit.
    Here we use as an example the function fitting the entanglement
    entropy which is used to extract the correlation length of the system
    at criticality, i.e. determining the phase transition.

    xs: Any - x data of the function
    results: Any - y data of the function
    guess: Any - guess values to assign for the optimization

    """
    assert len(xs) == len(
        results
    ), f"The x and y must have the same dimension, but x has dim({len(xs)}) and y has dim({len(results)})"

    # define the function to fit
    def fit(x, c, corr_length):
        return c / 6 * np.log(x - np.log(x / corr_length + np.exp(-x / corr_length)))

    # fit your data with a given guess
    param_opt, covar_opt = curve_fit(fit, xs, results, guess)
    return param_opt, covar_opt


def mps_to_vector(mps):
    D = mps[0].shape[0]
    a = np.zeros(D)
    a[0] = 1
    a = ncon([a, mps[0]], [[1], [1, -1, -2]])
    for i in range(1, len(mps)):
        a_index = list(range(-i, 0))
        a_index.reverse()
        a_index.append(1)
        a = ncon([a, mps[i]], [a_index, [1, -(i + 1), -(i + 2)]])

    a_index = list(range(-i - 1, 0))
    a_index.reverse()
    a_index.append(1)
    b = np.zeros(D)
    b[-1] = 1
    final_vec = ncon([a, b.T], [a_index, [1]])
    final_vec = final_vec.reshape(2 ** len(mps))
    return final_vec


def mpo_to_matrix(mpo):
    v_l = np.zeros(mpo[0].shape[0])
    v_l[0] = 1
    L = len(mpo)
    a = ncon([v_l, mpo[0]], [[1], [1, -(2 * L + 1), -1, -(L + 1)]])
    for i in range(1, L):
        first_index = list(range(-i, 0))
        first_index.reverse()
        second_index = list(range(-(L + i), -L))
        second_index.reverse()
        a_index = first_index + second_index + [1]
        a = ncon([a, mpo[i]], [a_index, [1, -(2 * L + 1), -(i + 1), -(L + i + 1)]])

    first_index = list(range(-i - 1, 0))
    first_index.reverse()
    second_index = list(range(-(L + i) - 1, -L))
    second_index.reverse()
    a_index = first_index + second_index + [1]
    v_r = np.zeros(mpo[0].shape[0])
    v_r[-1] = 1
    final_matrix = ncon([a, v_r.T], [a_index, [1]])
    final_matrix = final_matrix.reshape((2**L, 2**L))
    return final_matrix


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
"""
Defining a series of functions to have single and double site operators
"""
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Before
# ---------------------------------------------------------------------------------------
def before(next: csr_matrix):
    """
    before

    This function computes the tensor product between identity
    and whatever other operator specified by next.

    next: np.ndarray - Usually choose the operator that you want to add after
            an identity, e.g. I,next = I,X,I,I

    """
    id = np.eye(2)
    id = csr_matrix(id)
    return spkron(id, next)


# ---------------------------------------------------------------------------------------
# Before Tot
# ---------------------------------------------------------------------------------------
def before_tot(next: csr_matrix, site: int):
    """
    before_tot

    This function computes the tensor product between the subspace
    before the operator at a give site and the what comes after.

    next: np.ndarray - Usually choose the operator that you want to add after
            an identity, e.g. I,next = I,X,I,I
    site: int - The site where the operator is acting. Indicates the number of
            identities to be applied before

    """
    for i in range(site - 1):
        next = before(next)
    return next


# ---------------------------------------------------------------------------------------
# After
# ---------------------------------------------------------------------------------------
def after(op: csr_matrix, site: int, L: int):
    """
    after

    This function takes into account on which site the operator is acting
    and how many site there are left after it to leave unaltered,
    e.g. we act with identity.

    op: np.ndarray - operator defining our single site generalized operator
    site: int - The site where the operator is acting
    L: int - Chain length. The difference (L-site) indicates the number of
            identities to be applied after the operator op
    """
    next = np.eye(2)
    next = csr_matrix(next)
    if site < L:
        for i in range(site, L - 1):
            next = before(next)
        return spkron(op, next)
    else:
        return op


# ---------------------------------------------------------------------------------------
# Single site operator
# ---------------------------------------------------------------------------------------
def single_site_op(op: Union[csr_matrix, np.ndarray], site: int, L: int):
    """
    single_site_op

    This function combines the identities before the operator with the
    op+identities after. E.g., with after we compute op,I,···,I
    and we add to the identities before I,···,I,op,I,···,I.

    op: np.ndarray - operator defining our single site generalized operator
    site: int - The site where the operator is acting
    L: int - Chain length. The difference (L-site) indicates the number of
            identities to be applied after the operator op

    """
    assert isinstance(
        op, (csr_matrix, np.ndarray)
    ), "Input must be a NumPy array or a sparse CSR op"

    if isinstance(op, np.ndarray):
        # Convert the NumPy array to a CSR matrix
        op = csr_matrix(op)
    op_tot = before_tot(after(op, site, L), site=site)
    return op_tot


# ---------------------------------------------------------------------------------------
# Double After
# ---------------------------------------------------------------------------------------
def double_after(op: Union[csr_matrix, np.ndarray], site: int, L: int):
    """
    double_after

    This function takes into account on which site the two-site operator
    is acting and how many site there are left after it to leave unaltered,
    e.g. we act with identity.

    op: np.ndarray - operator defining our two-site generalized operator
    site: int - The tuple (site,site+1) is where the operator is acting
    L: int - Chain length. The difference (L-site+1) indicates the number of
            identities to be applied after the operator op

    """
    assert isinstance(
        op, (csr_matrix, np.ndarray)
    ), "Input must be a NumPy array or a sparse CSR matrix"

    if isinstance(op, np.ndarray):
        # Convert the NumPy array to a CSR matrix
        op = csr_matrix(op)

    next = np.eye(2)
    next = csr_matrix(next)
    if site <= L - 2:
        for i in range(site + 1, L - 1):
            next = before(next)
        next = spkron(op, next)
        return spkron(op, next)
    else:
        return spkron(op, op)


# ---------------------------------------------------------------------------------------
# Two-site operator
# ---------------------------------------------------------------------------------------
def two_site_op(op: csr_matrix, site: int, L: int):
    """
    two_site_op

    This function combines the identities before the operator with the
    op+identities after. E.g., with after we compute op,op,I,···,I
    and we add to the identities before I,···,I,op,op,I,···,I.

    op: np.ndarray - operator defining our two-site generalized operator
    site: int - The tuple (site,site+1) is where the operator is acting
    L: int - Chain length. The difference (L-site+1) indicates the number of
            identities to be applied after the operator op

    """
    assert site < L, "Site out of bounds"

    op_tot = before_tot(double_after(op, site, L), site)
    return op_tot


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
"""
With a single and double site operators we can define different Hamiltonians
and other special operators, e.g. flipping half of the spins' chain
"""
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Interaction Hamiltonian
# ---------------------------------------------------------------------------------------
def H_int(L: int, op: csr_matrix):
    """
    H_int

    This function sums the two-site interactions terms of a given operator.

    L: int - Total number of spins in the chain
    op: np.ndarray - operator defining the two-site interaction

    """
    h = 0
    for site in range(1, L):
        h += two_site_op(op, site, L)
    return h


# ---------------------------------------------------------------------------------------
# Local Hamiltonian
# ---------------------------------------------------------------------------------------
def H_loc(L: int, op: csr_matrix):
    """
    H_int

    This function sums the single-site local terms of a given operator.

    L: int - Total number of spins in the chain
    op: np.ndarray - operator defining the single-site local term

    """
    h = 0
    for site in range(1, L + 1):
        h += single_site_op(op, site, L)
    return h


# ---------------------------------------------------------------------------------------
# Ising Hamiltonian
# ---------------------------------------------------------------------------------------
def H_ising_gen(
    L: int, op_l: csr_matrix, op_t: csr_matrix, J: float, h_l: float, h_t: float
) -> csr_matrix:
    """
    H_ising_gen

    This function finds the ising hamiltonian by summing local and
    interaction terms with given parameters and operators.

    L: int - Total number of spins in the chain
    op_l: np.ndarray - operator defining the two-site interaction term
    op_t: np.ndarray - operator defining the single-site local term
    J: float - couppling constant of the interaction term
    h_l: float - couppling constant of the local logitudinal term
    h_t: float - couppling constant of the local transverse term
    """
    return -J * H_int(L, op=op_l) - h_l * H_loc(L, op_l) - h_t * H_loc(L, op_t)


def sparse_non_diag_paulis_indices(n, N):
    """
    Returns a tuple (row_indices, col_indices) containing the row and col indices of the non_zero elements
    of the tensor product of a non diagonal pauli matrix (x, y) acting over a single qubit in a Hilbert
    space of N qubits
    
    """
    if 0 <= n < N:
        block_length = 2**(N - n - 1)
        nblocks = 2**n
        ndiag_elements = block_length*nblocks
        k = np.arange(ndiag_elements, dtype=int)
        red_row_col_ind = (k % block_length) + 2*(k // block_length)*block_length
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
        result = csc_array((data, (row_indices_cache, col_indices_cache)), shape=(2**L, 2**L), dtype=complex)
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < L")
    
def sparse_pauli_y(n, L, row_indices_cache=None, col_indices_cache=None):
    """
    Returns a CSC sparse matrix representation of the pauli_y matrix acting over qubit n in a Hilbert space of L qubits
    0 <= n < L
    
    """
    if 0 <= n < L :
        if (row_indices_cache is None) or (col_indices_cache is None):
            row_indices_cache, col_indices_cache = sparse_non_diag_paulis_indices(n, L)
        data = -1j*np.ones_like(row_indices_cache)
        data[len(data)//2::] = 1j
        result = csc_array((data, (row_indices_cache, col_indices_cache)), shape=(2**L, 2**L), dtype=complex)
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < L")
    
def sparse_pauli_z(n, L):
    """
    Returns a CSC sparse matrix representation of the pauli_z matrix acting over qubit n in a Hilbert space of L qubits
    0 <= n < L
    
    """
    if 0 <= n < L:
        block_length = 2**(L - n)
        nblocks = 2**n
        block = np.ones(block_length, dtype=int)
        block[block_length//2::] = -1
        diag = np.tile(block, nblocks)
        row_col_indices = np.arange(2**L, dtype=int)
        result = csc_array((diag, (row_col_indices, row_col_indices)), shape=(2**L, 2**L), dtype=complex)
        return result
    else:
        raise ValueError("Index n must fulfill 0 <= n < L")
    
def sparse_ising_hamiltonian(J, h_t, h_l, L):
    """
    Returns a sparse representation of the Hamiltonian of the 1D Heisemberg model in a chain of length L
    with periodic boundary conditions (hbar = 1)
    J < 0: Antiferromagnetic case (Unique ground state of total angular momentum S=0)
    J > 0: Ferromagnetic case (L+1-fold degeneracy of the ground state of angular momentum L/2) -> Dicke states for even L
    """
    hamiltonian_l = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_t = csc_array((2**L, 2**L), dtype=complex)
    hamiltonian_int = csc_array((2**L, 2**L), dtype=complex)
   
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
    
    for n in range(L-1):
        n_pauli_z = sparse_pauli_z(n, L)
        np1_pauli_z = sparse_pauli_z(n+1, L)
        hamiltonian_int += n_pauli_z @ np1_pauli_z

    return - J*hamiltonian_int - h_t*hamiltonian_t - h_l*hamiltonian_l


# ---------------------------------------------------------------------------------------
# Exact Initial State
# ---------------------------------------------------------------------------------------
def sparse_ising_ground_state(L: int, h_t: float, h_l: float = 1e-7, J: float = 1, k: int = 1) -> csc_array:
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
# Flipping half
# ---------------------------------------------------------------------------------------
def flipping_half(L: int, op: csr_matrix):
    """
    flipping_half

    This function flips the half of the spin states to the right of the chain.
    To achieve the flipping it is important a correct choice of the operator

    L: int - chain size
    op: np.ndarray - flipping operator. E.g. spin in Z basis -> op=X

    """
    O = op
    for _ in range(L // 2 - 1):
        O = spkron(op, O)

    for _ in range(L // 2):
        O = before(O)
    return O


# ---------------------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------------------
def truncation(array: Union[np.ndarray, csc_matrix, csr_matrix], threshold: float):
    """
    truncation

    This function truncates the entries of an array according to the preselected
    threshold.

    array: np.ndarray - array to truncate
    threshold: float - level of the truncation

    """
    if isinstance(array, np.ndarray):
        return np.where(np.abs(np.real(array)) > threshold, array, 0)
    if isinstance(array, csc_matrix) or isinstance(array, csr_matrix):
        # Apply the thresholding operation on the csc_matrix
        filtered_matrix = csc_matrix((array.shape), dtype=array.dtype)
        # Apply the thresholding operation
        filtered_matrix.data = array.data * (array.data > threshold)
        return filtered_matrix


# ---------------------------------------------------------------------------------------
# Compute Ising Spectrum
# ---------------------------------------------------------------------------------------
def compute_ising_spectrum(L: int, h_l: float = 0, n_points: int = 100, k: int = 6):
    """
    compute_ising_spectrum

    This function is computing the spectrum of an Ising Hamiltonian wrt the trnaverse field.

    L: int - chain size
    h_l: float - initial longitudinal field parameter.
        Can be used to break the symmetry of the ground state for h_t = 0
    n_points: int - number of point in the transverse field we want to evaluate the spectrum. By default a 100 points
    k: int - number of eigenvalues we want to compute. By default 6

    """
    X = np.array([[0, 1], [1, 0]])
    X = csr_matrix(X)
    Z = np.array([[1, 0], [0, -1]])
    Z = csr_matrix(Z)
    h_ts = np.linspace(0, 2, n_points)
    eigvals = []
    for h_t in h_ts:
        H = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=h_l, h_t=h_t)
        e, v = eigsh(H, k=k, which="SA")
        print(f"first 6 igenvalues SA (Smallest (algebraic) eigenvalues): {e}")
        eigvals.append(e)
    return eigvals


# ---------------------------------------------------------------------------------------
# Exact Initial State
# ---------------------------------------------------------------------------------------
def exact_initial_state(L: int, h_t: float, h_l: float = 1e-7, k: int = 1) -> csc_array:
    """
    exact_initial_state

    This function is computing the initial state given by an Ising Hamiltonian.

    L: int - chain size
    h_t: float - initial transverse field parameter
    h_l: float - initial longitudinal field parameter
    k: int - number of eigenvalues we want to compute. By default 1

    """
    X = np.array([[0, 1], [1, 0]])
    X = csr_matrix(X)
    Z = np.array([[1, 0], [0, -1]])
    Z = csr_matrix(Z)
    print("Finding the Hamiltonian...")
    H = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=h_l, h_t=h_t)
    print("Hamiltonian found")
    e, v = eigsh(H, k=k, which="SA")
    print(f"first {k} eigenvalue(s) SA (Smallest (algebraic) eigenvalues): {e}")
    psi = v[:, 0]
    flip = single_site_op(op=X, site=L // 2 + 1, L=L)
    psi = csc_array(flip @ psi)
    return psi


# ---------------------------------------------------------------------------------------
# U Evolution
# ---------------------------------------------------------------------------------------
def U_evolution(
    H_ev: csc_array,
    psi_init: csc_array,
    trotter_step: int,
    delta: float,
):
    """
    U_evolution

    This function applies a time evolution operator to some initial state.
    The evolution operator uses the Ising hamiltonian with some tunable parameters.

    L: int - chain size
    psi_init: csc_array - initial state to be evolved
    trotter_step: int - indicates the specific trotter step we are evolving
    delta: float - indicates the time step we are taking at each trotter step
    h_t: float - initial transverse field parameter
    h_l: float - initial longitudinal field parameter

    """
    time = delta * trotter_step
    H_ev = -1j * time * H_ev.tocsc()
    # U = expm(-1j*time*H_ev.tocsc())
    # U_new = truncation(array=U, threshold=1e-15)
    psi_ev = expm_multiply(H_ev, psi_init)
    return psi_ev

# ---------------------------------------------------------------------------------------
# U Evolution sparse
# ---------------------------------------------------------------------------------------
def U_evolution_sparse(
    psi_init: csc_array,
    H_ev: csc_array,
    trotter_step: int,
    delta: float,
):
    """
    U_evolution

    This function applies a time evolution operator to some initial state.
    The evolution operator uses the Ising hamiltonian with some tunable parameters.

    L: int - chain size
    psi_init: csc_array - initial state to be evolved
    trotter_step: int - indicates the specific trotter step we are evolving
    delta: float - indicates the time step we are taking at each trotter step
    h_t: float - initial transverse field parameter
    h_l: float - initial longitudinal field parameter

    """
    # time = delta * trotter_step
    time = delta
    H_ev = -1j * time * H_ev
    psi_ev = expm_multiply(H_ev, psi_init)
    return psi_ev


def get_middle_chain_schmidt_values(vec):
    L = int(np.log2(len(vec)))
    matrix = vec.reshape((2**(L//2),2**(L//2)))
    s, v, d = np.linalg.svd(matrix, full_matrices=False)
    return s
# ---------------------------------------------------------------------------------------
# Exact Evolution
# ---------------------------------------------------------------------------------------
def exact_evolution(
    L: int, h_t: float, h_ev: float, delta: float, trotter_steps: int, h_l: float = 1e-7, flip: bool = False
):
    """
    exact_evolution

    This function evolves an initial state for trotter steps times.
    We extract at each time step the local and total magnetization.

    L: int - chain size
    trotter_step: int - indicates the specific trotter step we are evolving
    delta: float - indicates the time step we are taking at each trotter step
    h_t: float - initial transverse field parameter
    h_l: float - initial longitudinal field parameter
    h_ev: float - evolution transverse field parameter

    """
    # psi_exact = exact_initial_state(L=L, h_t=h_t, h_l=h_l).reshape(2**L, 1)
    init_state = np.zeros((1, 2, 1))
    init_state[0, 0, 0] = 1
    mps = [init_state for _ in range(L)]
    psi_exact = mps_to_vector(mps).reshape(2**L, 1)
    X = np.array([[0, 1], [1, 0]])
    X = csr_matrix(X)
    Z = np.array([[1, 0], [0, -1]])
    Z = csr_matrix(Z)
    if flip:
        # flip = single_site_op(op=Z, site=L//2 + 1, L=L)
        flip = sparse_pauli_x(n=L//2, L=L)
        psi_exact = csc_array(flip @ psi_exact)
    # local
    # mag_loc_op = [single_site_op(op=Z, site=i, L=L) for i in range(1, L + 1)]
    mag_loc_op = [sparse_pauli_z(n=i, L=L) for i in range(L)]
    # total
    # mag_tot_op = H_loc(L=L, op=Z)
    mag_tot_op = sparse_ising_hamiltonian(J=0, h_t=0, h_l=-1, L=L)

    mag_exact_loc = []
    mag_exact_tot = []

    # local
    mag_exact = []
    for i in range(L):
        mag_exact.append(
            (psi_exact.T.conjugate() @ mag_loc_op[i] @ psi_exact).data[0].real
        )
    mag_exact_loc.append(mag_exact)

    # total
    mag = (psi_exact.T.conjugate() @ mag_tot_op @ psi_exact).data
    mag_exact_tot.append(mag.real)

    H_ev = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=h_l, h_t=h_ev)
    for trott in range(trotter_steps):
        print(f"-------- Trotter step {trott} ---------")
        # exact
        psi_new = U_evolution(
            H_ev=H_ev, psi_init=psi_exact, trotter_step=trott + 1, delta=delta
        )

        # local
        mag_exact = []
        for i in range(L):
            mag_exact.append(
                (psi_new.T.conjugate() @ mag_loc_op[i] @ psi_new).data[0].real
            )
        mag_exact_loc.append(mag_exact)

        # total
        mag = (psi_new.T.conjugate() @ mag_tot_op @ psi_new).data
        mag_exact_tot.append(mag.real)
    return psi_new, mag_exact_loc, mag_exact_tot

# ---------------------------------------------------------------------------------------
# Exact Evolution sparse
# ---------------------------------------------------------------------------------------
def exact_evolution_sparse(
    L: int, h_t: float, h_ev: float, delta: float, trotter_steps: int, h_l: float = 1e-7, flip: bool = False
):
    """
    exact_evolution

    This function evolves an initial state for trotter steps times.
    We extract at each time step the local and total magnetization.

    L: int - chain size
    trotter_step: int - indicates the specific trotter step we are evolving
    delta: float - indicates the time step we are taking at each trotter step
    h_t: float - initial transverse field parameter
    h_l: float - initial longitudinal field parameter
    h_ev: float - evolution transverse field parameter

    """
    # psi_exact = sparse_ising_ground_state(L=L, h_t=h_t, h_l=h_l, k=6).reshape(2**L, 1)
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
    mag_loc_X_op = sparse_pauli_x(n=L//2, L=L)
    # total
    mag_tot_op = sparse_ising_hamiltonian(L=L, h_t=0, h_l=-1, J=0)

    mag_exact_loc_Z = []
    mag_exact_loc_X = []
    mag_exact_tot = []

    # local Z
    mag_exact = []
    for i in range(L):
        mag_exact.append(
            (psi_exact.T.conjugate() @ mag_loc_Z_op[i] @ psi_exact).data[0].real
        )
    mag_exact_loc_Z.append(mag_exact)

    # local X
    # mag_exact_loc_X.append((psi_exact.T.conjugate() @ mag_loc_X_op @ psi_exact).data[0].real)
    
    # total
    mag = (psi_exact.T.conjugate() @ mag_tot_op @ psi_exact).data
    mag_exact_tot.append(mag.real)

    H_ev = sparse_ising_hamiltonian(L=L, J=1, h_l=h_l, h_t=h_ev)
    entropy = [0]
    psi_new = psi_exact
    for trott in range(trotter_steps):
        print(f"-------- Trotter step {trott} ---------")
        # exact
        # psi_new = U_evolution_sparse(
        #     psi_init=psi_exact, H_ev=H_ev, trotter_step=trott + 1, delta=delta
        # )
        psi_new = U_evolution_sparse(
            psi_init=psi_new, H_ev=H_ev, trotter_step=trott + 1, delta=delta
        )
        # entropy
        # s = get_middle_chain_schmidt_values(psi_new)
        # ent = von_neumann_entropy(s)
        # entropy.append(ent)

        # local Z
        mag_exact = []
        for i in range(L):
            mag_exact.append(
                (psi_new.T.conjugate() @ mag_loc_Z_op[i] @ psi_new).data[0].real
            )
        mag_exact_loc_Z.append(mag_exact)

        # local X
        # mag_exact_loc_X.append((psi_new.T.conjugate() @ mag_loc_X_op @ psi_new).data[0].real)

        # total
        mag = (psi_new.T.conjugate() @ mag_tot_op @ psi_new).data
        mag_exact_tot.append(mag.real)
    return psi_new, mag_exact_loc_Z, mag_exact_loc_X, mag_exact_tot, entropy
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
"""
Visualization tools
"""
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Plot Side By Side
# ---------------------------------------------------------------------------------------
def plot_side_by_side(
    data1, data2, cmap="viridis", title1="Imshow 1", title2="Imshow 2"
):
    """
    plot_side_by_side

    This visualization function creates two plots of colormaps one next to the other.

    data1: data of the left colormap
    data2: data of the right colormap
    cmap: string - colormap chosen to visualize the two colormap. Should be the same. By default 'viridis'
    title1: string - title of the left colormap
    title2: string - title of the right colormap

    """
    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the imshow plots on the subplots
    im1 = ax1.imshow(data1, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    im2 = ax2.imshow(data2, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Set titles for the subplots
    ax1.set_title(title1)
    ax2.set_title(title2)

    # Remove ticks from the colorbar subplot
    # ax2.set_xticks([])
    ax2.set_yticks([])

    # Create a colorbar for the second plot on the right
    cbar = fig.colorbar(im2, ax=ax2)
    cbar = fig.colorbar(im1, ax=ax1)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------
# Create Sequential Colors
# ---------------------------------------------------------------------------------------
def create_sequential_colors(num_colors, colormap_name):
    """
    create_sequential_colors

    This function creates a sequence of colors extracted from a specified colormap.

    num_colors: int - number of colors we want to extract
    colormap_name: string - colormap name we want to use

    """
    colormap = plt.cm.get_cmap(colormap_name)
    colormap_values = np.linspace(0, 1, num_colors)
    colors = [colormap(value) for value in colormap_values]
    return colors



