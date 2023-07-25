import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse.linalg import expm
from scipy.sparse import csr_matrix
import os
from ncon import ncon 
import matplotlib.pyplot as plt

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
# Variance
# ---------------------------------------------------------------------------------------
def variance(first_m,sm):
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
    return 1 - fourth_m/(3*sm**2)

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
    add = 2*np.pi/L
    k = -(L-1)*np.pi/L
    ks.append(k)
    for _ in range(L-1):
        ks.append(ks[-1]+add)

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
        e_0.append(np.sqrt(2+2*np.cos(k)))

    return - sum(e_0)

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
    return -np.sum((s**2)*np.log2(s**2))

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
            h = f'{h:.2f}'
            # We split the filename where the point is
            h = os.path.splitext(h)[1]
            if h in oldName:
                b = n + h
                newName = os.path.join(folder, b)
                # print(newName)
                # Rename the file
                os.rename(oldName, newName)

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
    assert len(xs) == len(results), f"The x and y must have the same dimension, but x has dim({len(xs)}) and y has dim({len(results)})"
    # define the function to fit
    def fit(x, c, corr_length):
        return c/6*np.log(x - np.log(x/corr_length + np.exp(-x/corr_length)))
    
    # fit your data with a given guess
    param_opt, covar_opt = curve_fit(fit, xs, results, guess)
    return param_opt, covar_opt

def mps_to_vector(mps):
    D = mps[0].shape[0]
    a = np.zeros(D)
    a[0] = 1
    a = ncon([a,mps[0]],[[1],[1,-1,-2]])
    for i in range(1,len(mps)):
        a_index = list(range(-i,0))
        a_index.reverse()
        a_index.append(1)
        a = ncon([a,mps[i]],[a_index,[1,-(i+1),-(i+2)]])

    a_index = list(range(-i-1,0))
    a_index.reverse()
    a_index.append(1)
    b = np.zeros(D)
    b[-1] = 1
    final_vec = ncon([a,b.T],[a_index,[1]])
    final_vec = final_vec.reshape(2**len(mps))
    return final_vec
 
def mpo_to_matrix(mpo):
    v_l = np.zeros(mpo[0].shape[0])
    v_l[0] = 1
    L = len(mpo)
    a = ncon([v_l,mpo[0]],[[1],[1,-(2*L+1),-1,-(L+1)]])
    for i in range(1,L):
        first_index = list(range(-i,0))
        first_index.reverse()
        second_index = list(range(-(L+i),-L))
        second_index.reverse()
        a_index = first_index + second_index + [1]
        a = ncon([a,mpo[i]],[a_index,[1,-(2*L+1),-(i+1),-(L+i+1)]])

    first_index = list(range(-i-1,0))
    first_index.reverse()
    second_index = list(range(-(L+i)-1,-L))
    second_index.reverse()
    a_index = first_index + second_index + [1]
    v_r = np.zeros(mpo[0].shape[0])
    v_r[-1] = 1
    final_matrix = ncon([a,v_r.T],[a_index,[1]])
    final_matrix = final_matrix.reshape((2**L,2**L))
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
def before(next):
    """
    before

    This function computes the tensor product between identity
    and whatever other operator specified by next.

    next: np.ndarray - Usually choose the operator that you want to add after
            an identity, e.g. I,next = I,X,I,I

    """
    id = np.eye(2)
    return np.kron(id,next)

# ---------------------------------------------------------------------------------------
# Before Tot
# ---------------------------------------------------------------------------------------
def before_tot(next, site):
    """
    before_tot

    This function computes the tensor product between the subspace
    before the operator at a give site and the what comes after.

    next: np.ndarray - Usually choose the operator that you want to add after
            an identity, e.g. I,next = I,X,I,I
    site: int - The site where the operator is acting. Indicates the number of 
            identities to be applied before

    """
    for i in range(site-1):
        next = before(next)
    return next

# ---------------------------------------------------------------------------------------
# After
# ---------------------------------------------------------------------------------------
def after(op, site, L):
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
    if site < L:
        for i in range(site,L-1):
            next = before(next)
        return np.kron(op,next)
    else:
        return op

# ---------------------------------------------------------------------------------------
# Single site operator
# ---------------------------------------------------------------------------------------
def single_site_op(op, site, L):
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
    op_tot = before_tot(after(op, site, L), site=site)
    return op_tot

# ---------------------------------------------------------------------------------------
# Double After
# ---------------------------------------------------------------------------------------
def double_after(op, site, L):
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
    next = np.eye(2)
    if site <= L-2:
        for i in range(site+1,L-1):
            next = before(next)
        next = np.kron(op,next)
        return np.kron(op,next)
    else:
        return np.kron(op,op)

# ---------------------------------------------------------------------------------------
# Two-site operator
# ---------------------------------------------------------------------------------------
def two_site_op(op, site, L):
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
    
    op_tot = before_tot(double_after(op,site,L), site)
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
def H_int(L, op):
    """
    H_int

    This function sums the two-site interactions terms of a given operator.

    L: int - Total number of spins in the chain
    op: np.ndarray - operator defining the two-site interaction

    """
    h = 0
    for site in range(1,L):
        h += two_site_op(op, site, L)
    return h

# ---------------------------------------------------------------------------------------
# Local Hamiltonian
# ---------------------------------------------------------------------------------------
def H_loc(L, op):
    """
    H_int

    This function sums the single-site local terms of a given operator.

    L: int - Total number of spins in the chain
    op: np.ndarray - operator defining the single-site local term
    
    """
    h = 0
    for site in range(1,L+1):
        h += single_site_op(op, site, L)
    return h

# ---------------------------------------------------------------------------------------
# Ising Hamiltonian
# ---------------------------------------------------------------------------------------
def H_ising_gen(L, op_l, op_t, J, h_l, h_t):
    """
    H_ising_gen

    This function finds the ising hamiltonian by summing local and
    interaction terms with given parameters and operators.

    L: int - Total number of spins in the chain
    op_int: np.ndarray - operator defining the two-site interaction term
    op_loc: np.ndarray - operator defining the single-site local term
    J: float - couppling constant of the interaction term
    h: float - couppling constant of the local term
    """    
    return - J*H_int(L, op=op_l) - h_l*H_loc(L, op_l) - h_t*H_loc(L, op_t)

# ---------------------------------------------------------------------------------------
# Flipping half
# ---------------------------------------------------------------------------------------
def flipping_half(op, L):
    """
    flipping_half

    This function flips the half of the spin states to the right of the chain.
    To achieve the flipping it is important a correct choice of the operator

    op: np.ndarray - flipping operator. E.g. spin in Z basis -> op=X

    """
    O = op
    for i in range(L//2-1):
        O = np.kron(op,O)

    for i in range(L//2):
        O = before(O)
    return O

# ---------------------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------------------
def truncation(array, threshold):
    """
    truncation

    This function truncates the entries of an array according to the preselected
    threshold. 

    array: np.ndarray - array to truncate
    threshold: float - level of the truncation

    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array should be an ndarray, not a {type(array)}")
    if not np.isscalar(threshold) and not isinstance(threshold, float):
        raise TypeError(f"threshold should be a SCALAR FLOAT, not a {type(threshold)}")
    return np.where(np.abs(np.real(array)) > threshold, array, 0)


def create_sequential_colors(num_colors, colormap_name):
    colormap = plt.cm.get_cmap(colormap_name)
    colormap_values = np.linspace(0, 1, num_colors)
    colors = [colormap(value) for value in colormap_values]
    return colors

def exact_initial_state(L, h_t):
    X = np.array([[0,1],[1,0]])
    Z = np.array([[1,0],[0,-1]])
    H = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=h_t)
    e, v = np.linalg.eig(H)
    psi = v[:,0]
    flip = single_site_op(op=X, site=L // 2 + 1, L=L)
    psi = flip @ psi
    return psi

def exact_evolution_operator(L, h_t, delta, trotter_step):
    X = np.array([[0,1],[1,0]])
    Z = np.array([[1,0],[0,-1]])
    H_ev = H_ising_gen(L=L, op_l=Z, op_t=X, J=1, h_l=0, h_t=h_t)
    time = delta*trotter_step
    U = expm(-1j*time*H_ev)
    U_new = truncation(array=U, threshold=1e-16)
    U_new = csr_matrix(U_new)
    return U_new