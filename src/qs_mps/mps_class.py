import numpy as np

from scipy.sparse import csr_matrix, csr_array, identity
from scipy.linalg import expm, solve
import scipy.linalg as la
import scipy.sparse.linalg as spla

from ncon import ncon

import time 
from qs_mps.utils import *
from qs_mps.checks import check_matrix
from qs_mps.sparse_hamiltonians_and_operators import (
    exact_evolution_sparse,
    sparse_ising_ground_state,
    U_evolution_sparse,
    sparse_pauli_x,
    sparse_pauli_y,
    sparse_pauli_z
)
from qs_mps.mpo_class import MPO_ladder
from qs_mps.TensorMultiplier import TensorMultiplierOperator

class MPS:
    def __init__(
        self, L, d, model=str, chi=None, w=None, h=None, lx=None, ly=None, eps=None, J=None, k=None, charges=None
    ):
        self.L = L
        self.d = d
        self.model = model
        self.chi = chi
        self.w = w
        self.w_dag = w
        self.h = h
        self.lx = lx
        self.ly = ly
        self.eps = eps
        self.J = J
        self.k = k # (take positive values for annni model)
        self.charges = charges
        self.site = 1
        self.sites = []
        self.bonds = []
        self.ancilla_sites = []
        self.ancilla_bonds = []
        self.env_left = []
        self.env_right = []
        self.env_left_sm = []
        self.env_right_sm = []
        self.Z2 = MPO_ladder(
            L=self.L, l=int(np.log2(self.d)), model=self.model, lamb=self.h
        )

    # -------------------------------------------------
    # Manipulation of tensors, state preparation
    # -------------------------------------------------
    def _random_state(self, seed, type_shape="trapezoidal", chi=None, ancilla=False):
        """
        _random_state

        This function helps us to initialize the quantum state in an MPS form with random tensors.
        We let grow the dimension of the bond dimension exponentially as d^(chi), with d the physical
        dimension, e.g. d=2 -> qubit.
        The random tensors are generated for a given seed and we can limit the growth of the chi by giving
        the MPS a trapezoidal shape.

        seed: int - seed for the generation of random tensor
        chi: int - the bond dimension. Given by the user already as a power of d, e.g. 4,8,16,...
        type_shape: string Literal - the MPS bond dimesion will form a pyramid which could be truncated
                    if we want to keep the computation fast and accurate enough. By default "trapezoidal"

        """
        chi = self.chi
        sites = self.sites
        if ancilla:
            sites = self.ancilla_sites

        if type_shape == "trapezoidal":
            assert (
                chi >= self.d
            ), "The bond dimension is too small for the selected physical dimension d"
            
            n = int(logarithm_base_d(x=chi, d=self.d))
            assert (
                self.L >= 2 * n
            ), "The spin chain is too small for the selected bond dimension chi"
            np.random.seed(seed)

            for i in range(n):
                sites.append(np.random.rand(self.d**i, self.d, self.d ** (i + 1)))
            for _ in range(self.L - int(2 * n)):
                sites.append(np.random.rand(self.d**n, self.d, self.d**n))
            for i in range(n):
                sites.append(
                    np.random.rand(self.d ** (n - i), self.d, self.d ** (n - i - 1))
                )

        elif type_shape == "pyramidal":
            np.random.seed(seed)
            for i in range(self.L // 2):
                sites.append(np.random.rand(2**i, self.d, 2 ** (i + 1)))

            if self.L % 2 != 0:
                sites.append(np.random.rand(2 ** (i + 1), self.d, 2 ** (i + 1)))

            for i in range(self.L // 2):
                tail = sites[self.L // 2 - i - 1]
                sites.append(
                    tail.reshape(
                        2 ** (self.L // 2 - i), self.d, 2 ** (self.L // 2 - i - 1)
                    )
                )
        elif type_shape == "rectangular":
            sites.append(np.random.rand(1, self.d, chi))
            for _ in range(self.L - 2):
                sites.append(np.random.rand(chi, self.d, chi))
            sites.append(np.random.rand(chi, self.d, 1))

        return self

    def canonical_form(
        self,
        svd_direction: str = "left",
        schmidt_tol: float = 1e-15,
        ancilla: bool = False,
        trunc_chi: bool = False,
        trunc_tol: bool = True,
    ):
        """
        canonical_form

        This function puts the tensors saved in self.sites through recursive svd.
        It corresponds in saving tensors in the (Vidal) Gamma-Lambda notation.
        It can be used both to initialize the random tensors in a normalized state
        or to bring the tensors from the amb form to the canonical one.

        svd_direction: string - the direction of the sequencial svd. Could be "right" or "left"
        schmidt_tol: float - tolerance used to cut the schmidt values after svd

        """
        if svd_direction == "left":
            self.left_svd(schmidt_tol, ancilla, trunc_chi, trunc_tol)

        elif svd_direction == "right":
            self.right_svd(schmidt_tol, ancilla, trunc_chi, trunc_tol)

        return self

    def right_svd(
        self, schmidt_tol: float, ancilla: bool, trunc_chi: bool, trunc_tol: bool
    ):
        """
        right_svd

        This function transforms the states in self.sites in a canonical
        form using svd. We start from the first site and sweeping through
        site self.L we save the Gamma tensors on each site and the Schmidt values on the bonds

        schmidt_tol: float - tolerance used to cut the schmidt values after svd

        """
        s_init = np.array([1])
        psi = np.diag(s_init)

        sites = self.sites
        bonds = self.bonds
        if ancilla:
            sites = self.ancilla_sites
            bonds = self.ancilla_bonds

        bonds.append(s_init)
        for i in range(self.L):
            new_site = ncon(
                [psi, sites[i]],
                [
                    [-1, 1],
                    [1, -2, -3],
                ],
            )
            new_site = new_site.reshape(new_site.shape[0] * self.d, new_site.shape[2])

            original_matrix = new_site
            scaled_matrix = original_matrix / np.max(np.abs(original_matrix))
            lambda_ = 1e-15
            regularized_matrix = scaled_matrix + lambda_ * np.eye(
                scaled_matrix.shape[0], scaled_matrix.shape[1]
            )
            u, s, v = la.svd(
                regularized_matrix,
                full_matrices=False,
            )

            bond_l = u.shape[0] // self.d
            u = u.reshape(bond_l, self.d, u.shape[1])
            if trunc_chi:
                if u.shape[0] > self.chi:
                    u = u[:, :, : self.chi]
                    s = s[: self.chi]
                    v = v[: self.chi, :]
                    s = s / la.norm(s)
            if trunc_tol:
                condition = s >= schmidt_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / la.norm(s_trunc)
                u = u[:, :, : len(s)]
                v = v[: len(s), :]

            sites[i] = u
            bonds.append(s)
            psi = ncon(
                [np.diag(s), v],
                [
                    [-1, 1],
                    [1, -2],
                ],
            )

        return self

    def left_svd(
        self, schmidt_tol: float, ancilla: bool, trunc_chi: bool, trunc_tol: bool
    ):
        """
        left_svd

        This function transforms the states in self.sites in a canonical
        form using svd. We start from the last site self.L and sweeping through
        site 1 we save the Gamma tensors on each site and the Schmidt values on the bonds

        schmidt_tol: float - tolerance used to cut the schmidt values after svd

        """
        s_init = np.array([1])
        psi = np.diag(s_init)

        sites = self.sites
        bonds = self.bonds
        if ancilla:
            sites = self.ancilla_sites
            bonds = self.ancilla_bonds
        bonds.append(s_init)
        # time_cf = time.perf_counter()
        for i in range(self.L - 1, -1, -1):
            new_site = ncon(
                [sites[i], psi],
                [
                    [-1, -2, 1],
                    [1, -3],
                ],
            )
            new_site = new_site.reshape(new_site.shape[0], self.d * new_site.shape[2])

            original_matrix = new_site
            scaled_matrix = original_matrix / np.max(np.abs(original_matrix))
            lambda_ = 1e-15
            regularized_matrix = scaled_matrix + lambda_ * np.eye(
                scaled_matrix.shape[0], scaled_matrix.shape[1]
            )
            u, s, v = la.svd(
                regularized_matrix,
                full_matrices=False,
            )

            bond_r = v.shape[1] // self.d
            v = v.reshape((v.shape[0], self.d, bond_r))
            if trunc_chi:
                if v.shape[0] > self.chi:
                    v = v[: self.chi, :, :]
                    s = s[: self.chi]
                    u = u[:, : self.chi]
                    s = s / la.norm(s)
            if trunc_tol:
                condition = s >= schmidt_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / la.norm(s_trunc)
                v = v[: len(s), :, :]
                bonds.append(s)
                u = u[:, : len(s)]

            sites[i] = v
            bonds.append(s)
            psi = ncon(
                [u, np.diag(s)],
                [
                    [-1, 1],
                    [1, -2],
                ],
            )

        bonds.append(s_init)
        bonds.reverse()
        # print(f"Time of svd during canonical form: {time.perf_counter()-time_cf}")
        # np.savetxt(f"results/times_data/svd_canonical_form_h_{self.h:.2f}", [time.perf_counter()-time_cf])
        return self

    def _compute_norm(self, site, ancilla=False, mixed=False):
        """
        _compute_norm

        This function computes the norm of our quantum state which is represented in mps.
        It takes the attributes .sites and .bonds of the class which gives us
        mps in canonical form (Vidal notation).

        site: int - sites in the chain
        ancilla: bool - if True we compute the norm of the ancilla_sites mps. By default False
        mixed: bool - if True we compute the braket between the ancilla_sites and the sites mps. By default False

        """
        a = np.array([1])
        if mixed:
            array_1 = self.ancilla_sites
            array_2 = self.sites

            ten = ncon(
                [a, a, array_1[0], array_2[0].conjugate()],
                [[1], [2], [1, 3, -1], [2, 3, -2]],
            )
            for i in range(1, self.L):
                ten = ncon(
                    [ten, array_1[i], array_2[i].conjugate()],
                    [[1, 2], [1, 3, -1], [2, 3, -2]],
                )

            N = ncon([ten, a, a], [[1, 2], [1], [2]]).real

        else:
            array = self.sites
            if ancilla:
                array = self.ancilla_sites

            ten = ncon(
                [a, a, array[0], array[0].conjugate()],
                [[1], [2], [1, 3, -1], [2, 3, -2]],
            )
            for i in range(1, self.L):
                ten = ncon(
                    [ten, array[i], array[i].conjugate()],
                    [[1, 2], [1, 3, -1], [2, 3, -2]],
                )

            N = ncon([ten, a, a], [[1, 2], [1], [2]]).real

        # print(f"-=-=-= Norm: {N}\n")
        return N

    def flipping_mps(self):
        """
        flipping_mps

        This function flips the mps middle site with the operator X,
        assuming to be in the computational (Z) basis.

        """
        X = np.array([[0, 1], [1, 0]])
        if len(self.sites) % 2 == 0:
            new_site = ncon([self.sites[self.L // 2 - 1], X], [[-1, 1, -3], [1, -2]])
            self.sites[self.L // 2 - 1] = new_site

        new_site = ncon([self.sites[self.L // 2], X], [[-1, 1, -3], [1, -2]])
        self.sites[self.L // 2] = new_site

        return self

    def flipping_all(self):
        """
        flipping_all

        This function flips all the sites of the mps with the operator X,
        assuming to be in the computational (Z) basis.

        """
        X = np.array([[0, 1], [1, 0]])
        for i in range(self.L):
            new_site = ncon([self.sites[i], X], [[-1, 1, -3], [1, -2]])
            self.sites[i] = new_site

        return self

    def enlarge_chi(self, type_shape: str="rectangular", prnt: bool=False):
        """
        enlarge_chi

        This function takes the mps tensors and make them of trapezoidal shape
        saturating at chi.

        """
        extended_array = []
        if type_shape == "trapezoidal":
            chi = int(np.log2(self.chi))
            for i in range(chi):
                extended_array.append(np.zeros((self.d**i, self.d, self.d ** (i + 1))))
            for _ in range(self.L - (2 * chi)):
                extended_array.append(np.zeros((self.d**chi, self.d, self.d**chi)))
            for i in range(chi):
                extended_array.append(
                    np.zeros((self.d ** (chi - i), self.d, self.d ** (chi - i - 1)))
                )

        elif type_shape == "rectangular":
            extended_array.append(np.zeros((1,self.d,self.chi)))
            for _ in range(self.L-2):
                extended_array.append(np.zeros((self.chi,self.d,self.chi)))
            extended_array.append(np.zeros((self.chi,self.d,1)))
        if prnt:
            print("shapes enlarged tensors:")
            tensor_shapes(extended_array)
            print("shapes original tensors:")
        shapes = tensor_shapes(self.sites, prnt=prnt)
        for i, shape in enumerate(shapes):
            extended_array[i][: shape[0], : shape[1], : shape[2]] = self.sites[i]

        self.sites = extended_array.copy()
        return self

    def check_canonical(self, site: int):
        """
        check_canonical

        This funciton checks if the tensor at a certain site is in the correct
        mixed canonical form, e.g., LCF on the left of the site and RCF on the right of the site.

        site: int - where we want to observe the canonical form of our mps

        """
        array = self.sites
        a = np.array([1])

        env = ncon([a, a], [[-1], [-2]])
        env_l = env
        for i in range(0, site - 1):
            I = np.eye(array[i].shape[2])
            env_l = ncon(
                [env_l, array[i], array[i].conjugate()],
                [[1, 2], [1, 3, -1], [2, 3, -2]],
            )
            env_trunc = truncation(env_l, threshold=1e-15)
            env_trunc = csr_matrix(env_trunc)
            I = csr_matrix(I)
            ratio = check_matrix(env_trunc, I)
            if ratio < 1e-12:
                print(f"the tensor at site {i+1} is in the correct LFC")
            else:
                print(f"the  tensor at site {i+1} has a ratio with the identity matrix equal to: {ratio}")

        env_r = env
        for i in range(self.L - 1, site, -1):
            I = np.eye(array[i].shape[0])
            env_r = ncon(
                [array[i], array[i].conjugate(), env_r],
                [[-1, 3, 1], [-2, 3, 2], [1, 2]],
            )
            env_trunc = truncation(env_r, threshold=1e-15)
            env_trunc = csr_matrix(env_trunc)
            I = csr_matrix(I)
            ratio = check_matrix(env_trunc, I)
            if ratio < 1e-12:
                print(f"the tensor at site {i+1} is in the correct RFC")
            else:
                print(f"the  tensor at site {i+1} has a ratio with the identity matrix equal to: {ratio}")
        return self

    # -------------------------------------------------
    # Density Matrix MPS and manipulation
    # -------------------------------------------------
    def density_matrix(self):
        kets = self.sites
        bras = [ket.conjugate() for ket in kets]
        a = np.array([1])
        env = ncon([a, a], [[-1], [-2]])
        up = [int(-elem) for elem in np.linspace(1, 0, 0)]
        down = [int(-elem) for elem in np.linspace(self.L + 1, 0, 0)]
        mid_up = [1]
        mid_down = [2]
        label_env = up + down + mid_up + mid_down
        # first site:
        for i in range(len(kets)):
            label_ket = [1, -1 - i, -self.L * 100]
            label_bra = [2, -self.L - 1 - i, -self.L * 100 - 1]
            env = ncon([env, kets[i], bras[i]], [label_env, label_ket, label_bra])
            up = [int(-elem) for elem in np.linspace(1, i + 1, i + 1)]
            down = [
                int(-elem) for elem in np.linspace(self.L + 1, self.L + 1 + i, i + 1)
            ]
            label_env = up + down + mid_up + mid_down

        mps_dm = ncon([env, a, a], [label_env, [1], [2]])
        return mps_dm

    def reduced_density_matrix(self, sites):
        """
        reduced_density_matrix

        This function allows us to get the reduced density matrix (rdm) of a mps.
        We trace out all the sites not specified in the argument sites.
        The algorithm only works for consecutive sites, e.g., [1,2,3],
        [56,57], etc. To implement a rdm of sites [5,37] we need another middle
        environment that manages the contractions between the specified sites
        """
        kets = self.sites
        bras = [ket.conjugate() for ket in kets]
        a = np.array([1])
        env = ncon([a, a], [[-1], [-2]])
        up = [int(-elem) for elem in np.linspace(1, 0, 0)]
        down = [int(-elem) for elem in np.linspace(self.L + 1, 0, 0)]
        mid_up = [1]
        mid_down = [2]
        label_env = up + down + mid_up + mid_down
        # left env:
        env_l = env
        for i in range(sites[0] - 1):
            label_ket = [1, 3, -1]
            label_bra = [2, 3, -2]
            env_l = ncon([env_l, kets[i], bras[i]], [label_env, label_ket, label_bra])
        # right env:
        env_r = env
        for i in range(self.L - 1, sites[-1]-1, -1):
            label_ket = [-1, 3, 1]
            label_bra = [-2, 3, 2]
            env_r = ncon([env_r, kets[i], bras[i]], [label_env, label_ket, label_bra])
        # central env
        # idx = 0
        for i in range(len(sites)):
            label_ket = [1, -1 - i, -len(sites) * 100]
            label_bra = [2, -len(sites) - 1 - i, -len(sites) * 100 - 1]
            env_l = ncon([env_l, kets[sites[i]-1], bras[sites[i]-1]], [label_env, label_ket, label_bra])
            up = [int(-elem) for elem in np.linspace(1, i + 1, i + 1)]
            down = [
                int(-elem)
                for elem in np.linspace(len(sites) + 1, len(sites) + 1 + i, i + 1)
            ]
            label_env = up + down + mid_up + mid_down
            # idx += 1
        mps_dm = ncon([env_l, env_r], [label_env, [1, 2]])

        return mps_dm

    def vector_to_mps(self, vec: np.ndarray, trunc_chi: bool=True, trunc_tol: bool=False, chi: int=1, schmidt_tol: float=1e-15):
        """
        vector_to_mps

        We decompose the vector with successive svd starting from the right towards the left,
        hence a left sweep. The final tensors will be in Right Canonical Form (RCF)
        
        vec: np.ndarray - vector we want to transform in a MPS

        """
        vec_legs = int(np.log2(len(vec)))
        sites = []
        bonds = []
        alpha = 1
        for i in range(vec_legs):
            matrix = vec.reshape((2**(vec_legs-(i+1)),2*alpha))
            u,s,v = la.svd(matrix, full_matrices=False)
            bond_r = v.shape[1] // 2
            v = truncation(v, threshold=1e-15)
            s = truncation(s, threshold=1e-15)
            u = truncation(u, threshold=1e-15)
            v = v.reshape((v.shape[0], 2, bond_r))
            if trunc_chi:
                if v.shape[0] > chi:
                    v = v[: chi, :, :]
                    s = s[: chi]
                    u = u[:, : chi]
                    s = s / la.norm(s)
            if trunc_tol:
                condition = s >= schmidt_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / la.norm(s_trunc)
                v = v[: len(s), :, :]
                u = u[:, : len(s)]

            sites.append(v)
            bonds.append(s)
            vec = u @ np.diag(s)
            alpha = vec.shape[1]
        
        sites.reverse()
        bonds.reverse()
        self.sites = sites.copy()
        self.bonds = bonds.copy()
        return self

    # -------------------------------------------------
    # Matrix Product Operators, MPOs
    # -------------------------------------------------
    def mpo(self, long: str="X", trans: str="Z"):
        """
        mpo

        This function selects which MPO to use according to the
        studied model. Here you can add other MPOs that you have
        independently defined in the class.

        """
        if self.model == "Ising":
            self.mpo_Ising(long=long)

        elif self.model == "ANNNI":
            self.mpo_ANNNI(long=long, deg_method=1)
        
        elif self.model == "Cluster":
            self.mpo_Cluster(long=long)
        
        elif self.model == "Cluster-XY":
            self.mpo_Cluster_xy(long=long)

        elif self.model == "Z2_dual":
            self.Z2.mpo_Z2_ladder_generalized()
            self.w = self.Z2.mpo

        elif self.model == "XXZ":
            self.mpo_xxz(long=long)

        return self

    # -------------------------------------------------
    # Hamiltonians, time evolution operators
    # -------------------------------------------------
    def mpo_Ising(self, long: str = "Z", trans: str = "X"):
        """
        mpo_Ising

        This function defines the MPO for the 1D transverse field Ising model.
        It takes the same MPO for all sites.

        """
        I = np.eye(2)
        O = np.zeros((2, 2))
        if long == "Z":
            long_op = sparse_pauli_z(n=0, L=1).toarray()
            trans_op = sparse_pauli_x(n=0, L=1).toarray()
        elif long == "X":
            long_op = sparse_pauli_x(n=0, L=1).toarray()
            trans_op = sparse_pauli_z(n=0, L=1).toarray()

        w_tot = []
        for i in range(self.L):
            if i == 0 or i == 1:
                c = 1
            else:
                c = 0
            w = np.array(
                [[I, -self.J * long_op, -self.h * trans_op - self.eps * c * (long_op - I)], [O, O, long_op], [O, O, I]]
            )
            w_tot.append(w)
        self.w = w_tot
        return self
    
    def mpo_ANNNI(self, long: str = "X", trans: str = "Z", deg_method: int = 2):
        """
        mpo_ANNNI

        This function defines the MPO for the 1D Axial Next-Nearest Neighbor Interaction model.
        It takes the same MPO for all sites apart from a correction term to break degeneracy.

        """
        I = identity(2, dtype=complex).toarray()
        O = csc_array((2, 2), dtype=complex).toarray()
        if long == "Z":
            long_op = sparse_pauli_z(n=0, L=1).toarray()
            trans_op = sparse_pauli_x(n=0, L=1).toarray()
        elif long == "X":
            long_op = sparse_pauli_x(n=0, L=1).toarray()
            trans_op = sparse_pauli_z(n=0, L=1).toarray()

        w_tot = []
        for i in range(self.L):
            if deg_method == 1:
                if i == 0 or i == 1:
                    c = 1
                else:
                    c = 0
                c_i = c
            elif deg_method == 2:
                c = (1 + (-1)**(i // 2))
                c_i = 1

            w = np.array(
                [[I, long_op, O, - (self.h * self.J) * trans_op - (self.eps * self.J * c) * long_op + (self.eps * self.J * c_i) * I], 
                 [O, O, I, - (self.J) * long_op], 
                 [O, O, O, (self.k * self.J) * long_op], 
                 [O, O, O, I]]
            )
            w_tot.append(w)
        self.w = w_tot
        return self

    def mpo_Cluster(self, long: str = "X", trans: str = "Z", eps: float=1e-5):
        """
        mpo_Cluster

        This function defines the MPO for the 1D Cluster model.
        It takes the same MPO for all sites.

        """
        I = identity(2, dtype=complex).toarray()
        O = csc_array((2, 2), dtype=complex).toarray()
        if long == "Z":
            long_op = sparse_pauli_z(n=0, L=1).toarray()
            trans_op = sparse_pauli_x(n=0, L=1).toarray()
        elif long == "X":
            long_op = sparse_pauli_x(n=0, L=1).toarray()
            trans_op = sparse_pauli_z(n=0, L=1).toarray()
        w_tot = []

        for i in range(self.L):
            if i == 0:
                c = -eps
            else:
                c = 0

            w = np.array(
                [[I, long_op, O, -self.h * trans_op + c * long_op],
                 [O, O, trans_op, O],
                 [O, O, O, -self.J * long_op],
                 [O, O, O, I]]
            )
            w_tot.append(w)
        self.w = w_tot
        return self

    def mpo_Cluster_xy(self, long: str="X", eps: float=1e-5):
        """
        mpo_Cluster

        This function defines the MPO for the 1D Cluster model.
        It takes the same MPO for all sites.

        """
        I = identity(2, dtype=complex).toarray()
        O = csc_array((2, 2), dtype=complex).toarray()
        Y = sparse_pauli_y(n=0, L=1).toarray()
        if long == "Z":
            long_op = sparse_pauli_z(n=0, L=1).toarray()
            trans_op = sparse_pauli_x(n=0, L=1).toarray()
        elif long == "X":
            long_op = sparse_pauli_x(n=0, L=1).toarray()
            trans_op = sparse_pauli_z(n=0, L=1).toarray()
        w_tot = []

        for i in range(self.L):
            if i == 0:
                c = -eps
            else:
                c = 0

            w = np.array(
                [[I, long_op, O, Y, -self.h * trans_op + c * long_op],
                 [O, O, trans_op, O, self.lx * long_op],
                 [O, O, O, O, -self.J * long_op],
                 [O, O, O, O, self.ly * Y],
                 [O, O, O, O, I]]
            )
            w_tot.append(w)
        self.w = w_tot
        return self
    
    def mpo_xxz(self, long: str="X", eps: float=1e-5):
        """
        mpo_Cluster

        This function defines the MPO for the 1D Cluster model.
        It takes the same MPO for all sites.

        """
        I = identity(2, dtype=complex).toarray()
        O = csc_array((2, 2), dtype=complex).toarray()
        Y = sparse_pauli_y(n=0, L=1).toarray()
        if long == "Z":
            long_op = sparse_pauli_z(n=0, L=1).toarray()
            trans_op = sparse_pauli_x(n=0, L=1).toarray()
        elif long == "X":
            long_op = sparse_pauli_x(n=0, L=1).toarray()
            trans_op = sparse_pauli_z(n=0, L=1).toarray()
        w_tot = []

        for i in range(self.L):
            if i == 0:
                c = -eps
            else:
                c = 0

            w = np.array(
                [[I, long_op, Y, trans_op, -self.h * trans_op + c * long_op],
                 [O, O, O, O, -self.J * long_op],
                 [O, O, O, O, -self.J * Y],
                 [O, O, O, O, -self.J * self.k * trans_op],
                 [O, O, O, O, I]]
            )
            w_tot.append(w)
        self.w = w_tot
        return self
    
    def mpo_quench(
        self,
        quench: str,
        delta: float = None,
        h_ev: float = None,
        J_ev: float = 1,
        sites: list = -1,
    ):
        """
        mpo_quench

        This function selects which quench we want to perform.

        quench: str - type of quench. Available are: 'flip', 'global'

        """
        if sites == -1:
            sites = [self.L // 2]

        if quench == "flip":
            self.mpo_quench_flip(sites)
        elif quench == "global":
            if self.model == "Ising":
                self.mpo_Ising_quench_global(delta, h_ev, J_ev)
            if self.model == "Z2":
                MPO_ladder.mpo_Z2_quench_global(delta, h_ev, J_ev)

    def mpo_quench_flip(self, sites):
        """
        mpo_quench_flip

        This function defines the quench of a hamiltonian
        which flips the spin system in some sites. The default
        flip is with the X operator.

        sites: list - list of sites we want to quench
        """
        I = np.eye(2)
        I = np.array([[I]])
        O = np.zeros((2, 2))
        X = np.array([[0, 1], [1, 0]])
        X_exp = np.array([[expm(1j * X)]])
        w_tot = []
        for i in range(1, self.L + 1):
            if i in sites:
                w_tot.append(X_exp)
            else:
                w_tot.append(I)
        self.w = w_tot
        return self

    def mpo_Ising_quench_global(self, delta: float, h_ev: float, J_ev: float = 1):
        """
        mpo_Ising_quench_global

        This function defines the MPO for the real time evolution of a 1D transverse field Ising model.
        We use this to perform a second order TEBD.

        delta: float - Trotter step for the time evolution
        h_ev: float - parameter of the local field for the quench
        J_ev: float - parameter of the interaction field for the quench

        """
        I = np.eye(2)
        O = np.zeros((2, 2))
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        w_tot = []
        w_loc = np.array(expm(1j * h_ev * delta / 2 * X))
        w_even = np.array(
            [
                [
                    np.sqrt(np.cos(J_ev * delta)) * I,
                    1j * np.sqrt(np.sin(J_ev * delta)) * Z,
                ]
            ]
        )
        w_in = ncon([w_even, w_loc, w_loc], [[-1, -2, 1, 2], [-3, 1], [2, -4]])
        w_odd = np.array(
            [[np.sqrt(np.cos(J_ev * delta)) * I, np.sqrt(np.sin(J_ev * delta)) * Z]]
        )
        w_odd = np.swapaxes(w_odd, axis1=0, axis2=1)
        w_fin = ncon([w_odd, w_loc, w_loc], [[-1, -2, 1, 2], [-3, 1], [2, -4]])
        w_tot.append(w_in)
        for site in range(2, self.L):
            if site % 2 == 0:
                w = ncon(
                    [w_loc, w_even, w_loc, w_odd],
                    [[1, -6], [-2, -4, 2, 1], [3, 2], [-1, -3, -5, 3]],
                ).reshape(
                    w_odd.shape[0] * w_even.shape[0],
                    w_odd.shape[1] * w_even.shape[1],
                    w_odd.shape[2],
                    w_even.shape[3],
                )
            else:
                w = ncon(
                    [w_odd, w_loc, w_even, w_loc],
                    [[-2, -4, 3, -6], [2, 3], [-1, -3, 1, 2], [-5, 1]],
                ).reshape(
                    w_odd.shape[0] * w_even.shape[0],
                    w_odd.shape[1] * w_even.shape[1],
                    w_even.shape[2],
                    w_odd.shape[3],
                )

            w_tot.append(w)

        w_tot.append(w_fin)
        self.w = w_tot
        return self

    # -------------------------------------------------
    # Observables, order parameters
    # -------------------------------------------------
    def order_param(self, op: np.ndarray=None, site: int=None, l: int=None, direction: str=None):
        """
        order_param

        This function selects which order parameter to use according to the
        studied model. Here you can add other order parameters that you have
        independently defined in the class.

        """
        if self.model == "Ising":
            self.order_param_Ising(op=op)
        
        elif self.model == "ANNNI":
            self.order_param_Ising(op=op)

        elif self.model == "Cluster":
            self.order_param_Ising(op=op)

        elif self.model == "Z2_two_ladder":
            self.order_param_Z2()

        elif self.model == "Z2_dual":
            self.order_param_Z2_dual()

        return self

    def order_param_Ising(self, op):
        """
        order_param_Ising

        This function defines the MPO order parameter for the 1D transverse field Ising model.
        It takes the same MPO for all sites.

        op: np.ndarray - operator that constitute with the order parameter of the theory.
            It depends on the choice of the basis for Ising Hamiltonian

        """
        I = np.eye(2)
        O = np.zeros((2, 2))
        op = sparse_pauli_z(n=0, L=1).toarray()
        w_tot = []
        for _ in range(self.L):
            w_mag = np.array([[I, op], [O, I]])
            w_tot.append(w_mag)
        self.w = w_tot
        return self
    
    def order_param_Z2(self):
        """
        order_param_Z2

        This function defines the MPO order parameter for the (2D) pure Z2 LGT model.
        It takes different MPOs among sites.

        """
        I = np.eye(2)
        O = np.zeros((2, 2))
        X = np.array([[0, 1], [1, 0]])
        w_tot = []
        for i in range(self.L):
            if i < (self.L // 2):
                beta = 1
                if i == 0:
                    alpha = 1
                else:
                    alpha = 0
                gamma = alpha
            else:
                beta = 0
                alpha = beta
                gamma = 1
            w = np.array(
                [
                    [I, O, alpha * X, O],
                    [O, O, O, O],
                    [
                        O,
                        O,
                        beta * X,
                        gamma * X @ (la.matrix_power(X, (1 - alpha))),
                    ],
                    [O, O, O, I],
                ]
            )
            w_tot.append(w)
        self.w = w_tot
        return self

    def order_param_Z2_dual(self):
        """
        order_param_Z2_dual

        This function defines the MPO order parameter for the Z2 pure lattice gauge theory,
        on the dual lattice. It is equivalent to a 2D transverse field Ising model.

        """

        self.Z2.mpo_skeleton(aux_dim=2)

        mpo_tot = []
        for mpo_site in range(self.Z2.L-1):
            if mpo_site == 0 or mpo_site == (self.Z2.L-2): # or mpo_site == 1 or mpo_site == (self.Z2.L-3)
                self.Z2.mpo_skeleton(aux_dim=2)
            else:
                for l in range(1,self.Z2.l-1):
                    self.Z2.mpo[0,-1] += sparse_pauli_z(n=l, L=self.Z2.l).toarray()
            mpo_tot.append(self.Z2.mpo)
            self.Z2.mpo_skeleton(aux_dim=2)
                    
        self.Z2.mpo = mpo_tot

        self.w = self.Z2.mpo
        return self

    def local_param(self, site: None, op: np.ndarray=None):
        """
        local_param

        This function selects which local parameter to use according to the
        studied model. Here you can add other local parameters that you have
        independently defined in the class.

        """
        if self.model == "Ising":
            self.single_operator_Ising(site=site, op=op)

        elif self.model == "ANNNI":
            self.single_operator_ANNNI(site=site)

        elif self.model == "Z2_dual":
            self.single_operator_Z2_dual(site=site, l=op)

        return self

    def single_operator_Ising(self, site, long: str="X"):
        """
        single_operator_Ising

        This function computes a local operator (op) for the 1D Ising model
        on a certain arbitrary site.

        site: int - local site where the operator acts
        op: np.ndarray - operator acting on the local site

        """
        I = identity(2, dtype=complex).toarray()
        O = csc_array((2, 2), dtype=complex).toarray()
        if long == "Z":
            long_op = sparse_pauli_z(n=0, L=1).toarray()
        elif long == "X":
            long_op = sparse_pauli_x(n=0, L=1).toarray()
        w_tot = []
        w_init = np.array([[I, O], [O, I]])
        for i in range(self.L):
            w_mag = w_init
            if i == site - 1:
                w_mag[0,-1] = long_op
        
            w_tot.append(w_mag)
        self.w = w_tot
        return self
    
    def single_operator_ANNNI(self, site, long: str="X"):
        """
        single_operator_Ising

        This function computes a local operator (op) for the 1D Ising model
        on a certain arbitrary site.

        site: int - local site where the operator acts
        op: np.ndarray - operator acting on the local site

        """
        I = identity(2, dtype=complex).toarray()
        O = csc_array((2, 2), dtype=complex).toarray()
        if long == "Z":
            long_op = sparse_pauli_z(n=0, L=1).toarray()
        elif long == "X":
            long_op = sparse_pauli_x(n=0, L=1).toarray()
        w_tot = []
        w_init = np.array([[I, O], [O, I]])
        for i in range(self.L):
            w_mag = w_init
            if i == site - 1:
                w_mag[0,-1] = long_op
        
            w_tot.append(w_mag)
        self.w = w_tot
        return self

    def single_operator_Z2_dual(self, site, l):
        """
        order_param_Z2_dual

        This function defines the MPO order parameter for the Z2 pure lattice gauge theory,
        on the dual lattice. It is equivalent to a 2D transverse field Ising model.

        """
        self.Z2.local_site_observable_Z2_dual(mpo_site=site, l=l)
        self.w = self.Z2.mpo
        return self

    def mps_local_exp_val(self, op):
        chain = []
        self.clear_envs()
        for i in range(1, self.L + 1):
            self.single_operator_Ising(site=i, op=op)
            self.envs(site=i)
            chain.append(self.braket(site=i))
        self.clear_envs()
        return chain

    def electric_field_Z2(self, E):
        """
        electric_field_Z2

        This function finds the mpo for the electric field in the direct lattice of a Z2 theory.
        To reconstruct the field in the direct lattices we need functions to compute the 
        borders and the bulk fields, weighted for the appropriate charges.

        """
        # let us find the observables for the boudary fields
        i = 0
        for mpo_site in range(self.Z2.L-1):
            j = 0
            if mpo_site == 0 or mpo_site == (self.Z2.L-2):
                E_v = []
                for l in range(self.Z2.l):
                    self.Z2.local_observable_Z2_dual(mpo_site=mpo_site, l=l)
                    coeff = self.Z2.charge_coeff_v_edge(mpo_site=mpo_site+i, l=l)
                    self.w = self.Z2.mpo.copy()
                    E_v.append(coeff * self.mpo_first_moment().real)
                    # E_v.append(self.mpo_first_moment().real)
                E[1::2, (mpo_site+i)*2] = E_v
                i = 1
            
            for l in [0,self.Z2.l-1]:
                self.Z2.local_observable_Z2_dual(mpo_site=mpo_site, l=l)
                coeff = self.Z2.charge_coeff_v(mpo_site=mpo_site, l=l)
                self.w = self.Z2.mpo.copy()
                E[(l+j)*2,mpo_site*2+1] = coeff * self.mpo_first_moment().real
                # E[(l+j)*2,mpo_site*2+1] = self.mpo_first_moment().real
                j = 1
            
        # now we can obtain the bulk values given by the zz interactions
        # vertical
        for l in range(self.Z2.l-1):
            E_v = []
            for mpo_site in range(self.Z2.L-1):
                self.Z2.zz_observable_Z2_dual(mpo_site=mpo_site, l=l, direction="vertical")
                self.w = self.Z2.mpo.copy()
                E_v.append(self.mpo_first_moment().real)
            E[(l+1)*2, 1::2] = E_v
        # horizontal
        for l in range(self.Z2.l):
            E_h = []
            for mpo_site in range(self.Z2.L-2):
                self.Z2.zz_observable_Z2_dual(mpo_site=mpo_site, l=l, direction="horizontal")
                coeff = self.Z2.charge_coeff_interaction(n=l+1,mpo_site=mpo_site)
                self.w = self.Z2.mpo.copy()
                E_h.append(coeff * self.mpo_first_moment().real)
            E_h.append(E[(l*2+1), -1])
            E[(l*2+1), 2::2] = E_h
        
        return E

    def connected_correlator(self, site, lad):
        """
        connected_correlator

        This function computes the correlator between a reference link
        and all the others links in the whole column of our ladder system.
        The reference link is located at a certain site and ladder.
        The correlator is connected because we subtract the expecation values
        of the two links we are referring to. E.g. <E_ref,E_r> - <E_ref><E_r>

        site: int - site we will use for the vertical section of the ladder system
        lad: int - It refers to the upper link of a ladder we use as reference

        """
        E_corr = []
        # find the exp val for the reference link
        self.Z2.zz_observable_Z2_dual(mpo_site=site, l=lad-1, direction="vertical")
        self.w = self.Z2.mpo.copy()
        E_lad = self.mpo_first_moment().real
        print(f"E_0: {E_lad}")
        for link in range(self.Z2.l+1):
            # if link != lad:
            if link in [0, self.Z2.l]:
                if link == 0:
                    l = link
                elif link == self.Z2.l:
                    l = link - 1                    
                # find the exp val for the link separated by r
                self.Z2.local_observable_Z2_dual(mpo_site=site, l=l)
                coeff = self.Z2.charge_coeff_v(mpo_site=site, l=l)
                self.w = self.Z2.mpo.copy()
                E_r = coeff * self.mpo_first_moment().real
                print(f"E_r: {E_r}")
                # E_r = self.mpo_first_moment().real
            else:
                l = link - 1
                # find the exp val for the link separated by r
                self.Z2.zz_observable_Z2_dual(mpo_site=site, l=l, direction="vertical")
                self.w = self.Z2.mpo.copy()
                E_r = self.mpo_first_moment().real
                print(f"E_r: {E_r}")

            # find the exp val of the correlator between reference and r link
            self.Z2.correlator(site=[site],ladders=[lad,link])
            self.w = self.Z2.mpo.copy()
            E_lad_r = self.mpo_first_moment().real
            print(f"E_0-r: {E_lad_r}")

            E_corr.append(E_lad_r - (E_lad * E_r))
            print(E_lad_r - (E_lad * E_r))

        return E_corr
    
    def electric_energy_density_Z2(self, site):
        """
        electric_energy_density_Z2

        This function computes the electric energy density for the Z2 model
        in the whole column of our ladder system.
        The column is located at a certain site.
        We can have a "connected" energy density if we subtract the expecation values
        of the plaquettes we are referring to with the expectation values of the vacuum state. 
        E.g. <q,q'|el_en_density|q,q'> - <0|el_en_density|0> (we do not do that here)

        site: int - site we will use for the vertical section of the ladder system
        lad: int - It refers to the upper link of a ladder we use as reference

        """
        E_en_density = []
        for lad in range(self.Z2.l):
            if lad in [0, self.Z2.l-1]:
                if lad == 0:
                    l = lad
                    # find the sigma_4^x first
                    self.Z2.zz_observable_Z2_dual(mpo_site=site, l=l, direction="vertical")
                    self.w += [(self.h / 2) * mpo for mpo in self.Z2.mpo].copy()
                elif lad == self.Z2.l-1:
                    l = lad    
                    # find the sigma_2^x last
                    self.Z2.zz_observable_Z2_dual(mpo_site=site, l=l-1, direction="vertical")
                    self.w += [(self.h / 2) * mpo for mpo in self.Z2.mpo].copy()
                # find the sigma_2^x first or sigma_4^x last
                self.Z2.local_observable_Z2_dual(mpo_site=site, l=l)
                coeff = self.Z2.charge_coeff_v(mpo_site=site, l=l)
                self.w += [(self.h * coeff) * mpo for mpo in self.Z2.mpo].copy() # times 2 because we do not share this link with any other plaquette
                
            else:
                l = lad - 1
                # find the sigma_2^x
                self.Z2.zz_observable_Z2_dual(mpo_site=site, l=l, direction="vertical")
                self.w += [(self.h / 2) * mpo for mpo in self.Z2.mpo].copy()
                # find the sigma_4^x
                self.Z2.zz_observable_Z2_dual(mpo_site=site, l=l+1, direction="vertical")
                self.w += [(self.h / 2) * mpo for mpo in self.Z2.mpo].copy()


            # find the sigma_1^x
            self.Z2.zz_observable_Z2_dual(mpo_site=site, l=lad, direction="horizontal")
            coeff = self.Z2.charge_coeff_interaction(n=lad+1,mpo_site=site)
            self.w += [(self.h / 2) * coeff * mpo for mpo in self.Z2.mpo].copy()
            # find the sigma_3^x
            self.Z2.zz_observable_Z2_dual(mpo_site=site+1, l=lad, direction="horizontal")
            coeff = self.Z2.charge_coeff_interaction(n=lad+1,mpo_site=site+1)
            self.w += [(self.h / 2) * coeff * mpo for mpo in self.Z2.mpo].copy()
            mpo_split = np.array_split(np.asarray(self.w), self.L)
            mpo_summed = np.sum(mpo_split, axis=1)
            self.w = mpo_summed
            eed = self.mpo_first_moment().real
            print(f"Electric energy density: {eed}")

            E_en_density.append(eed)

        return E_en_density

    # -------------------------------------------------
    # Manipulation of MPOs
    # -------------------------------------------------
    def mpo_dagger(self):
        w_tot = []
        for w in self.w:
            w_dag = w.conjugate()
            w_tot.append(w_dag)
        self.w_dag = w_tot
        return self

    def mpo_O_dag_O(self):
        """
        mpo_Ising_O_dag_O

        This function creates an mpo given by the product of a previous mpo O
        with its dagger. If O is hermitian then it is equal to perform O^2.

        """
        ws = self.w
        ws_dag = self.w_dag

        w_tot = []

        for w, w_dag in zip(ws, ws_dag):
            w = ncon([w, w_dag], [[-1, -3, -5, 1], [-2, -4, 1, -6]]).reshape(
                (
                    w.shape[0] * w_dag.shape[0],
                    w.shape[1] * w_dag.shape[1],
                    w.shape[2],
                    w.shape[3],
                )
            )
            w_tot.append(w)

        self.w = w_tot
        return self

    def mpo_to_mps(self, ancilla=True):
        if ancilla:
            array = self.ancilla_sites
            for i in range(self.L):
                self.ancilla_sites[i] = ncon(
                    [array[i], self.w[i]],
                    [
                        [-1, 2, -4],
                        [-2, -5, 2, -3],
                    ],
                ).reshape(
                    (
                        array[i].shape[0] * self.w[i].shape[0],
                        self.d,
                        array[i].shape[2] * self.w[i].shape[1],
                    )
                )
        else:
            array = self.sites
            for i in range(self.L):
                self.sites[i] = ncon(
                    [array[i], self.w[i]],
                    [
                        [-1, 2, -4],
                        [-2, -5, 2, -3],
                    ],
                ).reshape(
                    (
                        array[i].shape[0] * self.w[i].shape[0],
                        self.d,
                        array[i].shape[2] * self.w[i].shape[1],
                    )
                )
        return self

    # -------------------------------------------------
    # Help functions relative to DMRG and TEBD
    # -------------------------------------------------
    def envs(
        self,
        site: int = 1,
        sm: bool = False,
        fm: bool = False,
        opt: bool = False,
        ancilla: bool = False,
        mixed: bool = False,
        rev: bool = False,
    ):
        """
        envs

        This function computes the left and right environments to compute the effective Hamiltonian.
        In addition, computes the environments to calculate the second and fourth moment of a mpo.

        sm: bool - Compute the left and right environments for the second moment of self.w. Default False
        fm: bool - Compute the left and right environments for the fourth moment of self.w. Default False

        """
        a = np.array([1])
        D = self.w[0].shape[0]
        v_l = np.zeros(D)
        v_l[0] = 1
        v_r = np.zeros(D)
        v_r[-1] = 1
        aux = self.sites[0].shape[0]
        l = np.zeros(aux)
        l[0] = 1
        r = np.zeros(aux)
        r[-1] = 1
        E_r = ncon([r.T, v_r.T, r.T], [[-1], [-2], [-3]])
        E_l = ncon([l, v_l, l], [[-1], [-2], [-3]])

        if opt:
            special = np.array([[1], [0]])
            E_r = ncon([special, E_r, special], [[-1, 1], [1, -2, 2], [-3, 2]])
            special = np.array([[1, 0]])
            E_l = ncon([special, E_l, special], [[1, -1], [1, -2, 2], [2, -3]])

        if sm:
            array = self.sites
            w = self.w
            E_l = ncon([a, v_l, v_l, a],[[-1],[-2],[-3],[-4]])
            E_r = ncon([a, v_r.T, v_r.T, a],[[-1],[-2],[-3],[-4]])
            self.env_left.append(E_l)
            self.env_right.append(E_r)

            for i in range(1, site):
                E_l = ncon([E_l, ancilla_array[i - 1]], [[1, -3, -4, -5], [1, -2, -1]])
                E_l = ncon([E_l, w[i - 1]], [[-1, 1, 2, -4, -5], [2, -2, 1, -3]])
                E_l = ncon([E_l, w[i - 1]], [[-1, -2, 1, 2, -5], [2, -3, 1, -4]])
                E_l = ncon(
                    [E_l, array[i - 1].conjugate()], [[-1, -2, -3, 1, 2], [2, 1, -4]]
                )
                self.env_left.append(E_l)

            for i in range(self.L, site, -1):
                E_r = ncon([E_r, array[i - 1]], [[1, -3, -4, -5], [-1, -2, 1]])
                E_r = ncon([E_r, w[i - 1]], [[-1, 1, 2, -4, -5], [-2, 2, 1, -3]])
                E_r = ncon([E_r, w[i - 1]], [[-1, -2, 1, 2, -5], [-3, 2, 1, -4]])
                E_r = ncon(
                    [E_r, array[i - 1].conjugate()], [[-1, -2, -3, 1, 2], [-4, 1, 2]]
                )
                self.env_right.append(E_r) 

        elif fm:
            array = self.sites
            w = self.w
            E_l = ncon([a, v_l, v_l, v_l, v_l, a],[[-1],[-2],[-3],[-4],[-5],[-6]])
            E_r = ncon([a, v_r.T, v_r.T,v_r.T, v_r.T, a],[[-1],[-2],[-3],[-4],[-5],[-6]])
            self.env_left.append(E_l)
            self.env_right.append(E_r)

            for i in range(1, site):
                E_l = ncon([E_l, ancilla_array[i - 1]], [[1, -3, -4, -5, -6, -7], [1, -2, -1]])
                E_l = ncon([E_l, w[i - 1]], [[-1, 1, 2, -4, -5, -6, -7], [2, -2, 1, -3]])
                E_l = ncon([E_l, w[i - 1]], [[-1, -2, 1, 2, -5, -6, -7], [2, -3, 1, -4]])
                E_l = ncon([E_l, w[i - 1]], [[-1, -2, -3, 1, 2, -6, -7], [2, -4, 1, -5]])
                E_l = ncon([E_l, w[i - 1]], [[-1, -2, -3, -4, 1, 2, -7], [2, -5, 1, -6]])
                E_l = ncon(
                    [E_l, array[i - 1].conjugate()], [[-1, -2, -3, -4, -5, 1, 2], [2, 1, -6]]
                )
                self.env_left.append(E_l)

            for i in range(self.L, site, -1):
                E_r = ncon([E_r, array[i - 1]], [[1, -3, -4, -5, -6, -7], [-1, -2, 1]])
                E_r = ncon([E_r, w[i - 1]], [[-1, 1, 2, -4, -5, -6, -7], [-2, 2, 1, -3]])
                E_r = ncon([E_r, w[i - 1]], [[-1, -2, 1, 2, -5, -6, -7], [-3, 2, 1, -4]])
                E_r = ncon([E_r, w[i - 1]], [[-1, -2, -3, 1, 2, -6, -7], [-4, 2, 1, -5]])
                E_r = ncon([E_r, w[i - 1]], [[-1, -2, -3, -4, 1, 2, -7], [-5, 2, 1, -6]])
                E_r = ncon(
                    [E_r, array[i - 1].conjugate()], [[-1, -2, -3, -4, -5, 1, 2], [-6, 1, 2]]
                )
                self.env_right.append(E_r)

        elif mixed:
            env_right = []
            env_left = []

            env_right.append(E_r)
            env_left.append(E_l)
            if rev:
                array = self.ancilla_sites
                ancilla_array = self.sites
            else:
                array = self.sites.copy()
                ancilla_array = self.ancilla_sites.copy()
            w = self.w.copy()

            for i in range(1, site):
                E_l = ncon([E_l, ancilla_array[i - 1]], [[1, -3, -4], [1, -2, -1]])
                E_l = ncon([E_l, w[i - 1]], [[-1, 1, 2, -4], [2, -2, 1, -3]])
                E_l = ncon(
                    [E_l, array[i - 1].conjugate()], [[-1, -2, 1, 2], [2, 1, -3]]
                )
                env_left.append(E_l)

            for j in range(self.L, site, -1):
                E_r = ncon([E_r, ancilla_array[j - 1]], [[1, -3, -4], [-1, -2, 1]])
                E_r = ncon([E_r, w[j - 1]], [[-1, 1, 2, -4], [-2, 2, 1, -3]])
                E_r = ncon(
                    [E_r, array[j - 1].conjugate()], [[-1, -2, 1, 2], [-3, 1, 2]]
                )
                env_right.append(E_r)
            if rev:
                self.env_right_sm = env_right
                self.env_left_sm = env_left
            else:
                self.env_right = env_right
                self.env_left = env_left
        else:
            self.env_right.append(E_r)
            self.env_left.append(E_l)
            array = self.sites
            if ancilla:
                array = self.ancilla_sites
            w = self.w
            for i in range(1, site):
                E_l = ncon([E_l, array[i - 1]], [[1, -3, -4], [1, -2, -1]])
                E_l = ncon([E_l, w[i - 1]], [[-1, 1, 2, -4], [2, -2, 1, -3]])
                E_l = ncon(
                    [E_l, array[i - 1].conjugate()], [[-1, -2, 1, 2], [2, 1, -3]]
                )
                self.env_left.append(E_l)

            for i in range(self.L, site, -1):
                E_r = ncon([E_r, array[i - 1]], [[1, -3, -4], [-1, -2, 1]])
                E_r = ncon([E_r, w[i - 1]], [[-1, 1, 2, -4], [-2, 2, 1, -3]])
                E_r = ncon(
                    [E_r, array[i - 1].conjugate()], [[-1, -2, 1, 2], [-3, 1, 2]]
                )
                self.env_right.append(E_r)
        return self

    def H_eff(self, site):
        """
        H_eff

        This function contracts the left and right environments with the class mpos self.w
        and self.w_2. Then, we reshape the effective Hamiltonian as a matrix.

        site: int - site to optimize

        """
        # H_eff_time = time.perf_counter()
        H = ncon(
            [self.env_left[-1], self.w[site - 1]],
            [
                [-1, 1, -5],
                [1, -3, -2, -4],
            ]
        )
        H = ncon(
            [H, self.env_right[-1]],
            [
                [-1, -2, 1, -5, -4],
                [-3 , 1, -6]
            ]
        )
        # print(f"Time of H_eff contraction: {time.perf_counter()-H_eff_time}")

        # reshape_time = time.perf_counter()
        H = H.reshape(
            self.env_left[-1].shape[0] * self.d * self.env_right[-1].shape[0],
            self.env_left[-1].shape[2] * self.d * self.env_right[-1].shape[2],
        )
        # print(f"Time of H_eff reshaping: {time.perf_counter()-reshape_time}")
        # print(H.shape)

        return H

    def eigensolver(self, v0:np.ndarray=None, H_eff: np.ndarray=None):
        """
        eigensolver

        This function solves the eigenvalue problem for the effective Hamiltonian
        of both <H> and <H^2>. It extract directly the smallest eigenvalue and
        its relative eigenvector is reshaped to update the state. An initial guess
        of this state can be specified.

        H_eff: np.ndarray - the matrix we are interested in solving the eigenvalue problem
        site: int - site we are optimizing
        v0: np.ndarray - a guessing for the eigenvector. You can use the updated state in that
            site. Default Nones

        """
        # time_eig = time.perf_counter()
        if type(H_eff) == type(None):
            A = TensorMultiplierOperator((
                                self.env_left[-1].shape[0] * self.d * self.env_right[-1].shape[0],
                                self.env_left[-1].shape[2] * self.d * self.env_right[-1].shape[2],
                                ), matvec=self.mv, dtype=np.complex128)
            # print(f"shape of A: {A.shape}")
            if A.shape[0] == 2:
                H = self.H_eff(site=self.site)
                e, v = la.eigh(H)
            else:
                v0 = self.sites[self.site - 1]
                # print(f"v0 at site {self.site - 1} has shape: {v0.shape}")
                e, v = spla.eigsh(A, k=1, v0=v0, which='SA')
        else:
            e, v = spla.eigsh(H_eff, k=1, which="SA", v0=v0)
        # np.savetxt(
        #     f"/Users/fradm/mps/results/times_data/eigsh_eigensolver_site_{site}_h_{self.h:.2f}",
        #     [time.perf_counter() - time_eig],
        # )
        # print(f"Time of eigsh during eigensolver for site {site}: {time.perf_counter()-time_eig}")
        e_min = e[0].real
        eigvec = np.array(v[:,0])

        self.sites[self.site - 1] = eigvec.reshape(
            self.env_left[-1].shape[0], self.d, self.env_right[-1].shape[0]
        )
        return e_min

    def update_state(
        self,
        sweep: str,
        site: int,
        trunc_tol: bool = True,
        trunc_chi: bool = False,
        schmidt_tol: float = 1e-15,
    ):
        """
        update_state

        This function updates the self.a and self.b lists of tensors composing
        the mps. The update depends on the sweep direction. We take the self.m
        extracted from the eigensolver and we decomposed via svd.

        sweep: string - direction of the sweeping. Could be "left" or "right"
        site: int - indicates which site the DMRG is optimizing
        trunc: bool - if True will truncate the the Schmidt values and save the
                state accordingly.
        schmidt_tol: float - the tolerance accepted to truncate the Schmidt values
        precision: int - indicates the precision of the parameter h

        """
        if sweep == "right":
            # we want to write M (left,d,right) in LFC -> (left*d,right)
            m = self.sites[site - 1].reshape(
                self.env_left[-1].shape[2] * self.d, self.env_right[-1].shape[2]
            )
            u, s, v = la.svd(m, full_matrices=False)
            if trunc_tol:
                condition = s >= schmidt_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / la.norm(s_trunc)
                bond_l = u.shape[0] // self.d
                u = u.reshape(bond_l, self.d, u.shape[1])
                u = u[:, :, : len(s)]
                v = v[: len(s), :]
            elif trunc_chi:
                s_trunc = s[: self.chi]
                s = s_trunc / la.norm(s_trunc)
                bond_l = u.shape[0] // self.d
                u = u.reshape(bond_l, self.d, u.shape[1])
                u = u[:, :, : len(s)]
                v = v[: len(s), :]
            else:
                u = u.reshape(
                    self.env_left[-1].shape[2], self.d, self.env_right[-1].shape[2]
                )
            next_site = ncon(
                [np.diag(s), v, self.sites[site]],
                [
                    [-1, 1],
                    [1, 2],
                    [2, -2, -3],
                ],
            )
            self.sites[site - 1] = u
            self.sites[site] = next_site

        elif sweep == "left":
            # we want to write M (left,d,right) in RFC -> (left,d*right)
            m = self.sites[site - 1].reshape(
                self.env_left[-1].shape[2], self.d * self.env_right[-1].shape[2]
            )
            u, s, v = la.svd(m, full_matrices=False)
            if trunc_tol:
                condition = s >= schmidt_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / la.norm(s_trunc)
                bond_r = v.shape[1] // self.d
                v = v.reshape(v.shape[0], self.d, bond_r)
                v = v[: len(s), :, :]
                u = u[:, : len(s)]
            elif trunc_chi:
                s_trunc = s[: self.chi]
                s = s_trunc / la.norm(s_trunc)
                bond_r = v.shape[1] // self.d
                v = v.reshape(v.shape[0], self.d, bond_r)
                v = v[: len(s), :, :]
                u = u[:, : len(s)]
            else:
                v = v.reshape(
                    self.env_left[-1].shape[2], self.d, self.env_right[-1].shape[2]
                )
            next_site = ncon(
                [self.sites[site - 2], u, np.diag(s)],
                [
                    [-1, -2, 1],
                    [1, 2],
                    [2, -3],
                ],
            )
            self.sites[site - 1] = v
            self.sites[site - 2] = next_site

        return s

    def update_envs(self, sweep, site, mixed=False, rev=False):
        """
        update_envs

        This function updates the left and right environments for the next
        site optimization performed by the eigensolver. After the update of the mps
        in LCF and RCF we can compute the new environment and throw the one we do not need.

        sweep: string - direction of the sweeping. Could be "left" or "right"
        site: int - site we are optimizing

        """
        if sweep == "right":
            # time_upd_env = time.perf_counter()
            array = self.sites[site - 1]
            ancilla_array = array
            w = self.w[site - 1]
            if rev:
                E_l = self.env_left_sm[-1]
                array = self.ancilla_sites[site - 1]
                ancilla_array = self.sites[site - 1]
            else:
                if mixed:
                    ancilla_array = self.ancilla_sites[site - 1]
                E_l = self.env_left[-1]
            E_l = ncon([E_l, ancilla_array], [[1, -3, -4], [1, -2, -1]])
            E_l = ncon([E_l, w], [[-1, 1, 2, -4], [2, -2, 1, -3]])
            E_l = ncon([E_l, array.conjugate()], [[-1, -2, 1, 2], [2, 1, -3]])
            if rev:
                self.env_left_sm.append(E_l)
                self.env_right_sm.pop(-1)
            else:
                self.env_left.append(E_l)
                self.env_right.pop(-1)
            # np.savetxt(f"results/times_data/update_env_{site}_h_{self.h:.2f}", [time.perf_counter()-time_upd_env])

        if sweep == "left":
            array = self.sites[site - 1]
            ancilla_array = array
            w = self.w[site - 1]
            if rev:
                E_r = self.env_right_sm[-1]
                array = self.ancilla_sites[site - 1]
                ancilla_array = self.sites[site - 1]
            else:
                if mixed:
                    ancilla_array = self.ancilla_sites[site - 1]
                E_r = self.env_right[-1]
            E_r = ncon([E_r, ancilla_array], [[1, -3, -4], [-1, -2, 1]])
            E_r = ncon([E_r, w], [[-1, 1, 2, -4], [-2, 2, 1, -3]])
            E_r = ncon([E_r, array.conjugate()], [[-1, -2, 1, 2], [-3, 1, 2]])
            if rev:
                self.env_right_sm.append(E_r)
                self.env_left_sm.pop(-1)
            else:
                self.env_right.append(E_r)
                self.env_left.pop(-1)

        return self

    def mv(self, v):
        v = v.reshape(self.env_left[-1].shape[0], self.d, self.env_right[-1].shape[0])
        res = ncon([self.env_left[-1],v],[[1,-3,-4],[1,-2,-1]])
        res = ncon([res, self.w[self.site-1]],[[-1,1,2,-4],[2,-2,1,-3]])
        res = ncon([res,self.env_right[-1]],[[1,2,-2,-1],[1,2,-3]])
        res = res.flatten()
        return res

    def DMRG(
        self,
        trunc_tol: bool,
        trunc_chi: bool,
        long: str="X",
        trans: str="Z",
        schmidt_tol: float = 1e-15,
        conv_tol: float = 1e-10,
        n_sweeps: int = 2,
        bond: bool = True,
        where: int = -1,
    ):
        energies = []
        sweeps = ["right", "left"]
        sites = np.arange(1, self.L + 1).tolist()

        self.mpo(long=long, trans=trans)
        self.envs()

        iter = 1
        H = None
        v0 = None

        t_start = time.perf_counter()
        for n in range(n_sweeps):
            print(f"Sweep n: {n}\n")
            entropy = []
            schmidt_vals = []
            for i in range(self.L - 1):
                # print(f"Site: {sites[i]}\n")
                # t_start = time.perf_counter()
                if trunc_tol == True:
                    H = self.H_eff(sites[i])
                # print(f"Time effective Ham: {abs(time.perf_counter()-t_start)}")
                # t_start = time.perf_counter()
                self.site = sites[i]
                energy = self.eigensolver(site=sites[i], v0=v0, H_eff=H) # , v0=v0
                # energy = self.eigensolver(H_eff=H, site=sites[i], v0=v0) # , v0=v0
                # print(f"Time eigensolver: {abs(time.perf_counter()-t_start)}")
                energies.append(energy)
                # t_start = time.perf_counter()
                s = self.update_state(
                    sweeps[0], sites[i], trunc_tol, trunc_chi, schmidt_tol
                )
                # print(f"Time update state: {abs(time.perf_counter()-t_start)}")
                if bond:
                    if sites[i] - 1 == where:
                        entr = von_neumann_entropy(s)
                        entropy.append(entr)
                        schmidt_vals.append(s)
                else:
                    entr = von_neumann_entropy(s)
                    entropy.append(entr)
                    schmidt_vals.append(s)

                # t_start = time.perf_counter()
                self.update_envs(sweeps[0], sites[i])
                # print(f"Time update envs: {abs(time.perf_counter()-t_start)}")
                iter += 1

            if ((n % 2) - 1) == 0:
                energy_dist = np.abs(energies[-1] - energies[-2])/energies[-1]
                if energy_dist < conv_tol:
                    break
            
            # print("reversing the sweep")
            sweeps.reverse()
            sites.reverse()

        t_dmrg = abs(time.perf_counter()-t_start)
        if energy_dist < conv_tol:
            print("##############################")
            print(
                f"The energy between the two last updated states converged\n"
                + f"to an order of {conv_tol} after:\n"
                + f"{n} sweeps at site {sites[i]}\n"
                + f"total iterations {iter}\n"
                + f"total time: {t_dmrg}"
            )
            print("##############################")
        else:
            print("##############################")
            print(
                f"The energy between the two last updated states converged\n"
                + f"to an order of {energy_dist}\n"
                + f"instead of the convergence tolerance {conv_tol}\n"
                + f"total time: {t_dmrg}"
            )
            print("##############################")
        return energies, entropy, schmidt_vals, t_dmrg

    def environments_ev(self, site):
        a = np.array([1])
        E_l = ncon([a, a], [[-1], [-2]])
        E_r = E_l
        env_right = []
        env_left = []

        env_right.append(E_r)
        env_left.append(E_l)
        array = self.sites
        ancilla_array = self.ancilla_sites

        for i in range(1, site):
            E_l = ncon(
                [E_l, ancilla_array[i - 1], array[i - 1].conjugate()],
                [
                    [1, 3],
                    [1, 2, -1],
                    [3, 2, -2],
                ],
            )
            env_left.append(E_l)

        for j in range(self.L, site, -1):
            E_r = ncon(
                [E_r, ancilla_array[j - 1], array[j - 1].conjugate()],
                [
                    [1, 3],
                    [-1, 2, 1],
                    [-3, 2, 3],
                ],
            )
            env_right.append(E_r)

        self.env_right = env_right
        self.env_left = env_left
        return self

    def compute_M(self, site, mpo=True):
        """
        _compute_M

        This function computes the rank-3 tensor, in a specific site,
        given by the contraction of our variational state (phi) saved in self.sites,
        and the uncompressed state (psi) saved in self.ancilla_sites.

        site: int - site where to execute the tensor contraction

        """
        if mpo:
            M = ncon(
                [
                    self.env_left[-1],
                    self.ancilla_sites[site - 1],
                    self.w[site - 1],
                    self.env_right[-1],
                ],
                [[1, 3, -1], [1, 5, 2], [3, 4, 5, -2], [2, 4, -3]],
            )
        else:
            M = ncon(
                [self.env_left[-1], self.ancilla_sites[site - 1], self.env_right[-1]],
                [[1, -1], [1, -2, 2], [2, -3]],
            )
        return M

    def error(self, site, M, N_anc):
        A_dag_M = ncon([M, M.conjugate()], [[1, 2, 3], [1, 2, 3]])
        # print("Norm of variational state (before state/env update):")
        A_dag_N_eff_A = self._compute_norm(site=site)
        error = A_dag_N_eff_A - 2 * A_dag_M.real + N_anc
        return error

    def update_state_ev(
        self,
        sweep: str,
        site: int,
        trunc_tol: bool,
        trunc_chi: bool,
        schmidt_tol: float = 1e-15,
    ):
        """
        update_state

        This function updates the state accoring to the sweeping direction and
        the truncation procedure, if applied. The state undergoes in any
        case a svd procedure to obtain the schmidt values and unitary matrices.

        sweep: string - direction of the sweeping. Could be "left" or "right"
        site: int - indicates which site the TEBD is optimizing
        trunc_tol: bool - if True will truncate the the Schmidt values according to
                        a tolerance value schmidt_tol
        trunc_chi: bool - if True will truncate the the Schmidt values according to
                        a maximum (fixed) bond dimension
        schmidt_tol: float - the tolerance accepted to truncate the Schmidt values
        precision: int - indicates the precision to save parameters

        """
        s_mid = 0
        if sweep == "right":
            # we want to write M (left,d,right) in LFC -> (left*d,right)
            m = self.sites[site - 1].reshape(
                self.sites[site - 1].shape[0] * self.d, self.sites[site - 1].shape[2]
            )
            u, s, v = la.svd(m, full_matrices=False)

            if trunc_tol:
                condition = s >= schmidt_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / la.norm(s_trunc)
                bond_l = u.shape[0] // self.d
                u = u.reshape(bond_l, self.d, u.shape[1])
                u = u[:, :, : len(s)]
                v = v[: len(s), :]
                if site == self.L // 2:
                    # print(f'Schmidt values:\n{s}')
                    s_mid = s
                    # np.savetxt(
                    #     f"/Users/fradm/Google Drive/My Drive/projects/0_ISING/results/bonds_data/schmidt_values_middle_chain_{self.model}_flip_{flip}_L_{self.L}_chi_{self.chi}_trotter_step_{trotter_step}_delta_{delta}",
                    #     s,
                    # )
            elif trunc_chi:
                s_trunc = s[: self.chi]
                s = s / la.norm(s_trunc)
                bond_l = u.shape[0] // self.d
                u = u.reshape(bond_l, self.d, u.shape[1])
                u = u[:, :, : len(s)]
                v = v[: len(s), :]
                if site == self.L // 2:
                    s_mid = s
                    # print(f'Schmidt values:\n{s}')
                    # np.savetxt(
                    #     f"/Users/fradm/Google Drive/My Drive/projects/0_ISING/results/bonds_data/schmidt_values_middle_chain_{self.model}_flip_{flip}_L_{self.L}_chi_{self.chi}_trotter_step_{trotter_step}_delta_{delta}",
                    #     s,
                    # )
            else:
                u = u.reshape(
                    self.sites[site - 1].shape[0], self.d, self.sites[site - 1].shape[2]
                )

            # print(f"Schmidt sum: {sum(s**2)}")
            next_site = ncon(
                [np.diag(s), v, self.sites[site]],
                [
                    [-1, 1],
                    [1, 2],
                    [2, -2, -3],
                ],
            )
            self.sites[site - 1] = u
            self.sites[site] = next_site

        elif sweep == "left":
            # we want to write M (left,d,right) in RFC -> (left,d*right)
            m = self.sites[site - 1].reshape(
                self.sites[site - 1].shape[0], self.d * self.sites[site - 1].shape[2]
            )
            u, s, v = la.svd(m, full_matrices=False)

            if trunc_tol:
                condition = s >= schmidt_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / la.norm(s_trunc)
                bond_r = v.shape[1] // self.d
                v = v.reshape(v.shape[0], self.d, bond_r)
                v = v[: len(s), :, :]
                u = u[:, : len(s)]
                if site == self.L // 2:
                    s_mid = s
                    # print(f'Schmidt values:\n{s}')
                    # np.savetxt(
                    #     f"/Users/fradm/Google Drive/My Drive/projects/0_ISING/results/bonds_data/schmidt_values_middle_chain_{self.model}_flip_{flip}_L_{self.L}_chi_{self.chi}_trotter_step_{trotter_step}_delta_{delta}",
                    #     s,
                    # )
            elif trunc_chi:
                s_trunc = s[: self.chi]
                s = s / la.norm(s_trunc)
                # print(f"Schmidt Values:\n{s}")
                bond_r = v.shape[1] // self.d
                v = v.reshape(v.shape[0], self.d, bond_r)
                v = v[: len(s), :, :]
                u = u[:, : len(s)]
                if site == self.L // 2:
                    s_mid = s
                    # print(f'Schmidt values:\n{s}')
                    # np.savetxt(
                    #     f"/Users/fradm/Google Drive/My Drive/projects/0_ISING/results/bonds_data/schmidt_values_middle_chain_{self.model}_flip_{flip}_L_{self.L}_chi_{self.chi}_trotter_step_{trotter_step}_delta_{delta}",
                    #     s,
                    # )
            else:
                v = v.reshape(
                    self.sites[site - 1].shape[0], self.d, self.sites[site - 1].shape[2]
                )
            # print(f"Schmidt sum: {sum(s**2)}")
            next_site = ncon(
                [self.sites[site - 2], u, np.diag(s)],
                [
                    [-1, -2, 1],
                    [1, 2],
                    [2, -3],
                ],
            )
            self.sites[site - 1] = v
            self.sites[site - 2] = next_site

        return s_mid

    def update_envs_ev(self, sweep, site):
        """
        update_envs

        This function updates the left and right environments for the next
        site optimization performed by the eigensolver. After the update of the mps
        in LCF and RCF we can compute the new environment and throw the one we do not need.

        sweep: string - direction of the sweeping. Could be "left" or "right"
        site: int - site we are optimizing

        """
        if sweep == "right":
            array = self.sites[site - 1]
            ancilla_array = self.ancilla_sites[site - 1]
            E_l = self.env_left[-1]
            E_l = ncon(
                [E_l, ancilla_array, array.conjugate()],
                [
                    [1, 3],
                    [1, 2, -1],
                    [3, 2, -3],
                ],
            )
            self.env_left.append(E_l)
            self.env_right.pop(-1)

        if sweep == "left":
            array = self.sites[site - 1]
            ancilla_array = self.ancilla_sites[site - 1]
            E_r = self.env_right[-1]
            E_r = ncon(
                [E_r, ancilla_array, array.conjugate()],
                [
                    [1, 3],
                    [-1, 2, 1],
                    [-3, 2, 3],
                ],
            )
            self.env_right.append(E_r)
            self.env_left.pop(-1)

        return self

    def compression(
        self,
        trunc_tol: bool,
        trunc_chi: bool,
        schmidt_tol: float = 1e-15,
        n_sweeps: int = 6,
        conv_tol: float = 1e-10,
        where: int = -1,
        bond: bool = True,
    ):
        """
        compression

        This function compress the mps self.sites by a variational method that
        tries to compress the uncompressed state in the ancilla_sites. We
        reduce the distance between these two last updated states.

        delta: float - the step in time of a trotter step
        trunc_tol: bool - if True will truncate the the Schmidt values according to
                        a tolerance value schmidt_tol
        trunc_chi: bool - if True will truncate the the Schmidt values according to
                        a maximum (fixed) bond dimension
        schmidt_tol: float - the tolerance accepted to truncate the Schmidt values
        n_sweeps: int - number of sweepings
        precision: int - precision of the floats to save
        err: bool - decide if we want to monitor the error or not. Default is false
        cov_tol: float - the convergence tolerance we use to stop the sweeping. By default 1e-7
        """
        sweeps = ["right", "left"]
        sites = np.arange(1, self.L + 1).tolist()
        errors = []

        N_anc = self._compute_norm(site=1, ancilla=True)

        # self.environments_ev(site=1)
        self.envs(site=1, mixed=True)
        iter = 1
        for n in range(n_sweeps):
            print(f"Sweep n: {n}\n")
            entropy = []
            for i in range(self.L - 1):
                # print(f"\n============= Site: {sites[i]} ===================\n")

                M = self.compute_M(sites[i], mpo=True)
                self.sites[sites[i] - 1] = M

                errs = self.error(site=sites[i], M=M, N_anc=N_anc)
                errors.append(np.abs(errs))

                # s = self.update_state_ev(
                #     sweeps[0],
                #     sites[i],
                #     delta,
                #     trotter_step,
                #     trunc_tol,
                #     trunc_chi,
                #     flip,
                #     schmidt_tol,
                #     precision,
                # )
                s = self.update_state(
                    sweep=sweeps[0],
                    site=sites[i],
                    trunc_tol=trunc_tol,
                    trunc_chi=trunc_chi,
                    schmidt_tol=schmidt_tol,
                )
                if bond:
                    if sites[i] - 1 == where:
                        s_mid = s
                        entr = von_neumann_entropy(s)
                        entropy.append(entr)
                else:
                    entr = von_neumann_entropy(s)
                    entropy.append(entr)
                # self.update_envs_ev(sweeps[0], sites[i])
                self.update_envs(sweeps[0], sites[i], mixed=True)

                # self.check_canonical(site=sites[i])
                iter += 1
                # norm_sites = self._compute_norm(site=1)

            sweeps.reverse()
            sites.reverse()

            if ((n % 2) - 1) == 0:
                err_dist = np.abs(errors[-1] - errors[-2])
                if err_dist < conv_tol:
                    break

        if err_dist < conv_tol:
            print("##############################")
            print(
                f"The error between the two last updated states converged\n"
                + f"to an order of {conv_tol} after:\n"
                + f"{n} sweeps at site {sites[i]}\n"
                + f"total iterations {iter}"
            )
            print("##############################")
        else:
            print("##############################")
            print(
                f"The error between the two last updated states converged\n"
                + f"to an order of {err_dist}\n"
                + f"instead of the convergence tolerance {conv_tol}"
            )
            print("##############################")
        return errors, entropy, s_mid

    def TEBD_direct(self, trotter_steps, delta, h_ev, J_ev, fidelity=False, trunc=True):
        """
        direct_mpo_evolution

        This function computes the magnetization and (on demand) the fidelity
        of the trotter evolved MPS by the MPO direct application.

        trotter_steps: int - number of times we apply the mpo to the mps
        delta: float - time interval which defines the evolution per step
        h_ev: float - value of the external field in the evolving hamiltonian
        J_ev: float - value of the Ising interaction in the evolving hamiltonian
        fidelity: bool - we can compute the fidelity with the initial state
                if the chain is small enough. By default False

        """
        overlap = []
        mag_mps_tot = []
        mag_mps_loc = []
        Z = np.array([[1, 0], [0, -1]])

        # total
        self.order_param_Ising(op=Z)
        mag_mps_tot.append(np.real(self.mpo_first_moment()))

        # local
        mag_loc = []
        for i in range(self.L):
            self.single_operator_Ising(site=i + 1, op=Z)
            mag_loc.append(self.mpo_first_moment().real)
        mag_mps_loc.append(mag_loc)

        if fidelity:
            psi_exact_0 = sparse_ising_ground_state(L=self.L, h_t=self.h)
            psi_new_mpo = mps_to_vector(self.sites)
            overlap.append(np.abs((psi_new_mpo.T.conjugate() @ psi_exact_0).real))
        for T in range(trotter_steps):
            print(f"------ Trotter steps: {T} -------")

            self.mpo_Ising_quench_global(delta=delta, h_ev=h_ev, J_ev=1)
            self.mpo_to_mps()
            if trunc:
                self.canonical_form(svd_direction="left")
                self.canonical_form(svd_direction="right")
            # tensor_shapes(self.sites)
            # self.save_sites()

            # total
            self.order_param_Ising(op=Z)
            mag_mps_tot.append(np.real(self.mpo_first_moment()))

            # local
            mag_loc = []
            for i in range(self.L):
                self.single_operator_Ising(site=i + 1, op=Z)
                mag_loc.append(self.mpo_first_moment().real)
            mag_mps_loc.append(mag_loc)

            if fidelity:
                psi_exact = exact_evolution_sparse(
                    L=self.L,
                    psi_init=psi_exact_0,
                    trotter_step=(T + 1),
                    delta=delta,
                    h_t=h_ev,
                )
                psi_new_mpo = mps_to_vector(self.sites)
                overlap.append(np.abs((psi_new_mpo.T.conjugate() @ psi_exact).real))
        return mag_mps_tot, mag_mps_loc, overlap

    def TEBD_variational(
        self,
        trotter_steps: int,
        delta: float,
        h_ev: float,
        quench: str,
        flip: bool,
        n_sweeps: int = 2,
        conv_tol: float = 1e-7,
        fidelity: bool = False,
        bond: bool = True,
        where: int = -1,
    ):
        """
        variational_mps_evolution

        This function computes the magnetization and (on demand) the fidelity
        of the trotter evolved MPS by the MPO direct application.

        trotter_steps: int - number of times we apply the mpo to the mps
        delta: float - time interval which defines the evolution per step
        h_ev: float - value of the external field in the evolving hamiltonian
        flip: bool - flip the initial state middle qubit
        quench: str - type of quench we want to execute. Available are 'flip', 'global'
        fidelity: bool - we can compute the fidelity with the initial state
                if the chain is small enough. By default False
        err: bool - computes the distance error between the guess state and an
                uncompressed state. If True it is used as a convergence criterion.
                By default True

        """
        overlap = []
        mag_mps_tot = []
        mag_mps_loc = []
        mag_mps_loc_X = []
        entropies = []
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])

        if flip:
            if self.L % 2 == 0:
                self.sites[self.L // 2] = np.array([[[0], [1]]])
                self.sites[self.L // 2 - 1] = np.array([[[0], [1]]])
            else:
                self.sites[self.L // 2] = np.array([[[0], [1]]])

        # enlarging our local tensor to the max bond dimension
        self.enlarge_chi()

        # total
        self.order_param(op=Z)
        mag_mps_tot.append(np.real(self.mpo_first_moment()))
        # loc X
        self.local_param(site=self.L // 2 + 1, op=X)
        mag_mps_loc_X.append(np.real(self.mpo_first_moment()))
        # local glob Z
        mag_loc = []
        for i in range(self.L):
            self.local_param(site=i + 1, op=Z)
            mag_loc.append(np.real(self.mpo_first_moment()))
        mag_mps_loc.append(mag_loc)

        # fidelity
        if fidelity:
            psi_exact_0 = sparse_ising_ground_state(L=self.L, h_t=self.h).reshape(
                2**self.L, 1
            )
            psi_new_mpo = mps_to_vector(self.sites)
            overlap.append(np.abs((psi_new_mpo.T.conjugate() @ psi_exact_0).real))

        # initialize ancilla with a state in Right Canonical Form
        self.canonical_form(trunc_chi=True, trunc_tol=False)
        self._compute_norm(site=1)
        self.ancilla_sites = self.sites.copy()

        errors = [[0, 0]]

        for trott in range(trotter_steps):
            print(f"------ Trotter steps: {trott} -------")
            self.mpo_quench(quench, delta, h_ev)
            print(f"Bond dim ancilla: {self.ancilla_sites[self.L//2].shape[0]}")
            print(f"Bond dim site: {self.sites[self.L//2].shape[0]}")
            error, entropy = self.compression(
                trunc_tol=False,
                trunc_chi=True,
                n_sweeps=n_sweeps,
                conv_tol=conv_tol,
                bond=bond,
                where=where,
            )
            self.ancilla_sites = self.sites.copy()
            errors.append(error)
            entropies.append(entropy)

            # total
            self.order_param(op=Z)
            mag_mps_tot.append(np.real(self.mpo_first_moment(mixed=True)))
            # loc X
            self.local_param(site=self.L // 2 + 1, op=X)
            mag_mps_loc_X.append(np.real(self.mpo_first_moment(mixed=True)))
            # local glob Z
            mag = []
            for i in range(self.L):
                self.local_param(site=i + 1, op=Z)
                mag.append(self.mpo_first_moment(mixed=True).real)
            mag_mps_loc.append(mag)

            if fidelity:
                psi_exact = U_evolution_sparse(
                    L=self.L,
                    psi_init=psi_exact_0,
                    trotter_step=(trott + 1),
                    delta=delta,
                    h_t=h_ev,
                )
                psi_new_mpo = mps_to_vector(self.sites)
                overlap.append(np.abs((psi_new_mpo.T.conjugate() @ psi_exact).real))
        return mag_mps_tot, mag_mps_loc_X, mag_mps_loc, overlap, errors, entropies

    def TEBD_variational_Z2(
        self,
        trotter_steps: int,
        delta: float,
        h_ev: float,
        n_sweeps: int = 2,
        conv_tol: float = 1e-7,
        bond: bool = True,
        where: int = -1,
    ):
        """
        variational_mps_evolution

        This function computes the magnetization and (on demand) the fidelity
        of the trotter evolved MPS by the MPO direct application.

        trotter_steps: int - number of times we apply the mpo to the mps
        delta: float - time interval which defines the evolution per step
        h_ev: float - value of the external field in the evolving hamiltonian
        flip: bool - flip the initial state middle qubit
        quench: str - type of quench we want to execute. Available are 'flip', 'global'
        fidelity: bool - we can compute the fidelity with the initial state
                if the chain is small enough. By default False
        err: bool - computes the distance error between the guess state and an
                uncompressed state. If True it is used as a convergence criterion.
                By default True

        """
        entropies = []
        E = []
        M = []
        W = []
        S = []
        sites = [1,2,3,4]
        ladders = [2,3]
        errors = [[0, 0]]
        entropies = [0]

        self.enlarge_chi()


        # electric field
        E_h = np.zeros((2*self.Z2.l+1,2*self.Z2.L-1))
        E_h[:] = np.nan
        E_h = self.electric_field_Z2(E_h)

        # local dual mag
        self.order_param()
        mag = self.mpo_first_moment().real/(len(self.Z2.latt.plaquettes()) - (2 * (self.Z2.L-1) + 2 * (self.Z2.l-2)))
        
        # wilson loop
        self.Z2.wilson_Z2_dual(mpo_sites=sites, ls=ladders) #list(range(s))
        self.w = self.Z2.mpo.copy()
        loop = self.mpo_first_moment().real

        # t'hooft string
        self.Z2.thooft(site=[2], l=[2], direction="horizontal")
        self.w = self.Z2.mpo.copy()
        thooft = self.mpo_first_moment().real

        E.append(E_h)
        M.append(mag)
        # W.append(loop)
        S.append(thooft)

        self.ancilla_sites = self.sites.copy()

        

        for trott in range(trotter_steps):
            print(f"------ Trotter steps: {trott} -------")

            # start with the half mu_x on each ladder
            self.Z2._initialize_finalize_quench_local(delta=delta, h_ev=h_ev)
            self.w = self.Z2.mpo
            self.mpo_to_mps()

            self.Z2.mpo_Z2_quench_tot(delta=delta, h_ev=h_ev)
            self.w = self.Z2.mpo

            print(f"Bond dim ancilla: {self.ancilla_sites[self.L//2].shape[0]}")
            print(f"Bond dim site: {self.sites[self.L//2].shape[0]}")

            # compress the ladder evolution operator
            error, entropy = self.compression(
                trunc_tol=False,
                trunc_chi=True,
                n_sweeps=n_sweeps,
                conv_tol=conv_tol,
                bond=bond,
                where=where,
            )

            # finish with the other half mu_x on each ladder
            self.Z2._initialize_finalize_quench_local(delta=delta, h_ev=h_ev)
            self.w = self.Z2.mpo
            self.mpo_to_mps()

            self.ancilla_sites = self.sites.copy()

            # electric field
            E_h = np.zeros((2*self.Z2.l+1,2*self.Z2.L-1))
            E_h[:] = np.nan
            E_h = self.electric_field_Z2(E_h)

            # local dual mag
            self.order_param()
            mag = self.mpo_first_moment().real/(len(self.Z2.latt.plaquettes()) - (2 * (self.Z2.L-1) + 2 * (self.Z2.l-2)))
            
            # wilson loop
            self.Z2.wilson_Z2_dual(mpo_sites=sites, ls=ladders) #list(range(s))
            self.w = self.Z2.mpo.copy()
            loop = self.mpo_first_moment().real

            # t'hooft string
            self.Z2.thooft(site=[2], l=[2], direction="horizontal")
            self.w = self.Z2.mpo.copy()
            thooft = self.mpo_first_moment().real

            E.append(E_h)
            M.append(mag)
            W.append(loop)
            S.append(thooft)
            errors.append(error)
            entropies.append(entropy)


        return (
            E,
            M,
            W,
            S,
            errors,
            entropies,
        )
    
    def TEBD_variational_Z2_trotter_step(
        self,
        trotter_step: int,
        delta: float,
        h_ev: float,
        n_sweeps: int = 2,
        conv_tol: float = 1e-7,
        bond: bool = True,
        where: int = -1,
    ):
        """
        variational_mps_evolution

        This function computes the magnetization and (on demand) the fidelity
        of the trotter evolved MPS by the MPO direct application.

        trotter_steps: int - number of times we apply the mpo to the mps
        delta: float - time interval which defines the evolution per step
        h_ev: float - value of the external field in the evolving hamiltonian
        flip: bool - flip the initial state middle qubit
        quench: str - type of quench we want to execute. Available are 'flip', 'global'
        fidelity: bool - we can compute the fidelity with the initial state
                if the chain is small enough. By default False
        err: bool - computes the distance error between the guess state and an
                uncompressed state. If True it is used as a convergence criterion.
                By default True

        """

        print(f"------ Trotter steps: {trotter_step} -------")

        # start with the half mu_x on each ladder
        self.Z2._initialize_finalize_quench_local(delta=delta, h_ev=h_ev)
        self.w = self.Z2.mpo
        self.mpo_to_mps()

        self.Z2.mpo_Z2_quench_tot(delta=delta, h_ev=h_ev)
        self.w = self.Z2.mpo

        print(f"Bond dim ancilla: {self.ancilla_sites[self.L//2].shape[0]}")
        print(f"Bond dim site: {self.sites[self.L//2].shape[0]}")

        # compress the ladder evolution operator
        error, entropy, schmidt_values = self.compression(
            trunc_tol=False,
            trunc_chi=True,
            n_sweeps=n_sweeps,
            conv_tol=conv_tol,
            bond=bond,
            where=where,
        )

        # finish with the other half mu_x on each ladder
        self.Z2._initialize_finalize_quench_local(delta=delta, h_ev=h_ev)
        self.w = self.Z2.mpo
        self.mpo_to_mps()

        self.ancilla_sites = self.sites.copy()

        return (
            self,
            error,
            entropy,
            schmidt_values
        )

    # -------------------------------------------------
    # Computing expectation values
    # -------------------------------------------------
    def braket(self, site, ancilla=False, mixed=False, rev=False):
        ket = self.sites
        bra = ket
        env_right = self.env_right
        env_left = self.env_left
        w = self.w

        if ancilla:
            ket = self.ancilla_sites
            bra = ket
            w = self.w
        elif mixed:
            ket = self.ancilla_sites
            bra = self.sites
            w = self.w
        elif rev:
            ket = self.sites
            bra = self.ancilla_sites
            w = self.w
            env_left = self.env_left_sm
            env_right = self.env_right_sm

        # print(f"env_left:{env_left[-1].shape}")
        # print(f"ket:{ket[site - 1].shape}")
        # print(f"w:{w[site - 1].shape}")
        # print(f"bra:{bra[site - 1].shape}")
        # print(f"env_right:{env_right[-1].shape}")
        sandwich = ncon(
            [
                env_left[-1],
                ket[site - 1],
                w[site - 1],
                bra[site - 1].conjugate(),
                env_right[-1],
            ],
            [[1, 4, 7], [1, 3, 2], [4, 5, 3, 6], [7, 6, 8], [2, 5, 8]],
        )
        return sandwich

    def mpo_first_moment(self, site: int=1, ancilla: bool=False, mixed: bool=False):
        self.clear_envs()
        self.envs(site, ancilla=ancilla, mixed=mixed)
        sites = self.sites
        ancilla_sites = sites
        if ancilla:
            sites = self.ancilla_sites
            ancilla_sites = sites
        elif mixed:
            ancilla_sites = self.ancilla_sites

        first_moment = ncon(
            [
                self.env_left[-1],
                ancilla_sites[site - 1],
                self.w[site - 1],
                sites[site - 1].conjugate(),
                self.env_right[-1],
            ],
            [[1, 4, 7], [1, 3, 2], [4, 5, 3, 6], [7, 6, 8], [2, 5, 8]],
        )
        self.clear_envs()
        return first_moment

    def mpo_second_moment(self, opt: bool=False, op: np.ndarray=None, site: int=None, l: int=None, direction: str=None):
        """
        mpo_second_moment

        This function computes the second moment of a given mpo.
        If opt is true it means we are computing the variance during the optimization
        thus we need not to clear the environments and we need the extra attributes ".env_left_sm"
        and ".env_right_sm".

        opt: bool - allows the computation of the sm for the variance. By default False.

        """
        if opt:
            self.envs(sm=True, opt=opt)
            sm = ncon(
                [self.env_left_sm[0], self.env_right_sm[-1]],
                [[1, 2, 3, 4], [1, 2, 3, 4]],
            )
            self.env_left_sm = []
            self.env_right_sm = []
        else:
            self.order_param(op=op, site=site, l=l, direction=direction)
            self.clear_envs()
            self.envs(sm=True)
            sm = ncon(
                [self.env_left[-1], self.sites[0]], [[1, -3, -4, -5], [1, -2, -1]]
            )
            sm = ncon(
                [sm, self.w[0]], [[-1, 1, 2, -4, -5], [2, -2, 1, -3]]
            )
            sm = ncon(
                [sm, self.w[0]], [[-1, -2, 1, 2, -5], [2, -3, 1, -4]]
            )
            sm = ncon(
                [sm, self.sites[0].conjugate()], [[-1, -2, -3, 1, 2], [2, 1, -4]]
            )
            sm = ncon(
                [sm, self.env_right[-1]], [[1,2,3,4], [1,2,3,4]]
            )

        return sm

    def mpo_fourth_moment(self, op: np.ndarray=None, site: int=None, l: int=None, direction: str=None):
        self.order_param(op=op, site=site, l=l, direction=direction)
        self.clear_envs()
        self.envs(fm=True)
        fm = ncon(
                [self.env_left[-1], self.sites[0]], [[1, -3, -4, -5, -6, -7], [1, -2, -1]]
            )
        fm = ncon(
            [fm, self.w[0]], [[-1, 1, 2, -4, -5, -6, -7], [2, -2, 1, -3]]
        )
        fm = ncon(
            [fm, self.w[0]], [[-1, -2, 1, 2, -5, -6, -7], [2, -3, 1, -4]]
        )
        fm = ncon(
            [fm, self.w[0]], [[-1, -2, -3, 1, 2, -6, -7], [2, -4, 1, -5]]
        )
        fm = ncon(
            [fm, self.w[0]], [[-1, -2, -3, -4, 1, 2, -7], [2, -5, 1, -6]]
        )
        fm = ncon(
            [fm, self.sites[0].conjugate()], [[-1, -2, -3, -4, -5, 1, 2], [2, 1, -6]]
        )
        fm = ncon(
            [fm, self.env_right[-1]], [[1,2,3,4,5,6], [1,2,3,4,5,6]]
        )
        return fm

    # -------------------------------------------------
    # More help functions
    # -------------------------------------------------
    def clear_canonical(self):
        self.sites.clear()
        self.bonds.clear()
        return self

    def clear_envs(self):
        self.env_left.clear()
        self.env_right.clear()
        return self

    def save_bond_dimension(self):
        bond_dims = []
        for i in range(len(self.sites)):
            bond_dims.append(self.sites[i].shape[-1])

        return bond_dims

    def save_sites(self, path: str, precision: int=2, cx: list=None, cy: list=None):
        """
        save_sites

        This function saves the sites, e.g., the tensors composing our MPS.
        In order to do that we need to flatten the whole list of tensors and save
        their original shapes in order to reshape them in the loading step.

        precision: int - indicates the precision of the variable h
        """
        if "Ising" == self.model:
            self.save_sites_Ising(path=path, precision=precision)
        elif "Cluster" == self.model:
            self.save_sites_Ising(path=path, precision=precision)
        elif "Cluster-XY" == self.model:
            self.save_sites_Cluster_xy(path=path, precision=precision)
        elif "ANNNI" == self.model:
            self.save_sites_ANNNI(path=path, precision=precision)
        elif "Z2" in self.model:
            self.save_sites_Z2(path=path, precision=precision, cx=cx, cy=cy)
        elif "XXZ" in self.model:
            self.save_sites_XXZ(path=path, precision=precision)
        else:
            raise ValueError("Choose a correct model")
        return self
    
    def load_sites(self, path: str, precision: int=2, cx: list=None, cy: list=None):
        """
        load_sites

        This function load the tensors into the sites of the MPS.
        We fetch a completely flat list, split it to recover the original tensors
        (but still flat) and reshape each of them accordingly with the saved shapes.
        To initially split the list in the correct index position refer to the auxiliary
        function get_labels().

        """
        if "Ising" == self.model:
            self.load_sites_Ising(path=path, precision=precision)
        elif "ANNNI" == self.model:
            self.load_sites_ANNNI(path=path, precision=precision)
        elif "Cluster" == self.model:
            self.load_sites_Ising(path=path, precision=precision)
        elif "Cluster-XY" == self.model:
            self.load_sites_Cluster_xy(path=path, precision=precision)
        elif "Z2" in self.model:
            self.load_sites_Z2(path=path, precision=precision, cx=cx, cy=cy)
        elif "XXZ" in self.model:
            self.load_sites_XXZ(path=path, precision=precision)
        else:
            raise ValueError("Choose a correct model")
        return self

    def save_sites_Ising(self, path, precision: int=2):
        # shapes of the tensors
        shapes = tensor_shapes(self.sites, False)
        np.savetxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_J_{self.J:.{precision}f}",
            shapes,
            fmt="%1.i",  # , delimiter=','
        )

        # flattening of the tensors
        tensor = [element for site in self.sites for element in site.flatten()]
        np.savetxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_J_{self.J:.{precision}f}",
            tensor,
        )
        return self
    
    def save_sites_Cluster_xy(self, path, precision: int=2):
        # shapes of the tensors
        shapes = tensor_shapes(self.sites, False)
        np.savetxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_lx_{self.lx:.{precision}f}_ly_{self.ly:.{precision}f}_J_{self.J:.{precision}f}",
            shapes,
            fmt="%1.i",  # , delimiter=','
        )

        # flattening of the tensors
        tensor = [element for site in self.sites for element in site.flatten()]
        np.savetxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_lx_{self.lx:.{precision}f}_ly_{self.ly:.{precision}f}_J_{self.J:.{precision}f}",
            tensor,
        )
        return self
    
    def save_sites_ANNNI(self, path, precision: int=2):
        # shapes of the tensors
        shapes = tensor_shapes(self.sites, False)
        np.savetxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_k_{self.k:.{precision}f}",
            shapes,
            fmt="%1.i",  # , delimiter=','
        )

        # flattening of the tensors
        tensor = [element for site in self.sites for element in site.flatten()]
        np.savetxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_k_{self.k:.{precision}f}",
            tensor,
        )
        return self
    
    def save_sites_Z2(self, path, precision: int=2, cx: list=None, cy: list=None):
        # shapes of the tensors
        shapes = tensor_shapes(self.sites)
        np.savetxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_direct_lattice_{self.Z2.l}x{self.Z2.L-1}_{cx}-{cy}_chi_{self.chi}_h_{self.h:.{precision}f}",
            shapes,
            fmt="%1.i",  # , delimiter=','
        )

        # flattening of the tensors
        tensor = [element for site in self.sites for element in site.flatten()]
        np.savetxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_direct_lattice_{self.Z2.l}x{self.Z2.L-1}_{cx}-{cy}_chi_{self.chi}_h_{self.h:.{precision}f}",
            tensor,
        )

    def save_sites_XXZ(self, path, precision: int=2):
        # shapes of the tensors
        shapes = tensor_shapes(self.sites, False)
        np.savetxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_d_{self.k:.{precision}f}_h_{self.h:.{precision}f}",
            shapes,
            fmt="%1.i",  # , delimiter=','
        )

        # flattening of the tensors
        tensor = [element for site in self.sites for element in site.flatten()]
        np.savetxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_d_{self.k:.{precision}f}_h_{self.h:.{precision}f}",
            tensor,
        )
        return self

    def load_sites_Ising(self, path, precision: int=2):
        """
        load_sites

        This function load the tensors into the sites of the MPS.
        We fetch a completely flat list, split it to recover the original tensors
        (but still flat) and reshape each of them accordingly with the saved shapes.
        To initially split the list in the correct index position refer to the auxiliary
        function get_labels().

        """
        # loading of the shapes
        shapes = np.loadtxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_J_{self.J:.{precision}f}"
        ).astype(int)
        # loading of the flat tensors
        filedata = np.loadtxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_J_{self.J:.{precision}f}",
            dtype=complex,
        )
        # auxiliary function to get the indices where to split
        labels = get_labels(shapes)
        flat_tn = np.array_split(filedata, labels)
        flat_tn.pop(-1)
        # reshape the flat tensors and initializing the sites
        self.sites = [site.reshape(shapes[i]) for i, site in enumerate(flat_tn)]

        return self

    def load_sites_Cluster_xy(self, path, precision: int=2):
        """
        load_sites

        This function load the tensors into the sites of the MPS.
        We fetch a completely flat list, split it to recover the original tensors
        (but still flat) and reshape each of them accordingly with the saved shapes.
        To initially split the list in the correct index position refer to the auxiliary
        function get_labels().

        """
        # loading of the shapes
        shapes = np.loadtxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_lx_{self.lx:.{precision}f}_ly_{self.ly:.{precision}f}_J_{self.J:.{precision}f}"
        ).astype(int)
        # loading of the flat tensors
        filedata = np.loadtxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_lx_{self.lx:.{precision}f}_ly_{self.ly:.{precision}f}_J_{self.J:.{precision}f}",
            dtype=complex,
        )
        # auxiliary function to get the indices where to split
        labels = get_labels(shapes)
        flat_tn = np.array_split(filedata, labels)
        flat_tn.pop(-1)
        # reshape the flat tensors and initializing the sites
        self.sites = [site.reshape(shapes[i]) for i, site in enumerate(flat_tn)]

        return self
    
    def load_sites_ANNNI(self, path, precision: int=2):
        """
        load_sites

        This function load the tensors into the sites of the MPS.
        We fetch a completely flat list, split it to recover the original tensors
        (but still flat) and reshape each of them accordingly with the saved shapes.
        To initially split the list in the correct index position refer to the auxiliary
        function get_labels().

        """
        # loading of the shapes
        shapes = np.loadtxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_k_{self.k:.{precision}f}"
        ).astype(int)
        # loading of the flat tensors
        filedata = np.loadtxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}_k_{self.k:.{precision}f}",
            dtype=complex,
        )
        # auxiliary function to get the indices where to split
        labels = get_labels(shapes)
        flat_tn = np.array_split(filedata, labels)
        flat_tn.pop(-1)
        # reshape the flat tensors and initializing the sites
        self.sites = [site.reshape(shapes[i]) for i, site in enumerate(flat_tn)]

        return self
    
    def load_sites_Z2(self, path, precision: int=2, cx: list=None, cy: list=None):
        """
        load_sites

        This function load the tensors into the sites of the MPS.
        We fetch a completely flat list, split it to recover the original tensors
        (but still flat) and reshape each of them accordingly with the saved shapes.
        To initially split the list in the correct index position refer to the auxiliary
        function get_labels().

        """
        # loading of the shapes
        shapes = np.loadtxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_direct_lattice_{self.Z2.l}x{self.Z2.L-1}_{cx}-{cy}_chi_{self.chi}_h_{self.h:.{precision}f}",

        ).astype(int)
        # loading of the flat tensors
        filedata = np.loadtxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_direct_lattice_{self.Z2.l}x{self.Z2.L-1}_{cx}-{cy}_chi_{self.chi}_h_{self.h:.{precision}f}",
            dtype=complex,
        )
        # auxiliary function to get the indices where to split
        labels = get_labels(shapes)
        flat_tn = np.array_split(filedata, labels)
        flat_tn.pop(-1)
        # reshape the flat tensors and initializing the sites
        self.sites = [site.reshape(shapes[i]) for i, site in enumerate(flat_tn)]

        return self

    def load_sites_XXZ(self, path, precision: int=2):
        """
        load_sites

        This function load the tensors into the sites of the MPS.
        We fetch a completely flat list, split it to recover the original tensors
        (but still flat) and reshape each of them accordingly with the saved shapes.
        To initially split the list in the correct index position refer to the auxiliary
        function get_labels().

        """
        # loading of the shapes
        shapes = np.loadtxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_d_{self.k:.{precision}f}_h_{self.h:.{precision}f}"
        ).astype(int)
        # loading of the flat tensors
        filedata = np.loadtxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_d_{self.k:.{precision}f}_h_{self.h:.{precision}f}",
            dtype=complex,
        )
        # auxiliary function to get the indices where to split
        labels = get_labels(shapes)
        flat_tn = np.array_split(filedata, labels)
        flat_tn.pop(-1)
        # reshape the flat tensors and initializing the sites
        self.sites = [site.reshape(shapes[i]) for i, site in enumerate(flat_tn)]

        return self
    
    def save_sites_old(self, path, precision=2):
        """
        save_sites

        This function saves the sites, e.g., the tensors composing our MPS.
        In order to do that we need to flatten the whole list of tensors and save
        their original shapes in order to reshape them in the loading step.

        precision: int - indicates the precision of the variable h
        """
        # shapes of the tensors
        shapes = tensor_shapes(self.sites)
        np.savetxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}",
            shapes,
            fmt="%1.i",  # , delimiter=','
        )

        # flattening of the tensors
        tensor = [element for site in self.sites for element in site.flatten()]
        np.savetxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}",
            tensor,
        )

    def load_sites_old(self, path, precision=2):
        """
        load_sites

        This function load the tensors into the sites of the MPS.
        We fetch a completely flat list, split it to recover the original tensors
        (but still flat) and reshape each of them accordingly with the saved shapes.
        To initially split the list in the correct index position refer to the auxiliary
        function get_labels().

        """
        # loading of the shapes
        shapes = np.loadtxt(
            f"{path}/results/tensors/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}"
        ).astype(int)
        # loading of the flat tensors
        filedata = np.loadtxt(
            f"{path}/results/tensors/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}",
            dtype=complex,
        )
        # auxiliary function to get the indices where to split
        labels = get_labels(shapes)
        flat_tn = np.array_split(filedata, labels)
        flat_tn.pop(-1)
        # reshape the flat tensors and initializing the sites
        self.sites = [site.reshape(shapes[i]) for i, site in enumerate(flat_tn)]

        return self
    

def guess_state(L):
    """
    guess_state

    This function gives you a product state tensor of all up spins
    for a chain of length L
    
    """
    up = np.array([[[1],[0]]])
    init_tensor = [up for _ in range(L)]
    return init_tensor

L = 12
d = 2
model = "ANNNI"
chi = 64
h = 1.5


chain = MPS(L=L, d=2, chi=chi, model=model, eps=1, h=1.5, J=1, k=0.1)
init_tensor = guess_state(L)
try:
    print("try with guess state...")
    if h == 1.5:
        init_tensor = guess_state(L)
        chain.sites = init_tensor.copy()
        chain.enlarge_chi()
    else:
        chain.sites = init_tensor.copy()
    timer = time.monotonic()
    energy, entropy, schmidt_vals = chain.DMRG(trunc_tol=False, trunc_chi=True, where=L//2)
    timer = time.monotonic() - timer
except:
    print("try with random state...")
    timer = time.monotonic()
    chain._random_state(seed=3, chi=chi, type_shape="rectangular")
    chain.canonical_form()
    len(chain.sites)
    energy, entropy, schmidt_vals = chain.DMRG(trunc_tol=False, trunc_chi=True, where=L//2)
    timer = time.monotonic() - timer

# L = 6
# l = 5
# d = 2**l
# model = "Z2_dual"
# chi = 64
# h_i = 0
# h_f = 10
# npoints = 20
# hs = np.linspace(h_i,h_f,npoints)
# charges_x = None
# charges_y = None
# num = (h_f - h_i) / npoints
# precision = get_precision(num)
# path_tensor = "/Users/fradm/Desktop/projects/1_Z2"
# path_tensor = "D:/code/projects/1_Z2"
# sites = [1]
# ladders = [1]
# direction = "horizontal"

# S = []
# for h in hs:
#     lattice_mps = MPS(L=L, d=d, model=model, chi=chi, h=h)
#     lattice_mps.L = lattice_mps.L - 1

#     lattice_mps.load_sites(path=path_tensor, precision=precision, cx=charges_x, cy=charges_y)
#     S.append(lattice_mps.mpo_second_moment(site=sites, l=ladders, direction=direction).real/(len(lattice_mps.Z2.latt.plaquettes())**2))
