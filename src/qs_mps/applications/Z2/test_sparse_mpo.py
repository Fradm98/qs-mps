from scipy.sparse import csc_array, identity, linalg
from ncon import ncon
from qs_mps.sparse_hamiltonians_and_operators import sparse_pauli_z, sparse_pauli_x
from qs_mps.utils import mpo_to_matrix, tensor_shapes
from qs_mps.lattice import Lattice
from qs_mps.mps_class import MPS
import numpy as np


class MPO_ladder:
    def __init__(self, L, l, model=str, lamb=None):
        self.L = L
        self.l = l
        self.model = model
        self.charges = np.ones((l + 1, L))
        self.lamb = lamb
        self.mpo = []
        self.latt = self.latt = Lattice((self.L, self.l + 1), (False, False))
        self.dof = self.l * (self.L - 1)

    def charge_constraint(self):
        """
        charge_constraint

        We decide arbitrarily to impose a charge constraint on the bottom right
        corner of the direct lattice.

        """
        # take the product of the first l-1 rows
        productory = np.prod([self.charges[line, :] for line in range(self.l)])
        # take the last row but the last charge
        productory = productory * np.prod([self.charges[-1, :-1]])
        self.charges[-1, -1] = productory
        return self

    def add_charges(self, rows: list, columns: list):
        """
        add_charges

        This function adds the charges to the background
        vacuum sector (all positive charges). The number of
        charges we add are given by the len of each indices list

        rows: list - row indices of the charges to add
        columns: list - column indices of the charges to add
        """
        for i, j in zip(rows, columns):
            assert ((i != -1) and (j != -1)) or (
                (i == self.l - 1) and (j == self.L - 1)
            ), "Do not choose the last charge! We use it for Gauss Law constraint"
            self.charges[j, i] = -1
        # impose the constraint
        self.charge_constraint()
        self.charges = np.flip(self.charges, axis=1)
        return self

    def mpo_skeleton(self):
        """
        mpo_skeleton

        This function initializes the mpo tensor or shape (2+l,2+l,2**l,2**l)
        with O matrices. We add as well the identities in the first and last
        element of the mpo tensor.

        """
        I = identity(2**self.l, dtype=complex)
        O = csc_array((2**self.l, 2**self.l), dtype=complex)
        skeleton = np.array(
            [[O.toarray() for i in range(2 + self.l)] for j in range(2 + self.l)]
        )
        skeleton[0, 0] = I.toarray()
        skeleton[-1, -1] = I.toarray()
        self.mpo = skeleton
        return self

    def charge_coeff_interaction(self, n: int, mpo_site: int):
        """
        charge_coeff_interaction

        This function computes the coefficient for the interaction in the
        ladder Z2 hamiltonian.

        The coefficient

                c_n(j,j+1) = \prod_{k=(n+1)}^{l+1} q_{k,(n+1)}

        is the product of the charges in the direct lattice "below"
        the interacting sites of the dual lattice.

        n: int - goes from 1 to l and indicate in which row of the dual lattice
            is happening the interaction
        mpo_site: int - indicates for which mpo we are computing the coefficient.
            There are (self.L-1) mpos and everyone encodes the interaction term with a
            specific charge coefficient. We take the charges starting from
            the row below the one of the interaction. Accepted values are from
            0 to (self.L - 2)

        """
        assert 1 <= n <= self.l, "Select a value of n within (1,l)"
        assert (
            0 <= mpo_site <= (self.L - 2)
        ), "Select a value of charge column within (0,L-2)"

        if mpo_site == (self.L - 2):
            c_n_j = 1
        else:
            c_n_j = np.prod(self.charges[n:, mpo_site + 1])
        return c_n_j

    def charge_coeff_local_Z(self, n: int, mpo_site: int):
        """
        charge_coeff_local_Z

        This function computes the coefficient for the local Z term in the
        ladder Z2 hamiltonian.

        """
        assert 1 <= n <= self.l, "Select a value of n within (1,l)"

        if mpo_site == 0 or mpo_site == (self.L - 2):
            if mpo_site == 0:
                col = mpo_site
                if n == 1:
                    alpha = 1
                    n = n - 1
                elif n == self.l:
                    alpha = 1
                    n = n
                else:
                    alpha = 0
            elif mpo_site == (self.L - 2):
                col = mpo_site + 1
                if n == 1:
                    alpha = 1
                    n = n - 1
                elif n == self.l:
                    alpha = 1
                    n = n
                else:
                    alpha = 0

            c_n_j = (1 + self.charges[n, col]) ** (alpha) * np.prod(
                self.charges[:n, col]
            )
        else:
            if n == 1:
                c_n_j = np.prod([self.charges[:, col] for col in range(mpo_site + 1)])
            elif n == self.l:
                c_n_j = 1
            else:
                c_n_j = 0

        return c_n_j

    def mpo_Z2_ladder_generalized(self):
        """
        mpo_Z2_ladder_generalized

        This function computes the hamiltonian MPO for the Z2
        gauge pure theory for general number of ladders and charge configuration.

        """
        self.mpo_skeleton()
        mpo_list = []
        for mpo_site in range(self.L - 1):
            # -----------
            # first row
            # -----------
            # interaction terms (from column 2 to l+1)
            for n in range(1, self.l + 1):
                # self.mpo[0,n] = - 1 * sparse_pauli_z(n=n-1, L=self.l).toarray()
                self.mpo[0, n] = (
                    -self.charge_coeff_interaction(n=n, mpo_site=mpo_site)
                    * sparse_pauli_z(n=n - 1, L=self.l).toarray()
                )

            # local terms (column l+2)
            # local Z and local X
            for i in range(1, self.l + 1):
                # self.mpo[0,n+1] += - 3 * sparse_pauli_z(n=i-1, L=self.l).toarray() - self.lamb * sparse_pauli_x(n=i-1, L=self.l).toarray()
                self.mpo[0, n + 1] += (
                    -self.charge_coeff_local_Z(n=i, mpo_site=mpo_site)
                    * sparse_pauli_z(n=i - 1, L=self.l).toarray()
                    - self.lamb * sparse_pauli_x(n=i - 1, L=self.l).toarray()
                )
            # vertical Z interaction
            for j in range(self.l - 1):
                self.mpo[0, n + 1] += -(
                    sparse_pauli_z(n=j, L=self.l) @ sparse_pauli_z(n=j + 1, L=self.l)
                ).toarray()

            # -----------
            # last column
            # -----------
            # positive terms to complete the interaction between nearest neighbor sites (from row 2 to l+1)
            for n in range(1, self.l + 1):
                self.mpo[n, -1] = sparse_pauli_z(n=n - 1, L=self.l).toarray()

            mpo_list.append(self.mpo)
            self.mpo_skeleton()

        self.mpo = mpo_list
        return mpo_list

    def diagonalize(self):
        self.mpo_Z2_ladder_generalized()
        H = mpo_to_matrix(self.mpo)
        e, v = np.linalg.eigh(H)
        return e, v

    def mpo_Z2_quench_global(self, l: int = None, delta: float = None, mps: MPS = None):
        self._initialize_quench_local()
        mps.mpo_to_mps()
        self.mpo_Z2_quench_ladder(l=l, delta=delta)

    def _initialize_quench_local(self):
        I = identity(2**self.l, dtype=complex).toarray()
        w_tot = []
        for _ in range(self.L - 1):
            w_init_X = I
            for l in range(self.l):
                X_l = sparse_pauli_x(n=l, L=self.l).toarray()
                w_init_X_l = linalg.expm(1j * self.lamb * X_l)
                w_init_X = ncon([w_init_X_l, w_init_X], [[-1, 1], [1, -2]])
            w_tot.append(w_init_X)
        self.mpo = w_tot
        return self

    def mpo_Z2_quench_ladder(self, l, delta):
        I = identity(2**self.l, dtype=complex).toarray()
        Z = sparse_pauli_z(n=l - 1, L=self.l).toarray()
        Z_ll = (
            sparse_pauli_z(n=l - 1, L=self.l) @ sparse_pauli_z(n=l, L=self.l)
        ).toarray()
        w_int_loc = np.array(linalg.expm(1j * delta / 4 * Z_ll))

        if l != 1 and l != self.l:
            Z_ll_prime = (
                sparse_pauli_z(n=l, L=self.l) @ sparse_pauli_z(n=l + 1, L=self.l)
            ).toarray()
            w_int_loc_prime = np.array(linalg.expm(1j * delta / 4 * Z_ll_prime))

        w_tot = []
        for mpo_site in range(self.L - 1):
            c_loc = self.charge_coeff_interaction(n=l - 1, mpo_site=mpo_site)
            w_loc = np.array(linalg.expm(1j * c_loc * delta / 2 * Z))

            c_int = self.charge_coeff_interaction(n=l, mpo_site=mpo_site)
            w_even = np.array(
                [
                    [
                        np.sqrt(np.cos(c_int * delta)) * I,
                        1j * np.sqrt(np.sin(c_int * delta)) * Z,
                    ]
                ]
            )
            w_odd = np.array(
                [
                    [
                        np.sqrt(np.cos(c_int * delta)) * I,
                        np.sqrt(np.sin(c_int * delta)) * Z,
                    ]
                ]
            )
            w_odd = np.swapaxes(w_odd, axis1=0, axis2=1)
            if mpo_site == 0:
                if l != 1 and l != self.l:
                    w_in = ncon(
                        [
                            w_loc,
                            w_int_loc,
                            w_int_loc_prime,
                            w_even,
                            w_int_loc_prime,
                            w_int_loc,
                            w_loc,
                        ],
                        [
                            [-3, 5],
                            [5, 3],
                            [3, 1],
                            [-1, -2, 1, 2],
                            [2, 4],
                            [4, 6],
                            [6, -4],
                        ],
                    )
                else:
                    w_in = ncon(
                        [w_loc, w_int_loc, w_even, w_int_loc, w_loc],
                        [[-3, 3], [3, 1], [-1, -2, 1, 2], [2, 4], [4, -4]],
                    )
                w_tot.append(w_in)

            elif mpo_site == self.L - 1:
                if l != 1 and l != self.l:
                    w_fin = ncon(
                        [
                            w_loc,
                            w_int_loc,
                            w_int_loc_prime,
                            w_odd,
                            w_int_loc_prime,
                            w_int_loc,
                            w_loc,
                        ],
                        [
                            [-3, 5],
                            [5, 3],
                            [3, 1],
                            [-1, -2, 1, 2],
                            [2, 4],
                            [4, 6],
                            [6, -4],
                        ],
                    )
                else:
                    w_fin = ncon(
                        [w_loc, w_int_loc, w_odd, w_int_loc, w_loc],
                        [[-3, 3], [3, 1], [-1, -2, 1, 2], [2, 4], [4, -4]],
                    )

            else:
                if l != 1 and l != self.l:
                    w = ncon(
                        [
                            w_loc,
                            w_int_loc,
                            w_int_loc_prime,
                            w_even,
                            w_odd,
                            w_int_loc_prime,
                            w_int_loc,
                            w_loc,
                        ],
                        [
                            [-5, 8],
                            [8, 6],
                            [6, 4],
                            [-1, -3, 4, 3],
                            [-2, -4, 3, 5],
                            [5, 7],
                            [7, 9],
                            [9, -6],
                        ],
                    ).reshape(
                        w_odd.shape[0] * w_even.shape[0],
                        w_odd.shape[1] * w_even.shape[1],
                        w_odd.shape[2],
                        w_even.shape[3],
                    )
                else:
                    w = ncon(
                        [w_loc, w_int_loc, w_even, w_odd, w_int_loc, w_loc],
                        [
                            [-5, 6],
                            [6, 4],
                            [-1, -3, 4, 3],
                            [-2, -4, 3, 5],
                            [5, 7],
                            [7, -6],
                        ],
                    ).reshape(
                        w_odd.shape[0] * w_even.shape[0],
                        w_odd.shape[1] * w_even.shape[1],
                        w_odd.shape[2],
                        w_even.shape[3],
                    )
            w_tot.append(w)

        w_tot.append(w_fin)
        self.mpo = w_tot
        return self


# mpo = MPO_ladder(l=2, L=3, lamb=0)
# rows = []
# columns = []
# mpo.add_charges([1,1],[0,2])
# print(f"charges:\n{mpo.charges}")
# # print(f"degrees of freedom:\n{mpo.dof}")
# print(f"lattice:\n{mpo.latt._lattice_drawer.draw_lattice()}")
# # print(f"shape:\n{tensor_shapes(mpo.mpo)}")
# e, v = mpo.diagonalize()
# print(f"spectrum:\n{e}")
