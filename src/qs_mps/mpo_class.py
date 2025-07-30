from qs_mps.lattice import Lattice
from qs_mps.sparse_hamiltonians_and_operators import (
    sparse_pauli_z,
    sparse_pauli_x,
    sparse_pauli_y,
)
from qs_mps.utils import mpo_to_matrix
from ncon import ncon
from scipy.sparse import csc_array, identity, linalg
import numpy as np
from typing import Optional, Literal


class MPO_ladder:
    def __init__(self, l, L, model=str, lamb=None, bc="obc", cc="h"):
        self.L = L
        self.l = l
        self.model = model
        self.bc = bc
        self.charges = self._define_charges()
        self.lamb = lamb
        self.mpo = []
        self.latt = Lattice((self.L + 1, self.l + 1), (False, False))
        self.dof_dir = None
        self.dof_dual = None
        self.sector = self._define_sector()
        self.cc = cc

    def _define_charges(self):
        if self.bc == "obc":
            charges = np.ones((self.l + 1, self.L + 1))
        elif self.bc == "pbc":
            charges = np.ones((self.l, self.L + 1))
        return charges

    def _define_dof_dir(self):
        if self.bc == "obc":
            self.dof_dir = 2 * self.l * self.L + self.l + self.L
        elif self.bc == "pbc":
            self.dof_dir = 2 * self.l * self.L + self.l
        return self

    def _define_dof_dual(self):
        if self.bc == "obc":
            self.dof_dual = self.l * self.L
        elif self.bc == "pbc":
            self.dof_dual = self.l * self.L + 1
        return self

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
        # self.charge_constraint()
        # self.charges = np.flip(self.charges, axis=1)
        return self

    def _define_sector(self):
        particles = 0
        for charge in self.charges.flatten():
            if charge == -1:
                particles += 1

        if particles == 0:
            sector = "vacuum_sector"
        else:
            sector = f"{particles}_particle(s)_sector"

        self.sector = sector
        return sector

    def mpo_skeleton(self, aux_dim: int = None):
        """
        mpo_skeleton

        This function initializes the mpo tensor or shape (2+l,2+l,2**l,2**l)
        with O matrices. We add as well the identities in the first and last
        element of the mpo tensor.

        aux_dim: int - This auxiliary dimension represents how many rows and
                    columns we want in our MPO. By default None means that it adapts
                    to the system under study. Fixing the auxiliary dimension is
                    useful for known observables which will not need larger MPOs
        """
        I = identity(2**self.l, dtype=complex)
        O = csc_array((2**self.l, 2**self.l), dtype=complex)
        if aux_dim == None:
            skeleton = np.array(
                [[O.toarray() for i in range(2 + self.l)] for j in range(2 + self.l)]
            )
        else:
            skeleton = np.array(
                [[O.toarray() for i in range(aux_dim)] for j in range(aux_dim)]
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

                c_n(j,j+1) = \\prod_{k=(n+1)}^{l+1} q_{k,(n+1)}

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
            0 <= mpo_site <= (self.L - 1)
        ), "Select a value of charge column within (0,L-2)"

        if mpo_site == (self.L - 1):
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

        if mpo_site == 0 or mpo_site == (self.L - 1):
            if mpo_site == 0:
                col = mpo_site
            elif mpo_site == (self.L - 1):
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
                self.charges[n:, col]
            )
        else:
            if n == 1:
                c_n_j = np.prod([self.charges[:, col] for col in range(mpo_site + 1)])
            elif n == self.l:
                c_n_j = 1
            else:
                c_n_j = 0

        return c_n_j

    def charge_coeff_v_edge(self, mpo_site, l):
        """
        charge_coeff_v_edge

        This function gives the charge coefficients of the vertical edges of
        the fields in the direct lattice of a Z2 theory.

        """
        coeff = np.prod(self.charges[(l + 1) :, mpo_site])
        return coeff

    def charge_coeff_v(self, mpo_site, l):
        """
        charge_coeff_v

        This function gives the charge coefficients of the vertical edges of
        the fields in the direct lattice of a Z2 theory.

        """
        if l == 0:
            coeff = np.prod([self.charges[:, col] for col in range(mpo_site + 1)])
        elif l == (self.l - 1):
            coeff = 1
        return coeff

    ##########################################
    # Terms of the Hamiltonian
    ##########################################
    def coeff_vertical(self, file, column, from_zero: bool = True):
        """ """
        if from_zero:
            column = column - 1
            file = file - 1

        if column < self.L:
            coeff = 1
        else:
            prod_charges = np.prod(self.charges, axis=1).tolist()
            coeff = prod_charges[: (file + 1)]
        return coeff

    def mpo_Z2_vertical_edges_obc(
        self, side: str, file: int, column: int, from_zero: bool = True
    ):
        """
        mpo_Z2_vertical_left_edge

        This function takes into account the vertical left edge in obc and pbc
        of the Z2 LGT hamiltonian in the direct formalism but expressed in the dual
        form which is the one used then by the mps. The difficulty of this mapping is
        that we need to use the mpo sites to represent something that lies on the links.

        side: str - edge side. Could be left or right
        file: int - number of ladders, it goes from 1 to l
        column: int - number of plaquettes in a ladder, it goes from 1 to L
        from_zero: bool - decide if you want to start to count the file and column from 1 or 0. By default True
        """
        if side == "left":
            column = 1
            mpo_site = 0
        elif side == "right":
            column = self.L + 1
            mpo_site = self.L - 1

        if from_zero:
            column = column - 1
            file = file - 1

        self.mpo_skeleton(aux_dim=2)
        mpo_list = []
        for site in range(self.L):
            if site == mpo_site:
                self.mpo[0, 1] = (
                    -self.lamb
                    * self.coeff_vertical(file, column, from_zero=False)
                    * sparse_pauli_z(n=file, L=self.l).toarray()
                )
            mpo_list.append(self.mpo)
            self.mpo_skeleton()

        self.mpo = mpo_list
        return mpo_list

    def mpo_Z2_vertical_edges_pbc(
        self, side: str, file: int, column: int, from_zero: bool = True
    ):
        """
        mpo_Z2_vertical_left_edge

        This function takes into account the vertical left edge in obc and pbc
        of the Z2 LGT hamiltonian in the direct formalism but expressed in the dual
        form which is the one used then by the mps. The difficulty of this mapping is
        that we need to use the mpo sites to represent something that lies on the links.

        side: str - edge side. Could be left or right
        file: int - number of ladders, it goes from 1 to l
        column: int - number of plaquettes in a ladder, it goes from 1 to L
        from_zero: bool - decide if you want to start to count the file and column from 1 or 0. By default True
        """
        if side == "left":
            column = 1
            mpo_site = 0
        elif side == "right":
            column = self.L + 1
            mpo_site = self.L - 1

        if from_zero:
            column = column - 1
            file = file - 1

        self.mpo_skeleton(aux_dim=2)
        mpo_list = []
        for site in range(self.L + 1):
            if site == mpo_site:
                self.mpo[0, 1] = (
                    -self.lamb
                    * self.coeff_vertical(file, column, from_zero=False)
                    * sparse_pauli_z(n=file, L=self.l).toarray()
                )
            if (site - 1) == mpo_site:
                l_aux = self.l
                self.l = 1
                self.mpo_skeleton(aux_dim=(l_aux + 2))
                self.l = l_aux
                self.mpo[1, 1] = sparse_pauli_z(n=0, L=1).toarray()
            mpo_list.append(self.mpo)
            self.mpo_skeleton()

        self.mpo = mpo_list
        return mpo_list

    def mpo_Z2_ladder_generalized(self):
        """
        mpo_Z2_ladder_generalized

        This function computes the hamiltonian MPO for the Z2
        gauge pure theory for general number of ladders and charge configuration.

        """
        if self.bc == "obc":
            if self.cc == "v":
                mpo_list = self.mpo_Z2_ladder_generalized_obc_cc_v()
            elif self.cc == "h":
                mpo_list = self.mpo_Z2_ladder_generalized_obc_cc_h()
        elif self.bc == "pbc":
            # mpo_list = self.mpo_Z2_ladder_generalized_pbc()
            mpo_list = self.mpo_Z2_ladder_generalized_topological()
        self.mpo = mpo_list
        return mpo_list

    def mpo_Z2_ladder_generalized_topological(self, topological_sector: Optional[Literal[1,-1]] = 1):
        dof = self.l * self.L + 1

        prod_charges = np.prod(self.charges, axis=1).tolist()

        # initialize
        self.mpo_skeleton()
        mpo_list = []
        for c in range(self.L):
            for f in range(self.l):
                ## Vertical Bulk ----------------------------------------------
                # first row, for the zz horizontal interactions
                self.mpo[0, f + 1] = sparse_pauli_z(n=f, L=self.l).toarray()
                # last column, for the zz horizontal interactions
                self.mpo[f + 1, -1] = (
                    -self.lamb * sparse_pauli_z(n=f, L=self.l).toarray()
                )
                ## Horizontal Bulk ----------------------------------------------
                # first row last column, for the "local" zz vertical interaction
                coeff = np.prod(self.charges[(f + 1) % self.l, : c + 1])
                # print(f"coeff column {c}, file {f} is : {coeff}")
                self.mpo[0, -1] += (
                    -self.lamb
                    * coeff
                    * (
                        sparse_pauli_z(n=f, L=self.l).toarray()
                        @ sparse_pauli_z(n=(f + 1) % self.l, L=self.l).toarray()
                    )
                )
                ## Plaquette ----------------------------------------------
                # first row last column, for the local x plaquette terms
                self.mpo[0, -1] += (
                    -1 / self.lamb * sparse_pauli_x(n=f, L=self.l).toarray()
                )

            ## Vertical Left ----------------------------------------------
            # first row last column, for the local z vertical terms on the left boundary
            if c == 0:
                for f in range(self.l):
                    self.mpo[0, -1] += (
                        -self.lamb * sparse_pauli_z(n=f, L=self.l).toarray()
                    )


            ## Vertical Right ----------------------------------------------
            # last column, treat it as a local term with charge coefficient
            if c == (self.L - 1): 
                for f in range(self.l):
                    coeff = np.prod(prod_charges[: f + 1])
                    self.mpo[0, -1] += (
                            -self.lamb
                            * topological_sector
                            * coeff 
                            * sparse_pauli_z(n=f, L=self.l).toarray() @ sparse_pauli_z(n=(self.l-1), L=self.l).toarray()
                        )

            mpo_list.append(self.mpo)
            self.mpo_skeleton()

        self.mpo = mpo_list
        return mpo_list


    def mpo_Z2_ladder_generalized_pbc(self):
        # # degrees of freedom
        # self.L = self.L - 1
        dof = self.l * self.L + 1

        prod_charges = np.prod(self.charges, axis=1).tolist()

        # initialize
        self.mpo_skeleton()
        mpo_list = []
        for c in range(self.L):
            for f in range(self.l):
                ## Vertical Bulk ----------------------------------------------
                # first row, for the zz horizontal interactions
                self.mpo[0, f + 1] = sparse_pauli_z(n=f, L=self.l).toarray()
                # last column, for the zz horizontal interactions
                self.mpo[f + 1, -1] = (
                    -self.lamb * sparse_pauli_z(n=f, L=self.l).toarray()
                )
                ## Horizontal Bulk ----------------------------------------------
                # first row last column, for the "local" zz vertical interaction
                coeff = np.prod(self.charges[(f + 1) % self.l, : c + 1])
                # print(f"coeff column {c}, file {f} is : {coeff}")
                self.mpo[0, -1] += (
                    -self.lamb
                    * coeff
                    * (
                        sparse_pauli_z(n=f, L=self.l).toarray()
                        @ sparse_pauli_z(n=(f + 1) % self.l, L=self.l).toarray()
                    )
                )
                ## Plaquette ----------------------------------------------
                # first row last column, for the local x plaquette terms
                self.mpo[0, -1] += (
                    -1 / self.lamb * sparse_pauli_x(n=f, L=self.l).toarray()
                )

            ## Vertical Left ----------------------------------------------
            # first row last column, for the local z vertical terms on the left boundary
            if c == 0:
                for f in range(self.l):
                    self.mpo[0, -1] += (
                        -self.lamb * sparse_pauli_z(n=f, L=self.l).toarray()
                    )

            mpo_list.append(self.mpo)
            self.mpo_skeleton()

        ## Vertical Right ----------------------------------------------
        # last column, add the extra degree of freedom which depends on the charges
        l_aux = self.l
        self.l = 1
        self.mpo_skeleton(aux_dim=(l_aux + 2))
        self.l = l_aux
        for f in range(self.l):
            coeff = np.prod(prod_charges[: f + 1])
            self.mpo[f + 1, -1] = (
                -self.lamb * coeff * sparse_pauli_z(n=0, L=1).toarray()
            )
        self.mpo = self.mpo[:, -1].reshape((self.l + 2, 1, 2, 2))
        mpo_list.append(self.mpo)

        self.mpo = mpo_list
        return mpo_list

    def mpo_Z2_ladder_generalized_obc_cc_h(self):
        # degrees of freedom
        dof = self.l * self.L

        # initialize
        self.mpo_skeleton()
        mpo_list = []
        for c in range(self.L):
            ## Horizontal Up ----------------------------------------------
            # first row last column, for the local z horizontal up
            coeff = np.prod(self.charges[0, : c + 1])
            self.mpo[0, -1] += (
                -self.lamb * coeff * sparse_pauli_z(n=0, L=self.l).toarray()
            )

            ## Horizontal Bulk ----------------------------------------------
            for f in range(self.l - 1):
                # first row last column, for the "local" zz vertical interaction
                coeff = np.prod(self.charges[f + 1, : c + 1])
                self.mpo[0, -1] += (
                    -self.lamb
                    * coeff
                    * (
                        sparse_pauli_z(n=f, L=self.l).toarray()
                        @ sparse_pauli_z(n=f + 1, L=self.l).toarray()
                    )
                )

            f += 1
            ## Horizontal Bottom ----------------------------------------------
            # first row last column, for the local z horizontal bottom
            coeff = np.prod(self.charges[f + 1, : c + 1])
            self.mpo[0, -1] += (
                -self.lamb * coeff * sparse_pauli_z(n=f, L=self.l).toarray()
            )

            ## Vertical Left ----------------------------------------------
            # first row last column, for the local z vertical terms on the left boundary
            if c == 0:
                for f in range(self.l):
                    self.mpo[0, -1] += (
                        -self.lamb * sparse_pauli_z(n=f, L=self.l).toarray()
                    )

            for f in range(self.l):
                ## Vertical Bulk ----------------------------------------------
                # first row, for the zz horizontal interactions
                self.mpo[0, f + 1] = sparse_pauli_z(n=f, L=self.l).toarray()
                # last column, for the zz horizontal interactions
                self.mpo[f + 1, -1] = (
                    -self.lamb * sparse_pauli_z(n=f, L=self.l).toarray()
                )

            ## Plaquette ----------------------------------------------
            # first row last column, for the local x plaquette terms
            for f in range(self.l):
                self.mpo[0, -1] += (
                    -1 / self.lamb * sparse_pauli_x(n=f, L=self.l).toarray()
                )

            ## Vertical Right ----------------------------------------------
            # first row last column, for the local z vertical right
            if c == (self.L - 1):
                for f in range(self.l):
                    coeff = np.prod(self.charges[: f + 1, : self.L + 1])
                    self.mpo[0, -1] += (
                        -self.lamb * coeff * sparse_pauli_z(n=f, L=self.l).toarray()
                    )

            mpo_list.append(self.mpo)
            self.mpo_skeleton()

        self.mpo = mpo_list
        return mpo_list

    def mpo_Z2_ladder_generalized_obc_cc_v(self):
        """
        mpo_Z2_ladder_generalized

        This function computes the hamiltonian MPO for the Z2
        gauge pure theory for general number of ladders and charge configuration.

        """
        self.mpo_skeleton()
        mpo_list = []
        for mpo_site in range(self.L):
            # -----------
            # first row
            # -----------
            # interaction terms (from column 2 to l+1)
            for n in range(1, self.l + 1):
                # self.mpo[0,n] = - 1 * sparse_pauli_z(n=n-1, L=self.l).toarray()
                self.mpo[0, n] = (
                    -self.charge_coeff_interaction(n=n, mpo_site=mpo_site)
                    * self.lamb
                    * sparse_pauli_z(n=n - 1, L=self.l).toarray()
                )

            # local terms (column l+2)
            # local Z and local X
            for i in range(1, self.l + 1):
                # self.mpo[0,n+1] += - 3 * sparse_pauli_z(n=i-1, L=self.l).toarray() - self.lamb * sparse_pauli_x(n=i-1, L=self.l).toarray()
                self.mpo[0, n + 1] += (
                    -self.charge_coeff_local_Z(n=i, mpo_site=mpo_site)
                    * self.lamb
                    * sparse_pauli_z(n=i - 1, L=self.l).toarray()
                    - (1 / self.lamb) * sparse_pauli_x(n=i - 1, L=self.l).toarray()
                )
            # vertical Z interaction
            for j in range(self.l - 1):
                self.mpo[0, n + 1] += -(
                    self.lamb
                    * sparse_pauli_z(n=j, L=self.l)
                    @ sparse_pauli_z(n=j + 1, L=self.l)
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

    def mpo_Z2_plaquette_total_energy_density(
        self, site: int, ladder: int, cc: str = "h"
    ):
        """
        mpo_Z2_plaquette_total_energy_density

        This function computes the total energy density of a plaquette at
        an arbitrary site and ladder. In the direct hamiltonian we have two main
        contributions - electring and magnetic. The electric hamiltonian
        contributes with four terms. The latters are expressed in the dual formalism
        with ZZ interaction terms or local Z terms according to which link we are
        computing. Boundary conditions also affect the duality. The magnetic
        hamiltonian contributes with the product of four terms which in the dual form
        is simply a local X term without any charge coefficient.
        We enumerate the links starting from the vertical left (1) and proceed clockwise:

        _ | _ _ 2 _ _ | _
          |           |
          |           |
          1           3
          |           |
          |           |
        _ | _ _ 4 _ _ | _
          |           |

        site: int - column we are interested in computing the energy density
        ladder: int - row we are interested in computing the energy density
        cc: str - charge convention used to compute the MPS

        """
        self.mpo_skeleton(aux_dim=3)
        mpo_list = []
        for c in range(self.L):
            if c == site - 1:
                # prepare the ZZ interaction for the link 1
                self.mpo[0, 1] = sparse_pauli_z(n=ladder, L=self.l).toarray()

            if c == site:
                # prepare the ZZ interaction for the link 3
                self.mpo[0, 1] = sparse_pauli_z(n=ladder, L=self.l).toarray()
                # finish the ZZ interaction for the link 1
                if cc == "h":
                    coeff = 1
                elif cc == "v":
                    coeff = self.charge_coeff_interaction(n=ladder + 1, mpo_site=c)
                self.mpo[1, -1] = (
                    -(1 / 2)
                    * self.lamb
                    * coeff
                    * sparse_pauli_z(n=ladder, L=self.l).toarray()
                )  # link 1 is shared between two plaquettes

                if ladder == 0:
                    # local Z interaction for boundary link 2
                    if cc == "h":
                        coeff = np.prod(self.charges[ladder, : c + 1])
                    elif cc == "v":
                        coeff = self.charge_coeff_v(mpo_site=c, l=ladder)
                    if self.bc == "obc":
                        # local Z interaction for boundary link 2
                        self.mpo[0, -1] += (
                            -self.lamb
                            * coeff
                            * sparse_pauli_z(n=ladder, L=self.l).toarray()
                        )  # link 2 boundary is just in one plaquette
                    elif self.bc == "pbc":
                        # "local" ZZ interaction for bulk link 2
                        self.mpo[0, -1] += (
                            -self.lamb
                            * coeff
                            * sparse_pauli_z(
                                n=(ladder - 1) % self.l, L=self.l
                            ).toarray()
                            @ sparse_pauli_z(n=ladder, L=self.l).toarray()
                        )  # link 2 boundary is shared between the first and last plaquette for pbc
                    # "local" ZZ interaction for bulk link 4
                    coeff = np.prod(self.charges[ladder + 1, : c + 1])
                    self.mpo[0, -1] += (
                        -(1 / 2)
                        * self.lamb
                        * coeff
                        * sparse_pauli_z(n=ladder, L=self.l).toarray()
                        @ sparse_pauli_z(n=ladder + 1, L=self.l).toarray()
                    )  # link 4 is shared between two plaquettes

                elif ladder in range(1, self.l - 1):
                    # "local" ZZ interaction for bulk link 2
                    coeff = np.prod(self.charges[ladder, : c + 1])
                    self.mpo[0, -1] += (
                        -(1 / 2)
                        * self.lamb
                        * coeff
                        * sparse_pauli_z(n=ladder - 1, L=self.l).toarray()
                        @ sparse_pauli_z(n=ladder, L=self.l).toarray()
                    )  # link 2 is shared between two plaquettes
                    # "local" ZZ interaction for bulk link 4
                    coeff = np.prod(self.charges[ladder + 1, : c + 1])
                    self.mpo[0, -1] += (
                        -(1 / 2)
                        * self.lamb
                        * coeff
                        * sparse_pauli_z(n=ladder, L=self.l).toarray()
                        @ sparse_pauli_z(n=ladder + 1, L=self.l).toarray()
                    )  # link 4 is shared between two plaquettes

                elif ladder == self.l - 1:
                    # "local" ZZ interaction for bulk link 2
                    coeff = np.prod(self.charges[ladder, : c + 1])
                    self.mpo[0, -1] += (
                        -(1 / 2)
                        * self.lamb
                        * coeff
                        * sparse_pauli_z(n=ladder - 1, L=self.l).toarray()
                        @ sparse_pauli_z(n=ladder, L=self.l).toarray()
                    )  # link 2 is shared between two plaquettes
                    # local Z interaction for boundary link 4
                    if cc == "h":
                        coeff = np.prod(self.charges[ladder + 1, : c + 1])
                    elif cc == "v":
                        coeff = self.charge_coeff_v(mpo_site=c, l=ladder)
                    if self.bc == "obc":
                        # local Z interaction for boundary link 4
                        self.mpo[0, -1] += (
                            -self.lamb
                            * coeff
                            * sparse_pauli_z(n=ladder, L=self.l).toarray()
                        )  # link 4 boundary is just in one plaquette
                    elif self.bc == "pbc":
                        # "local" ZZ interaction for bulk link 4
                        self.mpo[0, -1] += (
                            -self.lamb
                            * coeff
                            * sparse_pauli_z(n=ladder, L=self.l).toarray()
                            @ sparse_pauli_z(
                                n=(ladder + 1) % self.l, L=self.l
                            ).toarray()
                        )  # link 4 bulk is shared between the first and last plaquette for pbc

                ### magnetic term ###
                self.mpo[0, -1] += (
                    -1 / self.lamb * sparse_pauli_x(n=ladder, L=self.l).toarray()
                )

            if c == site + 1:
                # finish the ZZ interaction for the link 3
                if cc == "h":
                    coeff = 1
                elif cc == "v":
                    coeff = self.charge_coeff_interaction(n=ladder + 1, mpo_site=c)
                self.mpo[1, -1] = (
                    -(1 / 2)
                    * self.lamb
                    * coeff
                    * sparse_pauli_z(n=ladder, L=self.l).toarray()
                )  # link 3 is shared between two plaquettes

            mpo_list.append(self.mpo)
            self.mpo_skeleton(aux_dim=3)
        self.mpo = mpo_list
        return mpo_list

    def mpo_Z2_plaquette_electric_energy_density(
        self, site: int, ladder: int, cc: str = "h"
    ):
        """
        mpo_Z2_plaquette_electric_energy_density

        This function computes the electric energy density of a plaquette at
        an arbitrary site and ladder. In the direct hamiltonian we have four terms
        to take into account. The latters are expressed in the dual formalism
        with ZZ interaction terms or local Z terms according to which link we are
        computing. Boundary conditions also affect the duality. We enumerate the links
        starting from the vertical left (1) and proceed clockwise:

        _ | _ _ 2 _ _ | _
          |           |
          |           |
          1           3
          |           |
          |           |
        _ | _ _ 4 _ _ | _
          |           |

        site: int - column we are interested in computing the energy density
        ladder: int - row we are interested in computing the energy density
        cc: str - charge convention used to compute the MPS

        """
        self.mpo_skeleton(aux_dim=3)
        mpo_list = []
        for c in range(self.L):
            # if c == site - 1:
            #     # prepare the ZZ interaction for the link 1
            #     self.mpo[0, 1] = sparse_pauli_z(n=ladder, L=self.l).toarray()

            if c == site:
                # # prepare the ZZ interaction for the link 3
                # self.mpo[0, 1] = sparse_pauli_z(n=ladder, L=self.l).toarray()
                # # finish the ZZ interaction for the link 1
                # if cc == "h":
                #     coeff = 1
                # elif cc == "v":
                #     coeff = self.charge_coeff_interaction(n=ladder + 1, mpo_site=c)
                # self.mpo[1, -1] = (
                #     -(1 / 2)
                #     * self.lamb
                #     * coeff
                #     * sparse_pauli_z(n=ladder, L=self.l).toarray()
                # )  # link 1 is shared between two plaquettes

                if ladder == 0:
                    if cc == "h":
                        coeff = np.prod(self.charges[ladder, : c + 1])
                    elif cc == "v":
                        coeff = self.charge_coeff_v(mpo_site=c, l=ladder)
                    if self.bc == "obc":
                        # local Z interaction for boundary link 2
                        self.mpo[0, -1] += (
                            coeff
                            * sparse_pauli_z(n=ladder, L=self.l).toarray()
                        )  # link 2 boundary is just in one plaquette
                    elif self.bc == "pbc":
                        # "local" ZZ interaction for bulk link 2
                        self.mpo[0, -1] += (
                            -(1 / 2)
                            * coeff
                            * sparse_pauli_z(
                                n=(ladder - 1) % self.l, L=self.l
                            ).toarray()
                            @ sparse_pauli_z(n=ladder, L=self.l).toarray()
                        )  # link 2 boundary is shared between the first and last plaquette for pbc
                    # "local" ZZ interaction for bulk link 4
                    coeff = np.prod(self.charges[ladder + 1, : c + 1])
                    self.mpo[0, -1] += (
                        -(1 / 2)
                        * coeff
                        * sparse_pauli_z(n=ladder, L=self.l).toarray()
                        @ sparse_pauli_z(n=ladder + 1, L=self.l).toarray()
                    )  # link 4 is shared between two plaquettes

                elif ladder in range(1, self.l - 1):
                    # "local" ZZ interaction for bulk link 2
                    coeff = np.prod(self.charges[ladder, : c + 1])
                    self.mpo[0, -1] += (
                        -(1 / 2)
                        * coeff
                        * sparse_pauli_z(n=ladder - 1, L=self.l).toarray()
                        @ sparse_pauli_z(n=ladder, L=self.l).toarray()
                    )  # link 2 is shared between two plaquettes
                    # "local" ZZ interaction for bulk link 4
                    coeff = np.prod(self.charges[ladder + 1, : c + 1])
                    self.mpo[0, -1] += (
                        -(1 / 2)
                        * coeff
                        * sparse_pauli_z(n=ladder, L=self.l).toarray()
                        @ sparse_pauli_z(n=ladder + 1, L=self.l).toarray()
                    )  # link 4 is shared between two plaquettes

                elif ladder == self.l - 1:
                    # "local" ZZ interaction for bulk link 2
                    coeff = np.prod(self.charges[ladder, : c + 1])
                    self.mpo[0, -1] += (
                        -(1 / 2)
                        * coeff
                        * sparse_pauli_z(n=ladder - 1, L=self.l).toarray()
                        @ sparse_pauli_z(n=ladder, L=self.l).toarray()
                    )  # link 2 is shared between two plaquettes
                    if self.bc == "obc":
                        if cc == "h":
                            coeff = np.prod(self.charges[ladder + 1, : c + 1])
                        elif cc == "v":
                            coeff = self.charge_coeff_v(mpo_site=c, l=ladder)
                        # local Z interaction for boundary link 4
                        self.mpo[0, -1] += (
                            -self.lamb
                            * coeff
                            * sparse_pauli_z(n=ladder, L=self.l).toarray()
                        )  # link 4 boundary is just in one plaquette
                    elif self.bc == "pbc":
                        if cc == "h":
                            coeff = np.prod(
                                self.charges[(ladder + 1) % self.l, : c + 1]
                            )
                        elif cc == "v":
                            coeff = self.charge_coeff_v(mpo_site=c, l=ladder)
                        # "local" ZZ interaction for bulk link 4
                        self.mpo[0, -1] += (
                            -(1 / 2)
                            * coeff
                            * sparse_pauli_z(n=ladder, L=self.l).toarray()
                            @ sparse_pauli_z(
                                n=(ladder + 1) % self.l, L=self.l
                            ).toarray()
                        )  # link 4 bulk is shared between the first and last plaquette for pbc

            # if c == site + 1:
            #     # finish the ZZ interaction for the link 3
            #     if cc == "h":
            #         coeff = 1
            #     elif cc == "v":
            #         coeff = self.charge_coeff_interaction(n=ladder + 1, mpo_site=c)
            #     self.mpo[1, -1] = (
            #         -(1 / 2)
            #         * self.lamb
            #         * coeff
            #         * sparse_pauli_z(n=ladder, L=self.l).toarray()
            #     )  # link 3 is shared between two plaquettes

            mpo_list.append(self.mpo)
            self.mpo_skeleton(aux_dim=3)
        self.mpo = mpo_list
        return mpo_list

    def mpo_Z2_plaquette_magnetic_energy_density(self, site: int, ladder: int):
        """
        mpo_Z2_plaquette_magnetic_energy_density

        This function computes the magnetic energy density of a plaquette at
        an arbitrary site and ladder. In the direct hamiltonian we have four terms
        to take into account. The product of this four originates the magnetic term
        which in the dual form is simply a local X term without any charge coefficient.
        We enumerate the links starting from the vertical left (1) and
        proceed clockwise:

        _ | _ _ 2 _ _ | _
          |           |
          |           |
          1           3
          |           |
          |           |
        _ | _ _ 4 _ _ | _
          |           |

        site: int - column we are interested in computing the energy density
        ladder: int - row we are interested in computing the energy density

        """
        self.mpo_skeleton(aux_dim=2)
        mpo_list = []
        for c in range(self.L):
            if c == site:
                self.mpo[0, -1] = (
                    -1 / self.lamb * sparse_pauli_x(n=ladder, L=self.l).toarray()
                )
            mpo_list.append(self.mpo)
            self.mpo_skeleton(aux_dim=2)
        self.mpo = mpo_list
        return mpo_list

    def diagonalize(self):
        self.mpo_Z2_ladder_generalized()
        H = mpo_to_matrix(self.mpo)
        e, v = np.linalg.eigh(H)
        return e, v

    def _initialize_finalize_quench_local(self, delta, h_ev):
        I = identity(2**self.l, dtype=complex).toarray()
        w_tot = []
        for _ in range(self.L):
            w_init_X = np.array([[I]])
            for l in range(self.l):
                X_l = sparse_pauli_x(n=l, L=self.l).toarray()
                w_init_X_l = np.array([[linalg.expm(1j * (1 / h_ev) * (delta / 2) * X_l)]])
                w_init_X = ncon(
                    [w_init_X_l, w_init_X], [[-1, -3, -5, 1], [-2, -4, 1, -6]]
                ).reshape((1, 1, w_init_X.shape[2], w_init_X_l.shape[3]))
            w_tot.append(w_init_X)
            
        w_tot.append(np.array([[np.eye(2)]]))
        self.mpo = w_tot
        return self


    def mpo_Z2_ladder_quench_int(self, delta: float, h_ev: float, l: int, cc:str="h"):
        """
        
        """  
        # if self.bc == "obc":
        #     w_tot = []
        #     for mpo_site in range(self.L):
        #         self.mpo_skeleton(aux_dim=2)
        #         # horizontal links up/bulk/bottom - local terms
        #         if l == 0:
        #             # up
        #             coeff = np.prod(self.charges[l, : mpo_site + 1])
        #             Z_l = coeff * sparse_pauli_z(n=l, L=self.l).toarray()
        #             w_loc = np.array(linalg.expm(1j * h_ev * delta * Z_l))
        #             w_l = ncon([w_loc, w_l], [[-3, 1], [-1, -2, 1, -4]])
        #         else:
        #             # bulk
        #             coeff = np.prod(self.charges[l, : mpo_site + 1])
        #             Z_ll = coeff * (
        #                 sparse_pauli_z(n=l - 1, L=self.l) @ sparse_pauli_z(n=l, L=self.l)
        #             ).toarray()
        #             w_int_loc = np.array(linalg.expm(1j * h_ev * delta * Z_ll))
        #             w_l = ncon([w_int_loc, w_l], [[-3, 1], [-1, -2, 1, -4]])
        #         if l == self.l-1:
        #             # bottom
        #             coeff = np.prod(self.charges[l + 1, : mpo_site + 1])
        #             Z_l = coeff * sparse_pauli_z(n=l, L=self.l).toarray()
        #             w_loc = np.array(linalg.expm(1j * h_ev * delta * Z_l))
        #             w_l = ncon([w_loc, w_l], [[-3, 1], [-1, -2, 1, -4]])
        #             for lad in range(self.l):
        #                 X_l = sparse_pauli_x(n=lad, L=self.l).toarray()
        #                 w_init_X_l = np.array([[linalg.expm(1j * (1 / h_ev) * (delta / 2) * X_l)]])
        #                 w_l = ncon(
        #                     [w_init_X_l, w_l], [[-1, -3, -5, 1], [-2, -4, 1, -6]]
        #                 ).reshape((w_l.shape[0], w_l.shape[1], w_init_X_l.shape[2], w_l.shape[3]))
                
        #         w_tot.append(w_l)
        
        # elif self.bc == "pbc":


        Z2 = sparse_pauli_z(n=0, L=1).toarray()
        I2 = identity(2, dtype=complex).toarray()
        I = identity(2**self.l, dtype=complex).toarray()

        w_odd_re = np.array(
                [
                    [ 
                        np.sqrt(np.cos(h_ev * delta)) * I2,
                    ],
                    [
                        np.sqrt(np.sin(h_ev * delta)) * Z2,
                    ]
                ]
            )
        
        w_tot = []
        for mpo_site in range(self.L):
            # print(f"site: {mpo_site}, ladder: {l}")

            # left, bulk, and right vertical links
            Z = sparse_pauli_z(n=l, L=self.l).toarray()
            w_even = np.array(
                    [
                        [
                            np.sqrt(np.cos(h_ev * delta)) * I,
                            1j
                            * np.sqrt(np.sin(h_ev * delta))
                            * Z,
                        ]
                    ]
                )
            w_odd = np.array(
                    [
                        [
                            np.sqrt(np.cos(h_ev * delta)) * I,
                            np.sqrt(np.sin(h_ev * delta)) * Z,
                        ]
                    ]
                )
            w_odd = np.swapaxes(w_odd, axis1=0, axis2=1)

            # bulk horizontal links
            coeff = np.prod(self.charges[(l + 1) % self.l, : mpo_site + 1])
            Z_ll = coeff * (
                sparse_pauli_z(n=l, L=self.l) @ sparse_pauli_z(n=(l + 1) % self.l, L=self.l)
            ).toarray()
            w_int_loc = linalg.expm(1j * h_ev * delta * Z_ll)

            if mpo_site == 0:
                # left vertical links + start interaction for bulk vertical links
                w_loc_le = linalg.expm(1j * h_ev * delta * Z)
                w_l = ncon(
                    [w_loc_le, w_even],
                    [[1, -4], [-1, -2, -3, 1]],
                )

            elif mpo_site == self.L-1:
                # finish interaction for bulk vertical links + start interaction for right vertical links
                w_even_re = np.array(
                                [
                                    [
                                        np.sqrt(np.cos(h_ev * delta)) * I,
                                        1j
                                        * np.sqrt(np.sin(h_ev * delta))
                                        * Z,
                                    ],
                                ]
                            )
                w_l = ncon([w_even_re, w_odd],
                        [[-1, -3, -5, 1],[-2, -4, 1, -6]],
                ).reshape(
                            (
                                w_even_re.shape[0] * w_odd.shape[0],
                                w_even_re.shape[1] * w_odd.shape[1],
                                w_even_re.shape[2],
                                w_odd.shape[-1],
                            )
                        )
            else:
                # interactions of bulk vertical links
                w_l = ncon(
                        [w_even, w_odd],
                        [[-1, -3, -5, 1], [-2, -4, 1, -6]],
                    ).reshape(
                            (
                                w_even.shape[0] * w_odd.shape[0],
                                w_even.shape[1] * w_odd.shape[1],
                                w_even.shape[2],
                                w_odd.shape[-1],
                            )
                        )

            # horizontal bulk links
            w_l = ncon([w_l, w_int_loc],[[-1,-2,1,-4],[-3,1]])
            w_tot.append(w_l)

        # finish interaction for right vertical links
        w_tot.append(w_odd_re)
        self.mpo = w_tot
        return w_tot
    
    # def mpo_Z2_quench_int(self, delta, h_ev, cc: str="h"):
    #     I = identity(2**self.l, dtype=complex).toarray()
    #     prod_charges = np.prod(self.charges, axis=1).tolist()

    #     w_tot = []
    #     for mpo_site in range(self.L):
    #         w_l = np.array([[I]])
    #         for l in range(self.l):
    #             Z = sparse_pauli_z(n=l, L=self.l).toarray()
    #             if cc == "v":
    #                 c_loc = self.charge_coeff_local_Z(n=l+1, mpo_site=mpo_site)
    #                 c_int = self.charge_coeff_interaction(n=l+1, mpo_site=mpo_site)
    #             elif cc == "h":
    #                 c_loc = 1
    #                 c_int = 1
    #             w_loc = np.array(linalg.expm(1j * c_loc * h_ev * delta * Z))

    #             w_even = np.array(
    #                 [
    #                     [
    #                         np.sqrt(np.cos(c_int * h_ev * delta)) * I,
    #                         1j
    #                         * np.sign(c_int)
    #                         * np.sqrt(np.sin(np.abs(c_int) * h_ev * delta))
    #                         * Z,
    #                     ]
    #                 ]
    #             )
    #             w_odd = np.array(
    #                 [
    #                     [
    #                         np.sqrt(np.cos(c_int * h_ev * delta)) * I,
    #                         np.sqrt(np.sin(np.abs(c_int) * h_ev * delta)) * Z,
    #                     ]
    #                 ]
    #             )
    #             w_odd = np.swapaxes(w_odd, axis1=0, axis2=1)
    #             if mpo_site == 0:
    #                 # vertical links left edge - local and even terms
    #                 w_l = ncon(
    #                     [w_loc, w_even, w_l],
    #                     [[-5, 2], [-1, -3, 2, 1], [-2, -4, 1, -6]],
    #                 ).reshape(
    #                     (
    #                         w_l.shape[0] * w_even.shape[0],
    #                         w_l.shape[1] * w_even.shape[1],
    #                         w_loc.shape[0],
    #                         w_l.shape[-1],
    #                     )
    #                 )
    #             elif mpo_site == self.L - 1:
    #                 # vertical links right edge - even terms
    #                 if self.bc == "obc":
    #                     w_l = ncon(
    #                         [w_loc, w_odd, w_l],
    #                         [[-5, 2], [-1, -3, 2, 1], [-2, -4, 1, -6]],
    #                     ).reshape(
    #                         (
    #                             w_l.shape[0] * w_odd.shape[0],
    #                             w_l.shape[1] * w_odd.shape[1],
    #                             w_loc.shape[0],
    #                             w_l.shape[-1],
    #                         )
    #                     )
    #                 elif self.bc == "pbc":
    #                     coeff = np.prod(prod_charges[: l + 1])
    #                     w_l = coeff * ncon(
    #                         [w_even, w_odd, w_l],
    #                         [[-1, -4, 1, -8], [-2, -5, 2, 1], [-3, -6, -7, 2]],
    #                     ).reshape(
    #                         (
    #                             w_l.shape[0] * w_odd.shape[0] * w_even.shape[0],
    #                             w_l.shape[1] * w_odd.shape[1] * w_even.shape[1],
    #                             w_l.shape[2],
    #                             w_even.shape[-1],
    #                         )
    #                     )
    #             else:
    #                 # vertical links bulk - even/odd terms
    #                 w_l = ncon(
    #                     [w_even, w_odd, w_l],
    #                     [[-1, -4, 1, -8], [-2, -5, 2, 1], [-3, -6, -7, 2]],
    #                 ).reshape(
    #                     (
    #                         w_l.shape[0] * w_odd.shape[0] * w_even.shape[0],
    #                         w_l.shape[1] * w_odd.shape[1] * w_even.shape[1],
    #                         w_l.shape[2],
    #                         w_even.shape[-1],
    #                     )
    #                 )

    #         w_l = np.array([[I]])
    #         if self.bc == "obc":
    #             # horizontal links up/bulk/bottom - local terms
    #             # up
    #             coeff = np.prod(self.charges[0, : mpo_site + 1])
    #             Z_l = coeff * sparse_pauli_z(n=0, L=self.l).toarray()
    #             w_loc = np.array(linalg.expm(1j * h_ev * delta * Z_l))
    #             w_l = ncon([w_loc, w_l], [[-3, 1], [-1, -2, 1, -4]])
    #             # w_tot.append(w_l)
    #             # bulk
    #             for l in range(self.l-1):
    #                 coeff = np.prod(self.charges[l + 1, : mpo_site + 1])
    #                 Z_ll = coeff * (
    #                     sparse_pauli_z(n=l, L=self.l) @ sparse_pauli_z(n=l + 1, L=self.l)
    #                 ).toarray()
    #                 w_int_loc = np.array(linalg.expm(1j * h_ev * delta * Z_ll))
    #                 w_l = ncon([w_int_loc, w_l], [[-3, 1], [-1, -2, 1, -4]])
    #                 # w_tot.append(w_l)
                
    #             l += 1
    #             # bottom
    #             coeff = np.prod(self.charges[l + 1, : mpo_site + 1])
    #             Z_l = coeff * sparse_pauli_z(n=l, L=self.l).toarray()
    #             w_loc = np.array(linalg.expm(1j * h_ev * delta * Z_l))
    #             w_l = ncon([w_loc, w_l], [[-3, 1], [-1, -2, 1, -4]])
                
    #             w_tot.append(w_l)
            
    #         elif self.bc == "pbc":
    #             # horizontal links bulk - local terms
    #             for l in range(self.l):
    #                 coeff = np.prod(self.charges[(l + 1) % self.l, : mpo_site + 1])
    #                 Z_ll = coeff * (
    #                     sparse_pauli_z(n=l, L=self.l) @ sparse_pauli_z(n=(l + 1) % self.l, L=self.l)
    #                 ).toarray()
    #                 w_int_loc = np.array(linalg.expm(1j * h_ev * delta * Z_ll))
    #                 w_l = ncon([w_int_loc, w_l], [[-3, 1], [-1, -2, 1, -4]])
    #                 # w_tot.append(w_l)

    #             w_tot.append(w_l)

    #     if self.bc == "pbc":
    #         # vertical links right edge - odd term (ancillary site)
    #         Z = sparse_pauli_z(n=0, L=1).toarray()
    #         I = identity(2, dtype=complex).toarray()
    #         w_odd = np.array(
    #                 [
    #                     [
    #                         np.sqrt(np.cos(c_int * h_ev * delta)) * I,
    #                         np.sqrt(np.sin(np.abs(c_int) * h_ev * delta)) * Z,
    #                     ]
    #                 ]
    #             )
    #         w_tot.append(w_odd)

    #     self.mpo = w_tot
    #     return self

    # def thooft(self, site: list, l: list, direction: str):
    #     """
    #     thooft

    #     This function finds the 't Hooft string for the Z2 dual lattice.
    #     It gives us vertical and horizontal strings from a specific dual lattice
    #     site and going, conventionally, up and left, respectively (to vertical and horizontal).

    #     site: list - the first element is the mps site of the interested dual lattice site, starts from 0
    #     l: list - the first element is the ladder of the interested dual lattice site, starts from 0
    #     direction: str - indicates in the direction of the string. We use the convention:
    #             hor -> from left to 'site' at a specific 'l'
    #             ver -> from up to 'l' at a specific 'site'

    #     """
    #     self.mpo_skeleton(aux_dim=2)

    #     site = site[0]
    #     l = l[0]
    #     mpo_tot = []
    #     if direction == "vertical": # depends on the bounday conditions
    #         coeff = self.charge_coeff_v(mpo_site=site, l=0)
    #     if direction == "horizontal": # no charges
    #         coeff = self.charge_coeff_v_edge(mpo_site=site, l=l)
    #         for mpo_site in range(site):
    #             coeff = coeff * self.charge_coeff_interaction(n=l+1, mpo_site=mpo_site)

    #     for mpo_site in range(self.L-1):
    #         if mpo_site == site:
    #             self.mpo[0,-1] = coeff * sparse_pauli_z(n=l, L=self.l).toarray()
    #         mpo_tot.append(self.mpo)
    #         self.mpo_skeleton(aux_dim=2)

    #     self.mpo = mpo_tot
    #     return self
    def thooft(self, site: list, l: list, direction: str):
        """
        thooft

        This function finds the 't Hooft string for the Z2 dual lattice.
        It gives us vertical and horizontal strings from a specific dual lattice
        site and going, conventionally, up and left, respectively (to vertical and horizontal).

        site: list - the first element is the mps site of the interested dual lattice site, starts from 0
        l: list - the first element is the ladder of the interested dual lattice site, starts from 0
        direction: str - indicates in the direction of the string. We use the convention:
                hor -> from left to 'site' at a specific 'l'
                ver -> from up to 'l' at a specific 'site'

        """
        self.mpo_skeleton(aux_dim=2)

        site = site[0]
        l = l[0]
        mpo_tot = []
        if direction == "vertical":  # depends on the bounday conditions
            coeff = self.charge_coeff_v(mpo_site=site, l=0)
        if direction == "horizontal":  # no charges
            coeff = 1

        for mpo_site in range(self.L):
            if mpo_site == site:
                self.mpo[0, -1] = coeff * sparse_pauli_z(n=l, L=self.l).toarray()
            mpo_tot.append(self.mpo)
            self.mpo_skeleton(aux_dim=2)

        self.mpo = mpo_tot
        return self

    def correlator(self, site: list, ladders: list):
        """
        thooft

        This function finds the 't Hooft string for the Z2 dual lattice.
        It gives us vertical and horizontal strings from a specific dual lattice
        site and going, conventionally, up and left, respectively (to vertical and horizontal).

        site: list - the first element is the mps site of the interested dual lattice site
        l: list - the first element is the ladder of the interested dual lattice site
        direction: str - indicates in the direction of the string. We use the convention:
                hor -> from left to 'site' at a specific 'l'
                ver -> from up to 'l' at a specific 'site'

        """
        self.mpo_skeleton(aux_dim=2)

        site = site[0]
        mpo_tot = []

        for mpo_site in range(self.L - 1):
            if mpo_site == site:
                self.mpo[0, -1] = identity(2**self.l, dtype=complex).toarray()
                for l in ladders:
                    if l == 0 or l == self.l:
                        if l == 0:
                            l = l
                        elif l == self.l:
                            l = l - 1
                        coeff = self.charge_coeff_v(mpo_site=site, l=l)
                        self.mpo[0, -1] = self.mpo[0, -1] @ (
                            coeff * sparse_pauli_z(n=l, L=self.l).toarray()
                        )
                        # self.mpo[0,-1] = self.mpo[0,-1] @ (sparse_pauli_z(n=l, L=self.l).toarray())
                    else:
                        l = l - 1
                        self.mpo[0, -1] = (
                            self.mpo[0, -1]
                            @ sparse_pauli_z(n=l, L=self.l).toarray()
                            @ sparse_pauli_z(n=l + 1, L=self.l).toarray()
                        )
            mpo_tot.append(self.mpo)
            self.mpo_skeleton(aux_dim=2)

        self.mpo = mpo_tot
        return self

    def wilson_Z2_dual(self, mpo_sites: list, ls: list):
        """
        wilson_Z2_dual

        This function finds the wilson loop given by a list of indices in the lattice.

        mpo_sites: list - define the x axis of the lattice, starts from zero (left). n° of rungs
        ls: list - define the y axis of the lattice, starts from 1 (up). n° of ladders

        """
        # initialize
        self.mpo_skeleton()
        I = identity(2**self.l, dtype=complex).toarray()
        mpo_tot = []

        for site in range(self.L - 1):
            if len(mpo_sites) == 1:
                # we are between two rungs
                if site in mpo_sites and (mpo_sites[0] == 0):
                    self.mpo[0, 1] = self.pauli_string(
                        string=ls, direction="vertical", pauli_type="X"
                    )
                elif site in mpo_sites:
                    self.mpo[1, -1] = self.pauli_string(
                        string=ls, direction="vertical", pauli_type="X"
                    )
            # take the string of pauli on the ladders that creates the interaction among the mpo sites
            else:
                if site == mpo_sites[0]:
                    self.mpo[0, 1] = self.pauli_string(
                        string=ls, direction="vertical", pauli_type="X"
                    )
                # take the string of pauli on the ladders that ends the interaction among the mpo sites
                elif site == mpo_sites[-1]:
                    self.mpo[1, -1] = self.pauli_string(
                        string=ls, direction="vertical", pauli_type="X"
                    )
                # take the string of pauli on the ladders that continues the interaction among the mpo sites
                elif site in mpo_sites[1:-1]:
                    self.mpo[1, 1] = self.pauli_string(
                        string=ls, direction="vertical", pauli_type="X"
                    )

            if site < mpo_sites[0]:
                # before the pauli strings, place identities for the previous mpo sites
                self.mpo[0, 1] = I
            if site > mpo_sites[-1]:
                # after the pauli strings, place identities for the remaining mpo sites
                self.mpo[1, -1] = I

            mpo_tot.append(self.mpo)
            self.mpo_skeleton()

        self.mpo = mpo_tot
        return self

    def pauli_string(self, string: list, direction: str, pauli_type: str):
        """
        pauli_string

        This function creates a string of certain paulis in a given direction
        of the lattice.

        string: list - list of the sites on the dual lattice. Starts from 1
        direction: str - direction of the string on the sites of the dual lattice
        pauli_type: str - type of pauli matrix we want to use. Available are: "X", "Y", "Z"

        """
        # define the pauli type
        if pauli_type == "X":
            sigma = sparse_pauli_x
        elif pauli_type == "Y":
            sigma = sparse_pauli_y
        elif pauli_type == "Z":
            sigma = sparse_pauli_z

        # define the direction
        if direction == "horizontal":
            dim = self.L - 1
        elif direction == "vertical":
            dim = self.l

        # compute the string
        pauli = identity(n=2**dim, dtype=complex)
        for n in string:
            pauli = pauli @ sigma(n=n, L=dim)

        return pauli.toarray()

    def mpo_Z2_vertical_right_edges_pbc(self, file):
        # mpo_list = []
        # self.mpo_skeleton()
        # for c in range(self.L):
        #     if c == self.L - 1:
        #         self.mpo[0, file + 1] = sparse_pauli_z(n=file, L=self.l).toarray()
        #     mpo_list.append(self.mpo)
        #     self.mpo_skeleton()

        # l_aux = self.l
        # self.l = 1
        # self.mpo_skeleton(aux_dim=(l_aux + 2))
        # self.l = l_aux
        # for f in range(self.l):
        #     coeff = np.prod(np.prod(self.charges, axis=1).tolist()[: f + 1])
        #     self.mpo[f + 1, -1] = (
        #         -self.lamb * coeff * sparse_pauli_z(n=0, L=1).toarray()
        #     )
        # self.mpo = self.mpo[:, -1].reshape((self.l + 2, 1, 2, 2))
        # mpo_list.append(self.mpo)

        # mpo_list = []
        # self.mpo_skeleton(aux_dim=3)
        # for c in range(self.L):
        #     if c == self.L - 1:
        #         self.mpo[0, 1] = sparse_pauli_z(n=file, L=self.l).toarray()
        #     mpo_list.append(self.mpo)
        #     self.mpo_skeleton(aux_dim=3)

        # l_aux = self.l
        # self.l = 1
        # self.mpo_skeleton(aux_dim=3)
        # self.l = l_aux
        
        # # coeff = np.prod(np.prod(self.charges, axis=1).tolist()[: file + 1])
        # self.mpo[1, -1] = sparse_pauli_z(n=0, L=1).toarray()
        # self.mpo = self.mpo[:, -1].reshape((3, 1, 2, 2))
        mpo_list = []
        self.mpo_skeleton(aux_dim=2)
        for c in range(self.L):
            if c == self.L - 1:
                self.mpo[0, 1] = sparse_pauli_z(n=file, L=self.l).toarray() @ sparse_pauli_z(n=(self.l-1), L=self.l).toarray()
            mpo_list.append(self.mpo)
            self.mpo_skeleton(aux_dim=2)

        self.mpo = mpo_list
        return mpo_list

    def local_observable_Z2_dual(self, mpo_site, l):
        """
        local_observable_Z2_dual

        This function finds the mpo representing local observables in the
        dual lattice of the Z2 theory.

        """
        assert (
            0 <= mpo_site < self.L
        ), "The mpo site is out of bound. Choose it 0 <= site < L-1"
        assert (
            0 <= l < self.l
        ), "The mpo ladder is out of bound. Choose it 0 <= site < l"

        self.mpo_skeleton(aux_dim=2)

        mpo_tot = []
        for site in range(self.L):
            if mpo_site == site:
                self.mpo[0, -1] = sparse_pauli_z(n=l, L=self.l).toarray()

            mpo_tot.append(self.mpo)
            self.mpo_skeleton(aux_dim=2)

        self.mpo = mpo_tot
        return self

    def zz_observable_Z2_dual(
        self, mpo_site: int, l: int, direction: str, aux_dim: int = 2
    ):
        """
        zz_observable_Z2_dual

        This function finds the mpo representing the interacting observales in the
        dual lattice of the Z2 theory. We make a distinction in the direction of
        the interaction because in our mps formalism the vertical interaction falls
        back into a local one.

        """
        self.mpo_skeleton(aux_dim=aux_dim)

        mpo_tot = []
        i = 0

        for site in range(self.L):
            if direction == "horizontal":
                assert (
                    0 <= mpo_site < self.L - 1
                ), "The mpo site is out of bound. Choose it 0 <= site < L-2"
                if (mpo_site + i) == site:
                    if i == 0:
                        self.mpo[0, 1] = sparse_pauli_z(n=l, L=self.l).toarray()
                    else:
                        self.mpo[1, -1] = sparse_pauli_z(n=l, L=self.l).toarray()
                    i = 1
            elif direction == "vertical":
                assert (
                    -1 <= l < self.l - 1
                ), "The mpo ladder is out of bound. Choose it 0 <= site < l-1"
                if mpo_site == site:
                    if self.bc == "obc":
                        self.mpo[0, -1] = (
                            sparse_pauli_z(n=l, L=self.l).toarray()
                            @ sparse_pauli_z(n=l + 1, L=self.l).toarray()
                        )
                    elif self.bc == "pbc":
                        self.mpo[0, -1] = (
                            sparse_pauli_z(n=l%self.l, L=self.l).toarray()
                            @ sparse_pauli_z(n=(l + 1), L=self.l).toarray()
                        )

            mpo_tot.append(self.mpo)
            self.mpo_skeleton(aux_dim=aux_dim)

        self.mpo = mpo_tot
        return self

    def zz_string_correlator_Z2_dual(self, cx: list, cy: list, aux_dim: int = 2):
        """
        zz_string_correlator_Z2_dual

        This function computes the spin-spin interaction in the dual lattice along
        the path of the string of the direct lattice. Thus, the direction of the
        interaction is in the vertical direction when the string lies on a path
        made out of the horizontal links.

        cx: list - coordinates of the charges in the x direction
        cy: list - coordinates of the charges in the y direction

        """
        l = cy[0] - 1
        self.mpo_skeleton(aux_dim=aux_dim)
        mpo_tot = []
        for site in range(self.L):
            if site in range(cx[0], cx[1]):
                if self.bc == "obc":
                    self.mpo[0, -1] = (
                        sparse_pauli_z(n=l, L=self.l).toarray()
                        @ sparse_pauli_z(n=l + 1, L=self.l).toarray()
                    )
                elif self.bc == "pbc":
                    self.mpo[0, -1] = (
                        sparse_pauli_z(n=l%self.l, L=self.l).toarray()
                        @ sparse_pauli_z(n=(l + 1), L=self.l).toarray()
                    )
            mpo_tot.append(self.mpo)
            self.mpo_skeleton(aux_dim=aux_dim)

        self.mpo = mpo_tot
        return self

    def zz_vertical_right_pbc_Z2_dual(self, mpo_site: int, l: int, aux_dim: int = 2
    ):
        """
        zz_observable_Z2_dual

        This function finds the mpo representing the interacting observales in the
        dual lattice of the Z2 theory. We make a distinction in the direction of
        the interaction because in our mps formalism the vertical interaction falls
        back into a local one.

        """
        self.mpo_skeleton(aux_dim=aux_dim)

        mpo_tot = []
        i = 0
        for site in range(self.L):
            
            if (mpo_site + i) == site:
                    if i == 0:
                        self.mpo[0, 1] = sparse_pauli_z(n=l, L=self.l).toarray()
                    else:
                        self.mpo[1, -1] = sparse_pauli_z(n=0, L=1).toarray()
                    i = 1
                
            mpo_tot.append(self.mpo)
            self.mpo_skeleton(aux_dim=aux_dim)
        
        # self.mpo[1, -1] = sparse_pauli_z(n=0, L=1).toarray()
        # # self.mpo[1,-1] = self.mpo[1, -1].reshape((2, 1, 2, 2))
        # mpo_tot.append(self.mpo)

        self.mpo = mpo_tot
        return self