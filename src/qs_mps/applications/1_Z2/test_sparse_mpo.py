# %%
from scipy.sparse import csc_array, identity
from ncon import ncon
from qs_mps.sparse_hamiltonians_and_operators import sparse_pauli_z, sparse_pauli_x
import numpy as np

# %%

class MPO_ladder():
    def __init__(
        self, L, l, model=str, lamb=None
    ):
        self.L = L
        self.l = l
        self.model = model
        self.charges = np.ones((l+1,L))
        self.lamb = lamb
        self.mpo = []
        
    def charge_constraint(self):
        """
        charge_constraint

        We decide arbitrarily to impose a charge constraint on the bottom right
        corner of the direct lattice.

        """
        # take the product of the first l-1 rows
        productory = np.prod([self.charges[line,:] for line in range(self.l-1)])
        # take the last row but the last charge
        productory = productory * np.prod([self.charges[-1,:-1]])
        self.charges[-1,-1] = productory
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
        for i,j in zip(rows,columns):
            assert ((i != -1) and (j != -1)) or ((i == self.l-1) and (j == self.L-1)), ("Do not choose the last charge! We use it for Gauss Law constraint")
            self.charges[i,j] = -1
        # impose the constraint
        self.charge_constraint()
        return self
    
    def mpo_skeleton(self):
        """
        mpo_skeleton

        This function initializes the mpo tensor or shape (2+l,2+l,2**l,2**l)
        with O matrices. We add as well the identities in the first and last 
        element of the mpo tensor.

        """
        I = identity(2**self.l)
        O = csc_array((2**self.l,2**self.l))
        skeleton = np.array([[O.toarray() for i in range(2+self.l)] for j in range(2+self.l)])
        skeleton[0,0] = I.toarray()
        skeleton[-1,-1] = I.toarray()
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
        assert 1 <= n <= self.l, ("Select a value of n within (1,l)") 
        assert 0 <= mpo_site <= (self.L - 2), ("Select a value of charge column within (0,L-2)") 

        if mpo_site == (self.L - 2):
            c_n_j = 1
        else:
            c_n_j = np.prod(self.charges[n:,mpo_site+1])
        return c_n_j

    def charge_coeff_local_Z(self, n: int, mpo_site: int):
        """
        charge_coeff_local_Z

        This function computes the coefficient fro the local Z term in the
        ladder Z2 hamiltonian.

        """
        assert 1 <= n <= self.l, ("Select a value of n within (1,l)") 

        if mpo_site == 0 or mpo_site == (self.L-2):
            if mpo_site == 0:
                col = mpo_site
                if n == 1:
                    alpha = 1
                    n = n-1
                elif n == self.l:
                    alpha = 1
                    n = n
                else:
                    alpha = 0
            elif mpo_site == (self.L-2):
                col = mpo_site + 1
                if n == 1:
                    alpha = 1
                    n = n-1
                elif n == self.l:
                    alpha = 1
                    n = n
                else:
                    alpha = 0
                
            c_n_j = (1+self.charges[n,col])**(alpha) * np.prod(self.charges[n+1:,col])
        else:
            if n == 1:
                c_n_j = np.prod([self.charges[:,col] for col in range(mpo_site+1)])
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
        for mpo_site in range(self.L-1):

            # -----------
            # first row
            # -----------
            # interaction terms (from column 2 to l+1)
            for n in range(1,self.l+1):
                self.mpo[0,n] = - self.charge_coeff_interaction(n=n, mpo_site=mpo_site) * sparse_pauli_z(n=n-1, L=self.l).toarray() 

            # local terms (column l+2)
            # local Z and local X
            for i in range(1,self.l+1):
                self.mpo[0,n+1] += - self.charge_coeff_local_Z(n=i, mpo_site=mpo_site) * sparse_pauli_z(n=i-1, L=self.l).toarray() - self.lamb * sparse_pauli_x(n=i-1, L=self.l).toarray()
            # vertical Z interaction
            for j in range(self.l-1):
                self.mpo[0,n+1] += - (sparse_pauli_z(n=j, L=self.l) @ sparse_pauli_z(n=j+1, L=self.l)).toarray()

            # -----------
            # last column
            # -----------
            # positive terms to complete the interaction between nearest neighbor sites (from row 2 to l+1)
            for n in range(1,self.l+1):
                self.mpo[n,-1] = sparse_pauli_z(n=n-1, L=self.l).toarray()

            mpo_list.append(self.mpo)

        self.mpo = mpo_list
        return mpo_list


# %%
mpo = MPO_ladder(l=3, L=5)
rows = []
columns = []
mpo.add_charges(rows, columns)
mpo.charge_coeff_local_Z(n=3, mpo_site=3)