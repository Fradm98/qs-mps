# %%
from scipy.sparse import csc_array, identity
from ncon import ncon
from qs_mps.exact_Ising_ground_state_and_time_evolution import sparse_pauli_z, sparse_pauli_x
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
        
    def mpo_skeleton(self):
        I = identity(2**self.l)
        O = csc_array((2**self.l,2**self.l))
        skeleton = np.array([[O.toarray() for i in range(2+self.l)] for j in range(2+self.l)])
        skeleton[0,0] = I.toarray()
        skeleton[-1,-1] = I.toarray()
        self.mpo = skeleton
        return self

    def charge_coeff_interaction(self, n: int, charge_direct: int):
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
        charge_direct: int - the charge column we want to multiply, starting from
            the row below the one of the interaction. Accepted values are from 
            2 to (self.L - 1) (we consider the index starting from 1)

        """
        assert 1 <= n <= self.l, ("Select a value of n within (1,l)") 
        assert 2 <= charge_direct <= (self.L - 1), ("Select a value of charge column within (2,L-1)") 

        if charge_direct == (self.L - 1):
            c_n_j = 1
        else:
            c_n_j = np.prod(self.charges[:(n+1),charge_direct-1])
        return c_n_j

    def charge_coeff_local_Z(self, n, charge_direct):
        """
        charge_coeff_local_Z

        This function computes the coefficient fro the local Z term in the
        ladder Z2 hamiltonian.

        There are three different coefficient formulas:
        if edge

        """
        if (charge_direct-1) == 1:
            c_n_j = (1+self.charges[n,0])
        pass

    def mpo_edge_l_interaction(self):
        charge_direct = 2
        
        # -----------
        # first row
        # -----------

        # interaction terms (from column 2 to l+1)
        for n in range(1,self.l+1):
            self.mpo[0,n] = - self.charge_coeff_interaction(n=n, charge_direct=charge_direct) * sparse_pauli_z(n=n-1, L=self.l).toarray() 

        # local terms (column l+2)
        for i in range(1,self.l+1):
            self.mpo[0,n+1] += - self.charge_coeff_local_Z() * sparse_pauli_z(n=i-1, L=self.l).toarray() - self.lamb * sparse_pauli_x(n=i-1, L=self.l).toarray()
        for j in range(self.l-1):
            self.mpo[0,n+1] += - (sparse_pauli_z(n=j, L=self.l) @ sparse_pauli_z(n=j+1, L=self.l)).toarray()

        # -----------
        # last column
        # -----------
        # positive terms to complete the interaction between nearest neighbor sites (from row 2 to l+1)
        for n in range(1,self.l+1):
            self.mpo[n,-1] = sparse_pauli_z(n=n-1, L=self.l).toarray()

# %%
