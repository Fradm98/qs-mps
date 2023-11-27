from qs_mps.utils import *
from qs_mps.sparse_hamiltonians_and_operators import sparse_pauli_x, sparse_pauli_z
from qs_mps.applications.Z2.lattice import Lattice
import numpy as np
from scipy.sparse import identity, csc_array, linalg

class H_Z2_gauss():

    def __init__(
    self, L, l, model=str, lamb=None
    ):
        self.L = L
        self.l = l
        self.model = model
        self.charges = np.ones((l+1,L))
        self.lamb = lamb
        self.latt = Lattice((self.L,self.l), (False,False))
        self.dof = self.latt.nlinks
        
    def local_term(self, link):
        sigma_x = sparse_pauli_x(link, self.latt.nlinks)
        return sigma_x

    def plaquette_term(self, loop):
        plaq = identity(n=2**self.latt.nlinks)
        for link in loop:
            plaq = plaq @ sparse_pauli_z(n=link,L=self.latt.nlinks)
        return plaq
    
    def hamiltonian(self):
        loc = csc_array((2**self.latt.nlinks,2**self.latt.nlinks))
        # local terms
        for link in range(self.latt.nlinks):
            loc += self.local_term(link)

        plaq = csc_array((2**self.latt.nlinks,2**self.latt.nlinks))
        # plaquette terms
        for loop in self.latt.plaquettes(from_zero=True):
            plaq += self.plaquette_term(loop)

        return - loc - self.lamb * plaq
    
    def diagonalize(self):
        H = self.hamiltonian()
        e, v = linalg.eigsh(H, k=30)
        return e, v
    

Z2_exact = H_Z2_gauss(L=3, l=3, model="Z2", lamb=0)
e, v = Z2_exact.diagonalize()
print(f"charges:\n{Z2_exact.charges}")
print(f"degrees of freedom:\n{Z2_exact.dof}")
print(f"spectrum:\n{e}")
