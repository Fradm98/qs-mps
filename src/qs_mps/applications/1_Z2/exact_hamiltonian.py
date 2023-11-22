from qs_mps.utils import *
from qs_mps.sparse_hamiltonians_and_operators import sparse_pauli_x, sparse_pauli_z
from .lattice import Lattice
import numpy as np
from scipy.sparse import identity, csc_array

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
        
    def local_term(self, link):
        sigma_x = sparse_pauli_x(link, self.latt.nlinks)

    def plaquette_term(self, site):
        plaq = identity(n=self.latt.nlinks)
        for link in self.latt.plaquette(site, from_zero=True):
            plaq = plaq @ sparse_pauli_z(n=link,L=self.latt.nlinks)
        return plaq
    
    def hamiltonian(self):
        loc = csc_array((self.latt.nlinks,self.latt.nlinks))
        # local terms
        for link in range(self.latt.nlinks):
            loc += self.local_term(link)

        plaq = csc_array((self.latt.nlinks,self.latt.nlinks))
        # plaquette terms
        for site in self.latt.sites:
            plaq += self.plaquette_term(site)

        return - loc - self.lamb * plaq
    
    def diagonalize(self):
        H = self.hamiltonian()
        e, v = np.linalg.eigh(H)
        return e, v
    

Z2_exact = H_Z2_gauss(L=3, l=3, model="Z2", lamb=0)
e, v = Z2_exact.diagonalize()
print(f"spectrum:\n{e}")