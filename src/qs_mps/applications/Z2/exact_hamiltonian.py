from qs_mps.utils import *
from qs_mps.sparse_hamiltonians_and_operators import sparse_pauli_x, sparse_pauli_z
from qs_mps.lattice import Lattice
import numpy as np
from scipy.sparse import identity, csc_array, linalg

class H_Z2_gauss():

    def __init__(
    self, L, l, model:str, lamb:None, U:float,
    ):
        self.L = L
        self.l = l
        self.model = model
        self.charges = np.ones((l,L))
        self.lamb = lamb
        self.U = U
        self.latt = Lattice((self.L,self.l), (False,False))
        self.dof = self.latt.nlinks


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
            self.charges[j,i] = -1
        
        self.charges = np.flip(self.charges, axis=1)
        return self
        
    def local_term(self, link):
        sigma_x = sparse_pauli_x(n=link, L=self.latt.nlinks)
        return sigma_x

    def plaquette_term(self, loop):
        plaq = identity(n=2**self.latt.nlinks)
        for link in loop:
            plaq = plaq @ sparse_pauli_z(n=link,L=self.latt.nlinks)
        return plaq
    
    def gauge_constraint(self,site):
        links = self.latt.star(site=site, L=self.L, l=self.l)
        G = identity(n=2**self.latt.nlinks)
        filtered_links = [element for element in links if element != 0]
        # print("links:")
        # print(filtered_links)
        for link in filtered_links:
            G = G @ sparse_pauli_x(n=link-1, L=self.latt.nlinks)

        return G

    def hamiltonian(self):
        loc = csc_array((2**self.latt.nlinks,2**self.latt.nlinks))
        # local terms
        for link in range(self.latt.nlinks):
            loc += self.local_term(link)

        plaq = csc_array((2**self.latt.nlinks,2**self.latt.nlinks))
        # plaquette terms
        for loop in self.latt.plaquettes(from_zero=True):
            plaq += self.plaquette_term(loop)

        # gauge constraint
        G = 0
        I = identity(n=2**self.latt.nlinks)
        for site in self.latt.sites:
            # print(site)
            g = self.gauge_constraint(site)
            G += (g - self.charges[site[1],site[0]]*I) @ (g - self.charges[site[1],site[0]]*I)
        return - loc - (self.lamb * plaq) + (self.U * G)
    
    def diagonalize(self):
        H = self.hamiltonian()
        e, v = linalg.eigsh(H, k=2**len(self.latt.plaquettes()), which='SA')
        # print(H.toarray())
        # e, v = np.linalg.eigh(H.toarray())
        return e, v

# Z2_exact = H_Z2_gauss(L=3, l=3, model="Z2", lamb=1e-7, U=1e+3)
# # Z2_exact.add_charges([0,2],[1,1])
# print(f"charges:\n{Z2_exact.charges}")
# print(f"degrees of freedom:\n{Z2_exact.dof}")
# print(f"lattice:\n{Z2_exact.latt._lattice_drawer.draw_lattice()}")
# e, v = Z2_exact.diagonalize()
# print(f"spectrum:\n{np.sort(e)}")
