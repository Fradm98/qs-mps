from qs_mps.utils import *
from qs_mps.sparse_hamiltonians_and_operators import sparse_pauli_x, sparse_pauli_z
from qs_mps.lattice import Lattice
import numpy as np
from scipy.sparse import identity, csc_array, linalg
from ncon import ncon
import time

class H_Z2_gauss:
    def __init__(
        self,
        l,
        L,
        model: str,
        lamb: float=0,
        J: float=1,
        U: float=1e+3,
    ):
        self.L = L
        self.l = l
        self.model = model
        self.charges = np.ones((l, L))
        self.lamb = lamb
        self.J = J
        self.U = U
        self.latt = Lattice((self.L, self.l), (False, False))
        self.dof = self.latt.nlinks
        self.sector = self._define_sector()

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

        self.charges = np.flip(self.charges, axis=0)
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

    def local_term(self, link):
        sigma_x = sparse_pauli_x(n=link, L=self.latt.nlinks)
        return sigma_x

    def plaquette_term(self, loop):
        plaq = identity(n=2**self.latt.nlinks)
        for link in loop:
            plaq = plaq @ sparse_pauli_z(n=link, L=self.latt.nlinks)
        return plaq

    def gauge_constraint(self, site):
        links = self.latt.star(site=site, L=self.L, l=self.l)
        G = identity(n=2**self.latt.nlinks)
        filtered_links = [element for element in links if element != 0]
        # print("links:")
        # print(filtered_links)
        for link in filtered_links:
            G = G @ sparse_pauli_x(n=link - 1, L=self.latt.nlinks)

        return G

    def hamiltonian(self):
        loc = csc_array((2**self.latt.nlinks, 2**self.latt.nlinks))
        # local terms
        for link in range(self.latt.nlinks):
            loc += self.local_term(link)

        plaq = csc_array((2**self.latt.nlinks, 2**self.latt.nlinks))
        # plaquette terms
        for loop in self.latt.plaquettes(from_zero=True):
            plaq += self.plaquette_term(loop)

        # gauge constraint
        G = 0
        I = identity(n=2**self.latt.nlinks)
        for site in self.latt.sites:
            # print(site)
            g = self.gauge_constraint(site)
            G += (g - self.charges[site[1], site[0]] * I) @ (
                g - self.charges[site[1], site[0]] * I
            )
        return - (self.J * loc) - (self.lamb * plaq) + (self.U * G)

    def diagonalize(self, v0: np.ndarray=None, sparse: bool=True, save: bool=True, path: str=None, precision: int=2, spectrum: str="gs", cx: list=None, cy: list=None):
        H = self.hamiltonian()

        if sparse:
            if spectrum == "all":
                k = (2 ** len(self.latt.plaquettes()))
            elif spectrum == "gs":
                k = 1
            e, v = linalg.eigsh(H, k=k, which="SA", v0=v0)
        else:
            H = H.toarray()
            e, v = np.linalg.eigh(H)
        if save:
            # print(self.sector)
            np.save(path+f"/results/eigenvectors/ground_state_direct_lattice_{self.l-1}x{self.L-1}_{self.sector}_{cx}-{cy}_U_{self.U}_h_{self.lamb:.{precision}f}.npy", v[:,0])
        return e, v

    # observables
    def wilson_loop(self, psi, sites):
        loop = self.latt.plaquettes(from_zero=True)
        plaq = self.plaquette_term(loop[sites])
        exp_val_wilson_loop = np.real(psi.T @ plaq @ psi)
        return exp_val_wilson_loop
    
    def h_electric_field(self, psi):
        exp_val_h_links = []
        for link in range((self.l*(self.L-1))):
            obs = self.local_term(link)
            exp_val_electric_field = np.real(psi.T @ obs @ psi)
            exp_val_h_links.append(exp_val_electric_field)
        exp_val_h_links = np.array(exp_val_h_links).reshape((self.l,self.L-1))
        return exp_val_h_links
    
    def v_electric_field(self, psi):
        exp_val_v_links = []
        for link in range((self.l*(self.L-1)),self.dof):
            obs = self.local_term(link)
            exp_val_electric_field = np.real(psi.T @ obs @ psi)
            exp_val_v_links.append(exp_val_electric_field)
        exp_val_v_links = np.array(exp_val_v_links).reshape((self.l-1,self.L))
        return exp_val_v_links
    
    def electric_field(self, psi, E):
        E[::2,1::2] = self.h_electric_field(psi)
        E[1::2,::2] = self.v_electric_field(psi)
        return E

    def v_thooft_idx(self, plaq_tot_spl: np.ndarray, mpo_site: int, l: int):
        plaqs = [pl for pl in plaq_tot_spl[mpo_site]].copy()
        # plaqs.reverse()
        pauli = []
        for i in range(self.l-(l+1)):
            pauli = pauli + [plaqs[i][0]]

        return pauli
    
    def h_thooft_idx(self, plaq_tot_spl: np.ndarray, mpo_site: int, l: int):
        plaqs = np.swapaxes(plaq_tot_spl, axis1=0, axis2=1)
        plaqs = [pl for pl in plaqs].copy()
        plaqs.reverse()
        plaqs_h = plaqs[l].copy()
        pauli = []
        for i in range(mpo_site+1):
            pauli = pauli + [plaqs_h[i][3]]
        
        return pauli

    def thooft(self, psi: np.ndarray, mpo_site: int, l: int, direction: str):
        plaq_tot = self.latt.plaquettes(from_zero=True)
        plaq_tot_spl = np.array_split(plaq_tot, self.L-1)
        if direction == "vertical":
            pauli = self.v_thooft_idx(plaq_tot_spl=plaq_tot_spl, 
                                      mpo_site=mpo_site, 
                                      l=l)
        if direction == "horizontal":
            pauli = self.h_thooft_idx(plaq_tot_spl=plaq_tot_spl, 
                                      mpo_site=mpo_site, 
                                      l=l)
            
        op = identity(n=2**self.latt.nlinks)
        for idx in pauli:
            op = op @ sparse_pauli_x(n=idx, L=self.latt.nlinks)
        
        thooft_string = (psi.T @ op @ psi).real
        return thooft_string


# lamb = 0
# U = 1e+3
# Z2_exact = H_Z2_gauss(L=4, l=3, model="Z2_dual", lamb=lamb, U=U)
# Z2_exact.add_charges([0,2],[1,1])
# print(f"charges:\n{Z2_exact.charges}")
# print(f"degrees of freedom:\n{Z2_exact.dof}")
# print(f"lattice:\n{Z2_exact.latt._lattice_drawer.draw_lattice()}")
# e, v = Z2_exact.diagonalize(save=False, sparse=False)
# psi = v[:,0]
# print(psi)
# print(f"spectrum:\n{e}")
# # print(f"Delta Energy gs - ex: {e[0]-e[1]}\nth: {8*lamb}")
# # print(f"H:\n{H.toarray()}")
# print(f"ground state:\n{v[:,0]}")
# X = sparse_pauli_z(n=0,L=4) @ sparse_pauli_z(n=1,L=4) @ sparse_pauli_z(n=2,L=4) @ sparse_pauli_z(n=3,L=4) 

# exp_val = (psi.T @ X @ psi).real
# print(exp_val)


# from qs_mps.mpo_class import MPO_ladder
# exp_val = []
# en = []
# en_1 = []
# en_mpo = []
# en_mpo_1 = []
# # delta_e = []
# hs = list(np.arange(-6, 6, 0.1))
# hs.reverse()
# hs.pop()
# l = 2
# L = 2
# dof = (2*l*L - l - L)
# v0 = np.array([-0.25 for _ in range(2**dof)])
# for h in hs:
#     Z2_mpo = MPO_ladder(L=L, l=l-1, model="Z2", lamb=h)
#     e, v = Z2_mpo.diagonalize()
#     en_mpo.append(e[0])
#     en_mpo_1.append(e[1])
#     Z2_exact = H_Z2_gauss(L=L, l=l, model="Z2", lamb=h, U=1e+3)
#     H, e, v = Z2_exact.diagonalize(v0=v0)
#     print(e)
#     en.append(e[0])
#     en_1.append(e[1])
#     # delta_e.append(np.abs(e[1]-e[0]))
#     # print(f"spectrum:\n{e}")
#     # print(f"{Z2_exact.latt.plaquettes()[1]}")
#     # loop = Z2_exact.latt.plaquettes(from_zero=True)[1]
#     # loop = Z2_exact.plaquette_term(loop=loop)
#     psi = v[:,0]
#     # # exp = ncon([psi.T,loop.toarray(),psi],[[1],[1,2],[2]]).real
#     # exp = (psi.T @ loop @ psi).real
#     # exp_val.append(exp)
#     v0 = psi

# print(f"lattice:\n{Z2_exact.latt._lattice_drawer.draw_lattice()}")
# # plt.plot(hs, np.abs(exp_val))
# # plt.show()
# plt.plot(hs, en_mpo, 'o', color='g')
# plt.plot(hs, en_mpo_1, '-', color='g')
# plt.plot(hs, en, '--', color='r')
# plt.plot(hs, en_1, '-', color='r')
# plt.show()
# # plt.plot(hs, 8*np.asarray(hs))
# # plt.show()