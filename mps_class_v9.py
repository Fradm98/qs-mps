import numpy as np
from ncon import ncon
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm
from utils import variance, tensor_shapes, get_labels
import time

class MPS:
    def __init__(self, L, d, model=str, chi=None, w=None, h=None, eps=None, J=None, charges=None):
        self.L = L
        self.d = d
        self.model = model
        self.chi = chi
        self.w = w
        self.w_dag = w
        self.h = h
        self.eps = eps
        self.J = J
        self.charges = charges
        self.sites = []
        self.bonds = []
        self.ancilla_sites = []
        self.ancilla_bonds = []
        self.schmidt_left = []
        self.schmidt_right = []
        self.env_left = []
        self.env_right = []
        self.env_left_sm = []
        self.env_right_sm = []

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
            chi = int(np.log2(chi))
            assert (
                self.L >= 2 * chi
            ), "The spin chain is too small for the selected bond dimension chi"
            np.random.seed(seed)

            for i in range(chi):
                sites.append(np.random.rand(self.d**i, self.d, self.d ** (i + 1)))
            for _ in range(self.L - (2 * chi)):
                sites.append(np.random.rand(self.d**chi, self.d, self.d**chi))
            for i in range(chi):
                sites.append(
                    np.random.rand(self.d ** (chi - i), self.d, self.d ** (chi - i - 1))
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

        return self

    def canonical_form(self, svd_direction="left", e_tol=10 ** (-15), ancilla=False):
        """
        canonical_form

        This function puts the tensors saved in self.sites through recursive svd.
        It corresponds in saving tensors in the (Vidal) Gamma-Lambda notation.
        It can be used both to initialize the random tensors in a normalized state
        or to bring the tensors from the amb form to the canonical one.

        svd_direction: string - the direction of the sequencial svd. Could be "right" or "left"
        e_tol: float - tolerance used to cut the schmidt values after svd

        """
        if svd_direction == "left":
            self.left_svd(e_tol, ancilla)

        elif svd_direction == "right":
            self.right_svd(e_tol, ancilla)

        return self

    def right_svd(self, e_tol, ancilla):
        """
        right_svd

        This function transforms the states in self.sites in a canonical
        form using svd. We start from the first site and sweeping through
        site self.L we save the Gamma tensors on each site and the Schmidt values on the bonds

        e_tol: float - tolerance used to cut the schmidt values after svd

        """
        s_init = np.array([1])
        psi = np.diag(s_init)

        sites = self.sites
        bonds = self.bonds
        if ancilla:
            sites = self.ancilla_sites
            bonds = self.ancilla_bonds

        bonds.append(s_init)
        for i in range(self.L - 1):
            new_site = ncon(
                [psi, sites[i]],
                [
                    [-1,1],
                    [1,-2,-3],
                ],
            )
            u, s, v = np.linalg.svd(
                new_site.reshape(new_site.shape[0] * self.d, new_site.shape[2]),
                full_matrices=False,
            )
            condition = s >= e_tol
            s_trunc = np.extract(condition, s)
            s_trunc = s_trunc / np.linalg.norm(s_trunc)
            s_trunc = s / np.linalg.norm(s)
            bond_l = u.shape[0] // self.d
            u = u.reshape(bond_l, self.d, u.shape[1])
            u = u[:, :, : len(s_trunc)]
            sites[i] = u
            bonds.append(s_trunc)
            v = v[: len(s_trunc), :]
            psi = ncon(
                [np.diag(s_trunc), v],
                [
                    [-1,1],
                    [1,-2],
                ],
            )
        bond_r = v.shape[1] // self.d
        sites[-1] = psi.reshape(v.shape[0], self.d, bond_r)
        bonds.append(s_init)

        return self

    def left_svd(self, e_tol, ancilla):
        """
        left_svd

        This function transforms the states in self.sites in a canonical
        form using svd. We start from the last site self.L and sweeping through
        site 1 we save the Gamma tensors on each site and the Schmidt values on the bonds

        e_tol: float - tolerance used to cut the schmidt values after svd

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
                    [-1,-2,1],
                    [1,-3],
                ],
            )
            u, s, v = np.linalg.svd(
                new_site.reshape(new_site.shape[0], self.d * new_site.shape[2]),
                full_matrices=False,
            )
            condition = s >= e_tol
            s_trunc = np.extract(condition, s)
            s_trunc = s_trunc / np.linalg.norm(s_trunc)
            s_trunc = s / np.linalg.norm(s)
            bond_r = v.shape[1] // self.d
            v = v.reshape(v.shape[0], self.d, bond_r)
            v = v[: len(s_trunc), :, :]
            sites[i] = v
            bonds.append(s_trunc)
            u = u[:, : len(s_trunc)]
            psi = ncon(
                [u, np.diag(s_trunc)],
                [
                    [-1,1],
                    [1,-2],
                ],
            )
            
        bonds.append(s_init)
        bonds.reverse()
        # print(f"Time of svd during canonical form: {time.perf_counter()-time_cf}")
        # np.savetxt(f"times_data/svd_canonical_form_h_{self.h:.2f}", [time.perf_counter()-time_cf])
        return self

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
            [env_left[-1], ket[site - 1], w[site - 1], bra[site - 1].conjugate(), env_right[-1]],
            [
                [1,4,7],
                [1,3,2],
                [4,5,3,6],
                [7,6,8],
                [2,5,8]
            ]
        )
        return sandwich

    def overlap_sites(self, array_1, array_2=None):
        if array_2 == None:
            array_2 = array_1
        return ncon([array_1, array_2.conjugate()],[[-1,1,-3],[-2,1,-4]])
    
    def _compute_norm(self, site, ancilla=False):
        """
        _compute_norm

        This function computes the norm of our quantum state which is represented in mps.
        It takes the attributes .sites and .bonds of the class which gives us
        mps in canonical form (Vidal notation).

        svd_direction: string - the direction the svd was performed. From left to right
                        is "right" and from right to left is "left"

        """
        array = self.sites
        if ancilla:
            array = self.ancilla_sites
        a = np.array([1])
        env = ncon([a,a,a,a],[[-1],[-2],[-3],[-4]])
        left = env

        for i in range(site-1):
            ten = self.overlap_sites(array_1=array[i])
            env = ncon([env,ten],[[-1,-2,1,2],[1,2,-3,-4]])
        left = env
        print("The left overlap of the state:")
        print(left)
        env = ncon([a,a,a,a],[[-1],[-2],[-3],[-4]])
        right =env
        for i in range(self.L-1, site-1, -1):
            ten = self.overlap_sites(array_1=array[i])
            env = ncon([ten,env],[[-1,-2,1,2],[1,2,-3,-4]])
        right = env
        print("The right overlap of the state:")
        print(right)

        ten_site = self.overlap_sites(array_1=array[site - 1])
        print(f"The tensor in the site {site}:")
        print(ten_site)
        N = ncon([left,ten_site,right],[[-1,-2,1,2],[1,2,3,4],[3,4,-3,-4]])
        N = N[0,0,0,0].real
        print(f"-=-=-= Norm: {N}\n")
        return N

    def mpo(self):
        """
        mpo

        This function selects which MPO to use according to the 
        studied model. Here you can add other MPOs that you have
        independently defined in the class.

        """
        if self.model == 'Ising':
            self.mpo_Ising()

        elif self.model == 'Z2_one_ladder':
            self.mpo_Z2_one_ladder()

        elif self.model == 'Z2_two_ladder':
            self.mpo_Z2_two_ladder()

        return self

    def mpo_Ising(self):
        """
        mpo_Ising

        This function defines the MPO for the 1D transverse field Ising model.
        It takes the same MPO for all sites.

        """
        I = np.eye(2)
        O = np.zeros((2,2))
        X = np.array([[0,1],[1,0]])
        Z = np.array([[1,0],[0,-1]])
        w_tot = []
        for _ in range(self.L):
            w = np.array([[I,-self.J*Z,-self.h*X-self.eps*X],[O,O,Z],[O,O,I]])
            w_tot.append(w)
        self.w = w_tot
        return self
    
    def flipping_mpo(self):
        I = np.eye(2)
        X = np.array([[0,1],[1,0]])
        O = np.zeros((2,2))
        w_tot = []
        for i in range(self.L):
            alpha = 0
            if i >= self.L//2:
                alpha = 1
            w = np.array([[I,alpha*X],[O,I]])
            w_tot.append(w)
        self.w = w_tot
        return self

    def mpo_Z2_one_ladder(self):
        """
        mpo_Z2_one_ladder

        This function defines the MPO for the Z2 lattice gauge theory
        model sitting on one single ladder. It takes a different MPO for 
        the first site and it is the same for the other sites.

        """
        I = np.eye(2)
        O = np.zeros((2,2))
        X = np.array([[0,1],[1,0]])
        Z = np.array([[1,0],[0,-1]])
        w_tot = []
        for i in range(self.L):
            if i == 0:
                theta = 1
            else:
                theta = 0
            w = np.array([[I,-self.J*Z,-2*self.h*theta*X,-self.h*X],[O,O,O,Z],[O,O,X,X @ (np.linalg.matrix_power(X,(1-theta)))],[O,O,O,I]])
            w_tot.append(w)
        self.w = w_tot
        return self
    
    def mpo_Z2_two_ladder(self):
        """
        mpo_Z2_two_ladder

        This function defines the MPO for the Z2 lattice gauge theory
        model sitting on two ladders. It takes a different MPO for the 
        first site and it is the same for the other sites.

        charges: list - list of charges for the Z2 on external vertices.
                their product must be one.
                They are ordered from the upper left vertex: 11,21,31,1N,2N,3N

        """
        charges = self.charges
        assert (np.prod(charges) == 1), "The charges do not multiply to one"

        O_small = np.zeros((2,2))
        I_small = np.eye(2)
        X = np.array([[0,1],[1,0]])
        Z = np.array([[1,0],[0,-1]])
        O_ext = np.kron(O_small,O_small)
        I_ext = np.kron(I_small,I_small)
        O = O_ext
        I = I_ext
        X_1 = np.kron(I_small,X)
        X_2 = np.kron(X,I_small)
        X_12 = np.kron(X,X)
        Z_1 = np.kron(Z,I_small)
        Z_2 = np.kron(I_small,Z)
        w_tot = []
        beta = 0
        for i in range(self.L):
            if i == 0:
                alpha = 1
            else:
                alpha = 0
            if i == (self.L-1):
                beta = 1
            w = np.array([[I,-1/self.h*Z_1,-1/self.h*Z_2,-self.h*charges[0]*alpha*X_1,-self.h*charges[2]*alpha*X_2,-self.h*charges[1]*alpha*X_12,-self.h*X_1-self.h*X_2-beta*1/self.h*(Z_1+Z_2)],
                          [O,O,O,O,O,O,Z_1],
                          [O,O,O,O,O,O,Z_2],
                          [O,O,O,X_1,O,O,X_1  @ (np.linalg.matrix_power(X_1,(1-alpha))) + beta*(1+charges[3])*X_1],
                          [O,O,O,O,X_2,O,X_2  @ (np.linalg.matrix_power(X_2,(1-alpha))) + beta*(1+charges[5])*X_2],
                          [O,O,O,O,O,X_12,X_12  @ (np.linalg.matrix_power(X_12,(1-alpha))) + beta*X_12],
                          [O,O,O,O,O,O,I],
                          ])
            w_tot.append(w)
        self.w = w_tot
        return self
    
    def mpo_Ising_time_ev(self, delta, h_ev, J_ev):
        """
        mpo_Ising_time_ev

        This function defines the MPO for the real time evolution of a 1D transverse field Ising model.
        We use this to perform a second order TEBD.

        delta: float - Trotter step for the time evolution
        h_ev: float - parameter of the local field for the quench
        J_ev: float - parameter of the interaction field for the quench

        """
        I = np.eye(2)
        O = np.zeros((2,2))
        X = np.array([[0,1],[1,0]])
        Z = np.array([[1,0],[0,-1]])
        w_tot = []
        w_loc = np.array(expm(-1j*h_ev*delta/2*X)) 
        w_in = np.array([[np.sqrt(np.cos(J_ev*delta))*I, -1j*np.sqrt(np.sin(J_ev*delta))*Z]])
        w_in = ncon([w_in, w_loc, w_loc],[[-1,-2,1,2],[-3,1],[2,-4]])
        w_fin = np.array([[np.sqrt(np.cos(J_ev*delta))*I, np.sqrt(np.sin(J_ev*delta))*Z]])
        w_fin = ncon([w_fin.T, w_loc, w_loc],[[1,2,-1,-2],[-3,1],[2,-4]])
        w_tot.append(w_in)
        for _ in range(1, self.L-1):
            w = np.array([[np.cos(J_ev*delta)*I,-1j*np.sqrt(np.cos(J_ev*delta)*np.sin(J_ev*delta))*Z],[np.sqrt(np.cos(J_ev*delta)*np.sin(J_ev*delta))*Z, -1j*np.sin(J_ev*delta)*I]])
            w = ncon([w, w_loc, w_loc],[[-1,-2,1,2],[-3,1],[2,-4]])
            w_tot.append(w)
        
        w_tot.append(w_fin)
        self.w = w_tot
        return self
    
    def mpo_Ising_O_dag_O(self):
        """
        mpo_Ising_O_dag_O

        This function creates an mpo given by the product of a previous mpo O
        with its dagger. If O is hermitian then it is equal to perform O^2.

        """
        ws = self.w
        ws_dag = self.w_dag

        w_tot = []
        # for w, w_dag in zip(ws,ws_dag):
        #     w = ncon([w_dag,w], [[1,-5,-3,-1],[-2,-4,1,-6]]).reshape((w.shape[0]*w_dag.shape[3],w.shape[1]*w_dag.shape[2],w.shape[2],w.shape[3]))
        #     w_tot.append(w)

        for w, w_dag in zip(ws,ws_dag):
            w = ncon([w, w_dag], [[-1,-3,-5,1],[-2,-4,1,-6]]).reshape((w.shape[0]*w_dag.shape[0],w.shape[1]*w_dag.shape[1],w.shape[2],w.shape[3]))
            w_tot.append(w)

        self.w = w_tot
        return self
    
    def mpo_dagger(self):
        w_tot = []
        for w in self.w:
            w_dag = w.conjugate()
            w_tot.append(w_dag)
        self.w_dag = w_tot
        return self
    
    def order_param(self):
        """
        order_param

        This function selects which order parameter to use according to the 
        studied model. Here you can add other order parameters that you have
        independently defined in the class.

        """
        if self.model == 'Ising':
            self.order_param_Ising()

        elif self.model == 'Z2':
            self.order_param_Z2()

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
        O = np.zeros((2,2))
        w_tot = []
        for _ in range(self.L):
            w_mag = np.array([[I , O , op],[O , O , O], [O , O , I]])
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
        O = np.zeros((2,2))
        X = np.array([[0,1],[1,0]])
        w_tot = []
        for i in range(self.L):
            if i < (self.L//2):
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
            w = np.array([[I,O,alpha*X,O],[O,O,O,O],[O,O,beta*X, gamma * X @ (np.linalg.matrix_power(X,(1-alpha)))],[O,O,O,I]])
            w_tot.append(w)
        self.w = w_tot
        return self
    
    def sigma_x_Z2_one_ladder(self, site):
        I = np.eye(2)
        O = np.zeros((2,2))
        X = np.array([[0,1],[1,0]])
        w_tot = []
        for i in range(self.L):
            if i == site-1:
                alpha = 1
            else:
                alpha = 0
            w_mag = np.array([[I, O, O, alpha*X],[O , O, O , O],[O , O , O , O], [O , O , O , I]])
            w_tot.append(w_mag)
        self.w = w_tot
        return self
    
    def sigma_x_Z2_two_ladder(self, site, ladder):
        I = np.eye(2)
        O = np.zeros((2,2))
        X = np.array([[0,1],[1,0]])
        if ladder == 1:
            X = np.kron(X,I)
        elif ladder == 2:
            X = np.kron(I,X)
        I = np.kron(I,I)
        O = np.kron(O,O)
        w_tot = []
        for i in range(self.L):
            if i == site-1:
                alpha = 1
            else:
                alpha = 0
            w_mag = np.array([[I, O , O , O , O , O , alpha*X],
                              [O, O , O , O , O , O , O],
                              [O, O , O , O , O , O , O],
                              [O, O , O , O , O , O , O],
                              [O, O , O , O , O , O , O],
                              [O, O , O , O , O , O , O],
                              [O, O , O , O , O , O , I]])
            w_tot.append(w_mag)
        self.w = w_tot
        return self
    
    def single_operator_Ising(self, site, op):
        """
        single_operator_Ising

        This function computes a local operator (op) for the 1D Ising model
        on a certain arbitrary site.

        site: int - local site where the operator acts
        op: np.ndarray - operator acting on the local site 
        
        """
        I = np.eye(2)
        O = np.zeros((2,2))
        w_tot = []
        for i in range(self.L):
            if i == site-1:
                alpha = 1
            else:
                alpha = 0
            w_mag = np.array([[I , O , alpha*op],[O , O , O],[O , O , I]])
            w_tot.append(w_mag)
        self.w = w_tot
        return self

    def envs(self, site=1, sm=False, fm=False, opt=False, ancilla=False, mixed=False, rev=False):
        """
        envs

        This function computes the left and right environments to compute the effective Hamiltonian.
        In addition, computes the environments to calculate the second and fourth moment of a mpo.

        sm: bool - Compute the left and right environments for the second moment of self.w. Default False
        fm: bool - Compute the left and right environments for the fourth moment of self.w. Default False

        """
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
        E_r = ncon([r.T,v_r.T,r.T], [[-1],[-2],[-3]])
        E_l = ncon([l,v_l,l], [[-1],[-2],[-3]])

        if opt:
            special = np.array([[1],[0]])
            E_r = ncon([special,E_r,special],[[-1,1],[1,-2,2],[-3,2]])
            special = np.array([[1,0]])
            E_l = ncon([special,E_l,special],[[1,-1],[1,-2,2],[2,-3]])

        if sm:
            a = np.array([1])
            array = self.sites
            E_l_sm = ncon(
                [a, v_l, v_l, a, array[0], self.w[0], self.w[0], array[0].conjugate()],
                [
                    [1],
                    [3],
                    [5],
                    [7],
                    [1, 2, -1],
                    [3, -2, 2, 4],
                    [5, -3, 4, 6],
                    [7, 6, -4],
                ],
            )
            if opt:
                self.env_left_sm.append(E_l_sm)
            else:
                self.env_left.append(E_l_sm)
            E_r_sm = ncon(
                [a, v_r.T, v_r.T, a, array[-1], self.w[-1], self.w[-1], array[-1].conjugate()],
                [
                    [1],
                    [3],
                    [5],
                    [7],
                    [-1, 2, 1],
                    [-2, 3, 2, 4],
                    [-3, 5, 4, 6],
                    [-4, 6, 7],
                ],
            )
            if opt:
                self.env_right_sm.append(E_r_sm)
            else:
                self.env_right.append(E_r_sm)
            for i in range(self.L - 1, site, -1):
                E_r_sm = ncon(
                    [E_r_sm, array[i - 1], self.w[i - 1], self.w[i - 1], array[i - 1].conjugate()],
                    [
                        [1, 3, 5, 7],
                        [-1, 2, 1],
                        [-2, 3, 2, 4],
                        [-3, 5, 4, 6],
                        [-4, 6, 7],
                    ],
                )
                if opt:
                    self.env_right_sm.append(E_r_sm)
                else:
                    self.env_right.append(E_r_sm)
                
        elif fm:
            a = np.array([1])
            array = self.sites
            E_l_sm = ncon(
                [a, v_l, v_l, v_l, v_l, a, array[0], self.w[0], self.w[0], self.w[0], self.w[0], array[0].conjugate()],
                [
                    [1],
                    [3],
                    [5],
                    [7],
                    [9],
                    [11],
                    [1, 2, -1],
                    [3, -2, 2, 4],
                    [5, -3, 4, 6],
                    [7, -4, 6, 8],
                    [9, -5, 8, 10],
                    [11, 10, -6],
                ],
            )
            self.env_left.append(E_l_sm)
            E_r_sm = ncon(
                [a, v_r.T, v_r.T, v_r.T, v_r.T, a, array[-1], self.w[-1], self.w[-1], self.w[-1], self.w[-1], array[-1].conjugate()],
                [
                    [1],
                    [3],
                    [5],
                    [7],
                    [9],
                    [11],
                    [-1, 2, 1],
                    [-2, 3, 2, 4],
                    [-3, 5, 4, 6],
                    [-4, 7, 6, 8],
                    [-5, 9, 8, 10],
                    [-6, 10, 11],
                ],
            )
            self.env_right.append(E_r_sm)
            for i in range(self.L - 1, site, -1):
                E_r_sm = ncon(
                    [E_r_sm, array[i - 1], self.w[i - 1], self.w[i - 1], self.w[i - 1], self.w[i - 1], array[i - 1].conjugate()],
                    [
                    [1,3,5,7,9,11],
                    [-1, 2, 1],
                    [-2, 3, 2, 4],
                    [-3, 5, 4, 6],
                    [-4, 7, 6, 8],
                    [-5, 9, 8, 10],
                    [-6, 10, 11],
                ],
                )
                self.env_right.append(E_r_sm)

        elif mixed:
            env_right = []
            env_left = []

            env_right.append(E_r)
            env_left.append(E_l)
            if rev:
                array = self.ancilla_sites
                ancilla_array = self.sites
                self.mpo_dagger()
                w = self.w_dag
            else:
                array = self.sites
                ancilla_array = self.ancilla_sites
                w = self.w

            for i in range(1, site):
                E_l = ncon(
                    [E_l, ancilla_array[i - 1], w[i - 1], array[i - 1].conjugate()],
                    [
                        [1,3,5],
                        [1,2,-1],
                        [3,-2,2,4],
                        [5,4,-3],
                    ],
                    )
                env_left.append(E_l)

            for j in range(self.L, site, -1):
                E_r = ncon(
                    [E_r, ancilla_array[j - 1], w[j - 1], array[j - 1].conjugate()],
                    [
                        [1,3,5],
                        [-1,2,1],
                        [-2,3,2,4],
                        [-3,4,5],
                    ],
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
            for i in range(1, site):
                E_l = ncon(
                    [E_l, array[i - 1], self.w[i - 1], array[i - 1].conjugate()],
                    [
                        [1,3,5],
                        [1,2,-1],
                        [3,-2,2,4],
                        [5,4,-3],
                    ],
                    )
                self.env_left.append(E_l)

            for i in range(self.L, site, -1):
                E_r = ncon(
                    [E_r, array[i - 1], self.w[i - 1], array[i - 1].conjugate()],
                    [
                        [1,3,5],
                        [-1,2,1],
                        [-2,3,2,4],
                        [-3,4,5],
                    ],
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
        H_eff_time = time.perf_counter()
        H = ncon(
            [self.env_left[-1],self.w[site - 1],self.env_right[-1]],
            [
                [-1,1,-4],
                [1,2,-2,-5],
                [-3,2,-6],
            ]
        )
        np.savetxt(f"times_data/H_eff_contraction_site_{site}_h_{self.h:.2f}", [time.perf_counter()-H_eff_time])
        # print(f"Time of H_eff contraction: {time.perf_counter()-H_eff_time}")

        reshape_time = time.perf_counter()
        H = H.reshape(
            self.env_left[-1].shape[0] * self.d * self.env_right[-1].shape[0],
            self.env_left[-1].shape[2] * self.d * self.env_right[-1].shape[2],
        )
        np.savetxt(f"times_data/H_eff_reshape_site_{site}_h_{self.h:.2f}", [time.perf_counter()-reshape_time])
        # print(f"Time of H_eff reshaping: {time.perf_counter()-reshape_time}")

        return H

    def N_eff(self, site):
        array = self.sites
        a = np.array([1])
        env = ncon([a,a,a,a],[[-1],[-2],[-3],[-4]])

        for i in range(site-1):
            ten = self.overlap_sites(array_1=array[i])
            env = ncon([env,ten],[[-1,-2,1,2],[1,2,-3,-4]])
        left = env
        left = ncon([a,a,left],[[1],[2],[1,2,-1,-2]])
        print(left.shape)
        env = ncon([a,a,a,a],[[-1],[-2],[-3],[-4]])
        for i in range(self.L-1, site-1, -1):
            ten = self.overlap_sites(array_1=array[i])
            env = ncon([ten,env],[[-1,-2,1,2],[1,2,-3,-4]])
        right = env
        right = ncon([right,a,a],[[-1,-2,1,2],[1],[2]])
        print(right.shape)
        kron = np.eye(2)
        N = ncon([left,kron,right],[[-1,-4],[-2,-5],[-3,-6]]).reshape((self.env_left[-1].shape[2]*self.d*self.env_right[-1].shape[2],self.env_left[-1].shape[2]*self.d*self.env_right[-1].shape[2]))
        return N
    
    def eigensolver(self, H_eff, site, v0=None):
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
        time_eig = time.perf_counter()
        e, v = eigsh(H_eff, k=1, which="SA", v0=v0)
        np.savetxt(f"times_data/eigsh_eigensolver_site_{site}_h_{self.h:.2f}", [time.perf_counter()-time_eig])
        # print(f"Time of eigsh during eigensolver for site {site}: {time.perf_counter()-time_eig}")
        e_min = e[0]
        eigvec = np.array(v)

        self.sites[site - 1] = eigvec.reshape(
            self.env_left[-1].shape[0], self.d, self.env_right[-1].shape[0]
        )

        return e_min

    def update_state(self, sweep, site, trunc, e_tol=10 ** (-15), precision=2):   
        """
        update_state

        This function updates the self.a and self.b lists of tensors composing
        the mps. The update depends on the sweep direction. We take the self.m
        extracted from the eigensolver and we decomposed via svd.

        sweep: string - direction of the sweeping. Could be "left" or "right"
        site: int - indicates which site the DMRG is optimizing
        trunc: bool - if True will truncate the the Schmidt values and save the
                state accordingly.
        e_tol: float - the tolerance accepted to truncate the Schmidt values
        precision: int - indicates the precision of the parameter h

        """
        if sweep == "right":
            # we want to write M (left,d,right) in LFC -> (left*d,right)
            m = self.sites[site - 1].reshape(
                self.env_left[-1].shape[2] * self.d, self.env_right[-1].shape[2]
            )
            # np.savetxt(f"site_to_update/state_to_update_{self.model}_L_{self.L}_chi_{self.chi}_site_{site}_right_sweep_n_{n}", m)
            time_svd = time.perf_counter()
            u, s, v = np.linalg.svd(m, full_matrices=False)
            np.savetxt(f"times_data/update_site_{site}_h_{self.h:.2f}", [time.perf_counter()-time_svd])
            # print(f"Time of svd during update state during sweeping {sweep} for site {site}: {time.perf_counter()-time_svd}")
            if trunc:
                condition = s >= e_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / np.linalg.norm(s_trunc)
                bond_l = u.shape[0] // self.d
                u = u.reshape(bond_l, self.d, u.shape[1])
                u = u[:, :, : len(s)]
                v = v[: len(s), :]
                if site == self.L//2:
                    # print(f'Schmidt values:\n{s}')
                    np.savetxt(
                            f"bonds_data/schmidt_values_middle_chain_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}",
                            s,
                        )
            else:
                u = u.reshape(
                    self.env_left[-1].shape[2], self.d, self.env_right[-1].shape[2]
                )
            if site == self.L//2:
                # print(f'Schmidt values:\n{s}')
                np.savetxt(
                        f"bonds_data/schmidt_values_middle_chain_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}",
                        s,
                    )
            next_site = ncon(
                [np.diag(s), v, self.sites[site]], 
                [
                    [-1,1],
                    [1,2],
                    [2,-2,-3],
                ],
            )
            self.sites[site - 1] = u
            self.sites[site] = next_site

        elif sweep == "left":
            # we want to write M (left,d,right) in RFC -> (left,d*right)
            m = self.sites[site - 1].reshape(
                self.env_left[-1].shape[2], self.d * self.env_right[-1].shape[2]
            )
            time_svd = time.perf_counter()
            u, s, v = np.linalg.svd(m, full_matrices=False)
            np.savetxt(f"times_data/update_site_{site}_h_{self.h:.2f}", [time.perf_counter()-time_svd])
            # print(f"Time of svd during update state during sweeping {sweep} for site {site}: {time.perf_counter()-time_svd}")
            if trunc:
                condition = s >= e_tol
                s_trunc = np.extract(condition, s)
                s = s_trunc / np.linalg.norm(s_trunc)
                bond_r = v.shape[1] // self.d
                v = v.reshape(v.shape[0], self.d, bond_r)
                v = v[: len(s), :, :]
                u = u[:, : len(s)]
                if site == self.L // 2:
                    # print(f"Schmidt values:\n{s}")
                    np.savetxt(
                        f"bonds_data/schmidt_values_middle_chain_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}",
                        s,
                    )
            else:
                v = v.reshape(
                    self.env_left[-1].shape[2], self.d, self.env_right[-1].shape[2]
                )

            if site == self.L//2:
                # print(f'Schmidt values:\n{s}')
                np.savetxt(
                        f"bonds_data/schmidt_values_middle_chain_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}",
                        s,
                    )

            next_site = ncon(
                [self.sites[site - 2], u, np.diag(s)], 
                [
                    [-1,-2,1],
                    [1,2],
                    [2,-3],
                ],
            )
            self.sites[site - 1] = v
            self.sites[site - 2] = next_site

        return self

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
            time_upd_env = time.perf_counter()
            array = self.sites[site - 1]
            ancilla_array = array
            if rev:
                E_l = self.env_left_sm[-1]
                array = self.ancilla_sites[site - 1]
                ancilla_array = self.sites[site - 1]
            else:
                if mixed:
                    ancilla_array = self.ancilla_sites[site - 1]
                    self.mpo_dagger()
                    self.w = self.w_dag
                E_l = self.env_left[-1]
            E_l = ncon(
                [E_l,ancilla_array,self.w[site - 1],array.conjugate()],
                [
                    [1,3,5],
                    [1,2,-1],
                    [3,-2,2,4],
                    [5,4,-3],
                ],
            )
            if rev:
                self.env_left_sm.append(E_l)
                self.env_right_sm.pop(-1)
            else:
                self.env_left.append(E_l)
                self.env_right.pop(-1)
            np.savetxt(f"times_data/update_env_{site}_h_{self.h:.2f}", [time.perf_counter()-time_upd_env])

        if sweep == "left":
            array = self.sites[site - 1]
            ancilla_array = array
            if rev:
                E_r = self.env_right_sm[-1]
                array = self.ancilla_sites[site - 1]
                ancilla_array = self.sites[site - 1]
            else:
                if mixed:
                    ancilla_array = self.ancilla_sites[site - 1]
                    self.mpo_dagger()
                    self.w = self.w_dag
                E_r = self.env_right[-1]
            E_r = ncon(
                [E_r,ancilla_array,self.w[site - 1],array.conjugate()],
                [
                    [1,3,5],
                    [-1,2,1],
                    [-2,3,2,4],
                    [-3,4,5],
                ],
            )
            if rev:
                self.env_right_sm.append(E_r)
                self.env_left_sm.pop(-1)
            else:
                self.env_right.append(E_r)
                self.env_left.pop(-1)

        return self

    def sweeping(self, trunc, e_tol=10 ** (-15), n_sweeps=2, precision=2, var=False):  # iterations, sweep,
        energies = []
        variances = []
        sweeps = ["right", "left"]
        sites = np.arange(1, self.L + 1).tolist()

        self.mpo()
        # tensor_shapes(self.w)
        env_time = time.perf_counter()
        self.envs()
        np.savetxt(f"times_data/env_h_{self.h:.2f}", [time.perf_counter()-env_time])
        # print(f"Time of env contraction: {time.perf_counter()-env_time}")
        iter = 1
        for n in range(n_sweeps):
            print(f"Sweep n: {n}\n")
            for i in range(self.L - 1):
                time_site = time.perf_counter()
                H = self.H_eff(sites[i])
                # np.savetxt(f"effective_ham/H_eff_{self.model}_L_{self.L}_h_{self.h:.2f}_chi_{self.chi}_site_{sites[i]}_sweep_n_{n}", H)
                energy = self.eigensolver(
                    H_eff=H, site=sites[i]
                )  # , v0=self.sites[sites[i]].flatten()
                energies.append(energy)
                
                # if var:
                #     sm = self.mpo_second_moment(opt=True)
                #     v = variance(first_m=energy, sm=sm)
                #     variances.append(v)
                total_state_time = time.perf_counter()
                self.update_state(sweeps[0], sites[i], trunc, e_tol, precision)
                # print(f"Total time of state updating: {time.perf_counter()-total_state_time}")
                update_env_time = time.perf_counter()
                self.update_envs(sweeps[0], sites[i])
                np.savetxt(f"times_data/update_env_h_{self.h:.2f}", [time.perf_counter()-update_env_time])
                # print(f"Time of env updating: {time.perf_counter()-update_env_time}")
                iter += 1
                # print('\n=========================================')
                # print(f"Time of site {sites[i]} optimization: {time.perf_counter()-time_site}")
                # print('=========================================\n')


            middle_chain = np.loadtxt(f"bonds_data/schmidt_values_middle_chain_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}")
            s_min = np.min(middle_chain)

            if s_min < e_tol:
                print('\n=========================================')
                print('=========================================')
                print('Optimal Schmidt values achieved, breaking the DMRG optimization algorithm\n')
                
                np.savetxt(f"energy_data/energies_sweeping_{self.model}_two_charges_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}", energies)
                if var:
                    np.savetxt(f"energy_data/variances_sweeping_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}", variances)

            # print("reversing the sweep")
            sweeps.reverse()
            sites.reverse()

        np.savetxt(f"energy_data/energies_sweeping_{self.model}_two_charges_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}", energies)
        if var:
            np.savetxt(f"energy_data/variances_sweeping_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}", variances)
        return energies

    def TEBD_ising(self, trunc, trotter_steps, delta, h_ev, J_ev, e_tol=10 ** (-15), n_sweeps=2, precision=2):
        errors = []
        self.clear_envs()
        self._random_state(seed=3, chi=self.chi, ancilla=True)
        self.canonical_form(ancilla=True)
        for i in range(trotter_steps):
            print("\n======================")
            print(f"Trotter step: {i}")
            print("======================\n")
            self, err = self.time_ev_sweeping(trunc, delta, h_ev, J_ev, e_tol=10 ** (-15), n_sweeps=2, precision=2)
            errors.append(err)
            print(f"Error at trotter step {i}: {err:.5f}")
        return self

    def time_ev_sweeping(self, trunc, delta, h_ev, J_ev, e_tol=10 ** (-15), n_sweeps=2, precision=2):
        sweeps = ["right", "left"]
        sites = np.arange(1, self.L + 1).tolist()
        errors = []
        # computation of constant error
        self.mpo_Ising_time_ev(delta, h_ev, J_ev)
        self.mpo_dagger()
        self.mpo_Ising_O_dag_O()
        self.envs(site=1, ancilla=True)
        err_const = self.braket(site=1, ancilla=True)
        self.clear_envs()

        # computation of mixed environments
        self.mpo_Ising_time_ev(delta, h_ev, J_ev)
        self.envs(site= 1, mixed=True)
        self.envs(site= 1, mixed=True, rev=True)
        iter = 1
        for n in range(n_sweeps):
            print(f"Sweep n: {n}\n")
            for i in range(self.L - 1):
                self.contraction_with_ancilla(sites[i])
                self.update_state(sweeps[0], sites[i], trunc, e_tol, precision)
                err = self.error(site=sites[i], err_const=err_const)
                print(f"error per site {sites[i]}: {err:.5f}")
                errors.append(err)
                self.update_envs(sweeps[0], sites[i], mixed=True)
                self.update_envs(sweeps[0], sites[i], rev=True)
                iter += 1

            sweeps.reverse()
            sites.reverse()

        self.ancilla_sites = self.sites
        return self, errors[-1]
 
    def contraction_with_ancilla(self, site):
        ancilla_sites = self.ancilla_sites
        new_tensor = ncon(
            [self.env_left[-1], ancilla_sites[site-1], self.w[site-1], self.env_right[-1]],
            [
                [1,2,-1],
                [1,3,5],
                [2,4,3,-2],
                [5,4,-3]
            ]
        )
        # self.sites[site - 1] = new_tensor
        return self, new_tensor

    def error(self, site, N_eff, M):
        N_eff = N_eff.reshape((self.env_left[-1].shape[2],self.d,self.env_right[-1].shape[2],self.env_left[-1].shape[2],self.d,self.env_right[-1].shape[2]))
        A_dag_N_eff_A = ncon([N_eff,self.sites[site - 1],self.sites[site - 1].conjugate()],[[1,2,3,4,5,6],[1,2,3],[4,5,6]])
        print(f"error A^dagger N_eff A: {A_dag_N_eff_A}")
        A_dag_M = ncon([self.sites[site - 1].conjugate(), M],[[1,2,3],[1,2,3]])
        print(f"error A^dagger M: {A_dag_M}")
        err = A_dag_N_eff_A - 2*A_dag_M.real
        print(f"Total error: {err}")
        return err
    
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

    def mpo_first_moment(self, site=1, ancilla=False):
        # self.order_param()
        # self.sigma_x_Z2(site=site)
        self.clear_envs()
        self.envs(site, ancilla=ancilla)
        sites = self.sites
        if ancilla:
            sites = self.ancilla_sites
        first_moment = ncon(
            [self.env_left[-1], sites[site - 1], self.w[site - 1], sites[site - 1].conjugate(), self.env_right[-1]],
            [
                [1,4,7],
                [1,3,2],
                [4,5,3,6],
                [7,6,8],
                [2,5,8]
            ]
        )
        self.clear_envs()
        return first_moment

    def mpo_second_moment(self, opt=False):
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
            sm = ncon([self.env_left_sm[0], self.env_right_sm[-1]], [[1, 2, 3, 4], [1, 2, 3, 4]])
            self.env_left_sm = []
            self.env_right_sm = []
        else:
            self.order_param()
            self.clear_envs()
            self.envs(sm=True)
            sm = ncon([self.env_left[0], self.env_right[-1]], [[1, 2, 3, 4], [1, 2, 3, 4]])

        return sm

    def mpo_fourth_moment(self):
        self.order_param()
        self.clear_envs()
        self.envs(fm=True)
        fm = ncon([self.env_left[0], self.env_right[-1]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
        return fm

    def mpo_to_mps(self):

        array = self.sites
        for i in range(self.L):
            self.sites[i] = ncon(
                [array[i], self.w[i]],
                [
                    [-1,2,-3],
                    [-2,-4,2,-5],
                ],
                ).reshape((array[i].shape[0]*self.w[i].shape[0],array[i].shape[1],array[i].shape[2]*self.w[i].shape[1]))
        return self

    def save_sites(self, precision=2):
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
            f"sites_data/shapes_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}", shapes, fmt='%1.i'#, delimiter=','
        )
        # flattening of the tensors
        tensor = [element for site in self.sites for element in site.flatten()]
        np.savetxt(
            f"sites_data/tensor_sites_{self.model}_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}", tensor
        )

    def load_sites(self, precision=2):
        """
        load_sites

        This function load the tensors into the sites of the MPS.
        We fetch a completely flat list, split it to recover the original tensors
        (but still flat) and reshape each of them accordingly with the saved shapes.
        To initially split the list in the correct index position refer to the auxiliary
        function get_labels().

        """
        # # loading of the shapes
        # shapes = np.loadtxt(
        #     f"sites_data/shapes_sites_{self.model}_two_charges_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}"
        # ).astype(int)
        # # loading of the flat tensors
        # filedata = np.loadtxt(
        #     f"sites_data/tensor_sites_{self.model}_two_charges_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}"
        # )
        # loading of the shapes
        shapes = np.loadtxt(
            f"sites_data/shapes_sites_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}"
        ).astype(int)
        # loading of the flat tensors
        filedata = np.loadtxt(
            f"sites_data/tensor_sites_L_{self.L}_chi_{self.chi}_h_{self.h:.{precision}f}"
        )
        # auxiliary function to get the indices where to split
        labels = get_labels(shapes)
        flat_tn = np.array_split(filedata, labels)
        flat_tn.pop(-1)
        # reshape the flat tensors and initializing the sites
        self.sites = [site.reshape(shapes[i]) for i, site in enumerate(flat_tn)]

        return self
