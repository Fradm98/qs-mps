import numpy as np

from scipy.sparse import csc_array, kron
import scipy.sparse as sp
from qs_mps.sparse_hamiltonians_and_operators import diagonalization, U_evolution_sparse

# Single-site identity
Id = sp.identity(3, format="csr").toarray()
O = csc_array((3, 3), dtype=complex).toarray()

# Spin operators
Sz = (1/2) * sp.diags([1, 0, -1], 0, format="csr")

S_plus  = sp.csr_matrix([[0, 0, 1],
                         [0, 0, 0],
                         [0, 0, 0]])

S_minus = sp.csr_matrix([[0, 0, 0],
                         [0, 0, 0],
                         [1, 0, 0]])

# Hole hopping operators

# hole goes into a spin up state
T_up_h   = sp.csr_matrix([[0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]])

# hole goes into a spin down state
T_down_h = sp.csr_matrix([[0, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0]])

# spin up goes into a hole state
T_h_up   = sp.csr_matrix([[0, 0, 0],
                          [1, 0, 0],
                          [0, 0, 0]])

# spin down goes into a hole state
T_h_down = sp.csr_matrix([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]])

# Hole number operator
n_h = sp.csr_matrix([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])

def kron_sparse_op(Op, i, n):
    if (i > 0) and (i < n-1):
        left = Id
        right = Id
        for k in range(i-1):
            left = kron(left, Id)
        for k in range(i+1,n-1):
            right = kron(Id, right)
        return kron(kron(left,Op),right)
    elif i == 0:
        right = Id
        for k in range(1,n-1):
            right = kron(Id, right)
        return kron(Op,right)
    elif i == n-1:
        left = Id
        for k in range(0,i-1):
            left = kron(left, Id)        
        return kron(left, Op)
    
def local_hole_occupation(n):
    return [kron_sparse_op(n_h, i, n) for i in range(n)]

def local_mag_occupation(n):
    return [kron_sparse_op(2*Sz, i, n) for i in range(n)]


# Heisenberg ham
def heis_ham(n, Jz, J_perp, eps):
    # zz-int
    H_zz = sp.csr_matrix((3**n,3**n))
    if Jz != 0:
        for i in range(n-1):
            H_zz += kron_sparse_op(Sz,i,n) @ kron_sparse_op(Sz,i+1,n)
    
    # pm-int
    H_pm = sp.csr_matrix((3**n,3**n))
    if J_perp != 0:
        for i in range(n-1):
            H_pm += kron_sparse_op(S_plus,i,n) @ kron_sparse_op(S_minus,i+1,n)
    
    # pm-int
    H_mp = sp.csr_matrix((3**n,3**n))
    if J_perp != 0:
        for i in range(n-1):
            H_mp += kron_sparse_op(S_minus,i,n) @ kron_sparse_op(S_plus,i+1,n)
    
    # breaking degeneracy (if eps<0 -> up, if eps>0 -> down)
    H_z = sp.csr_matrix((3**n,3**n))
    if eps != 0:
        for i in range(n):
            H_z += kron_sparse_op(Sz,i,n)

    return Jz * H_zz + (J_perp / 2) * (H_pm + H_mp) + eps * H_z

# hopping ham
def hop_ham(n, t_up, t_down):
    # up-hole
    H_uh = sp.csr_matrix((3**n,3**n))
    if t_up != 0:
        for i in range(n-1):
            H_uh += kron_sparse_op(T_up_h,i,n) @ kron_sparse_op(T_h_up,i+1,n)

    # hole-up
    H_hu = sp.csr_matrix((3**n,3**n))
    if t_up != 0:
        for i in range(n-1):
            H_hu += kron_sparse_op(T_h_up,i,n) @ kron_sparse_op(T_up_h,i+1,n)
    
    # down-hole
    H_dh = sp.csr_matrix((3**n,3**n))
    if t_down != 0:
        for i in range(n-1):
            H_dh += kron_sparse_op(T_down_h,i,n) @ kron_sparse_op(T_h_down,i+1,n)

    # hole-down
    H_hd = sp.csr_matrix((3**n,3**n))
    if t_down != 0:
        for i in range(n-1):
            H_hd += kron_sparse_op(T_h_down,i,n) @ kron_sparse_op(T_down_h,i+1,n)
    

    return - t_up * (H_uh + H_hu) - t_down * (H_dh + H_hd)

# hopping ham for next to nearest neighbor
def hop_2_ham(n, tp_up, tp_down):
    # up-hole
    H_uh = sp.csr_matrix((3**n,3**n))
    if tp_up != 0:
        for i in range(n-2):
            H_uh += kron_sparse_op(T_up_h,i,n) @ kron_sparse_op(T_h_up,i+2,n)

    # hole-up
    H_hu = sp.csr_matrix((3**n,3**n))
    if tp_up != 0:
        for i in range(n-2):
            H_hu += kron_sparse_op(T_h_up,i,n) @ kron_sparse_op(T_up_h,i+2,n)
    
    # down-hole
    H_dh = sp.csr_matrix((3**n,3**n))
    if tp_down != 0:
        for i in range(n-2):
            H_dh += kron_sparse_op(T_down_h,i,n) @ kron_sparse_op(T_h_down,i+2,n)

    # hole-down
    H_hd = sp.csr_matrix((3**n,3**n))
    if tp_down != 0:
        for i in range(n-2):
            H_hd += kron_sparse_op(T_h_down,i,n) @ kron_sparse_op(T_down_h,i+2,n)
    

    return - (tp_up/8) * (H_uh + H_hu) - (tp_down/8) * (H_dh + H_hd)


# holes ham
def hol_ham(n, V):
    # zz-int
    H_hh = sp.csr_matrix((3**n,3**n))
    if V != 0:
        for i in range(n-1):
            H_hh += kron_sparse_op(n_h,i,n) @ kron_sparse_op(n_h,i+1,n)
    
    return V * H_hh

def tJV_ham(n, t_up, t_down, Jz, J_perp, eps, V, tp_up=0, tp_down=0):
    H_t = hop_ham(n=n, t_up=t_up, t_down=t_down)
    H_J = heis_ham(n=n, Jz=Jz, J_perp=J_perp, eps=eps)
    H_V = hol_ham(n=n, V=V)
    H_nnn = hop_2_ham(n=n, tp_up=tp_up, tp_down=tp_down)

    return H_t + H_J + H_V + H_nnn

def half_hole_quench_init(half_chain_length, t_up, t_down, Jz, J_perp, eps, V, n_holes=1, tp_up=0, tp_down=0):
    H_tJ_0 = tJV_ham(n=half_chain_length, t_up=t_up, t_down=t_down, Jz=Jz, J_perp=J_perp, eps=eps, V=V, tp_up=tp_up, tp_down=tp_down)
    e, v = diagonalization(H_tJ_0, sparse=False)
    psi_init_side = v[:,0]

    psi_init_left = psi_init_side.copy()
    psi_init_right = psi_init_side.copy()

    psi_hole = np.array([0,1,0])
    if n_holes == 1:
        psi_init = kron(kron(psi_init_left, psi_hole), psi_init_right).T.toarray()
    if n_holes == 2:
        psi_init = kron(psi_init_left, kron(psi_hole, kron(psi_hole, psi_init_right))).T.toarray()
    psi_init = psi_init.reshape(3**(2*half_chain_length+n_holes))

    H_tJ_ev = tJV_ham(n=2*half_chain_length+n_holes, t_up=t_up, t_down=t_down, Jz=Jz, J_perp=J_perp, eps=eps, V=V, tp_up=tp_up, tp_down=tp_down)

    return H_tJ_ev, psi_init

def half_hole_quench_evolution(half_chain_length, H_tJ_ev, psi_init, trotter_steps, final_time, obs=[], n_holes=1, save=False):
    psi_ev = psi_init.copy()

    exp_vals = []
    psi_save = []

    if len(obs) == 0:
        obs = ['h_loc']
    
    if 'h_loc' in obs:
        ops_h = local_hole_occupation(2*half_chain_length + n_holes)

    if 'm_loc' in obs:
        ops_m = local_mag_occupation(2*half_chain_length + n_holes)

    occup_tot_h = []
    occup_tot_m = []

    for trott in range(trotter_steps):
        print(f"Trotter step: {trott}")
        psi_ev = U_evolution_sparse(psi_init=psi_ev, H_ev=H_tJ_ev, trotter=trotter_steps, time=final_time)
        if save:
            psi_save.append(psi_ev.copy())
        
        if 'h_loc' in obs:
            occup = []
            for op in ops_h:
                occup.append((psi_ev.conjugate().T @ op @ psi_ev).real)
                # occup.append(((psi_ev.conjugate().T @ op @ psi_ev).real).toarray())
            occup_tot_h.append(occup)
    
        if 'm_loc' in obs:
            occup = []
            for op in ops_m:
                occup.append((psi_ev.conjugate().T @ op @ psi_ev).real)
                # occup.append(((psi_ev.conjugate().T @ op @ psi_ev).real).toarray())
            occup_tot_m.append(occup)
    
    exp_vals.append(occup_tot_h)
    exp_vals.append(occup_tot_m)
    return exp_vals, psi_save