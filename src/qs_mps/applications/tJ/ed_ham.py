import numpy as np

from scipy.sparse import csc_array, kron
import scipy.sparse as sp
from scipy.linalg import svd 

from qs_mps.sparse_hamiltonians_and_operators import diagonalization, U_evolution_sparse
from ncon import ncon

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
    
def neel_prod_state(n):
    spin_up_tn = np.array([1,0,0]).reshape((1,3,1))
    hole_tn = np.array([0,1,0]).reshape((1,3,1))
    spin_down_tn = np.array([0,0,1]).reshape((1,3,1))
    tn_list = [spin_up_tn if (i%2) == 0 else spin_down_tn for i in range(n)]
    return tn_list

def local_hole_occupation(n):
    return [kron_sparse_op(n_h, i, n) for i in range(n)]

def local_mag_occupation(n):
    return [kron_sparse_op(2*Sz, i, n) for i in range(n)]


def id_gate(d,chi):
    I = np.zeros((chi, chi, d, d))

    for a in range(chi):
        for s in range(d):
            I[a,a,s,s] = 1.0
    return I


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

def U_i_ip1_tJV(Jz, t_up, t_down, J_perp, V, delta):
    """
    U_i_ip1

    This function computes the exponential of the 2-site hamiltonian for the t-J model.
    It returns to versions: 
    1. one with a time step delta/2 to use at the initial and final step
    of the trotterization
    2. one with a time step delta to use in the bulk steps of the trotterization

    """

    # choose the hamiltonian parameters
    H_i_ip1 = (Jz * kron(Sz, Sz) 
    + (J_perp/2) * kron(S_plus, S_minus) 
    + (J_perp/2) * kron(S_minus, S_plus) 
    - (t_up) * kron(T_up_h, T_h_up) 
    - (t_up) * kron(T_h_up, T_up_h) 
    - (t_down) * kron(T_down_h, T_h_down) 
    - (t_down) * kron(T_h_down, T_down_h)
    + (V) * kron(n_h, n_h)).toarray()

    # initial and final 2-site evolution operator
    op_ev_if = sp.linalg.expm(-1j * delta/2 * H_i_ip1)

    # bulk 2-site evolution operator
    op_ev_bulk = sp.linalg.expm(-1j * delta * H_i_ip1)
    
    return op_ev_if, op_ev_bulk

def evolution_mpo_step_i_ip1_tJV(n, site_i_if, site_ip1_if, site_i_b, site_ip1_b):
    """
    evolution_mpo_start_tJ

    This function finds the starting evolution mpo for a chain n of the t-J model.
    It is the first step of a second order trotterization.

    """
    # first site has one tensor only
    mpo_id = Id.reshape((1,1,3,3))
    mpo_start = ncon([site_i_if, mpo_id, site_i_if],[[-1,-4,-7,1],[-2,-5,1,2],[-3,-6,2,-8]]).reshape((site_i_if.shape[0]**2,site_i_if.shape[1]**2, 3, 3))
    # two bulk operators for even and odd sites
    mpo_bulk_odd = ncon([site_ip1_if, site_i_b, site_ip1_if],[[-1,-4,-7,1],[-2,-5,1,2],[-3,-6,2,-8]]).reshape((site_ip1_if.shape[0]**2*site_i_b.shape[0],site_ip1_if.shape[1]**2*site_i_b.shape[1], 3, 3))
    mpo_bulk_even = ncon([site_i_if, site_ip1_b, site_i_if],[[-1,-4,-7,1],[-2,-5,1,2],[-3,-6,2,-8]]).reshape((site_i_if.shape[0]**2*site_ip1_b.shape[0],site_i_if.shape[1]**2*site_ip1_b.shape[1], 3, 3))
    # two last-site operators depending on the parity of the chain
    mpo_end_n_odd = ncon([mpo_id, site_ip1_b, mpo_id],[[-1,-4,-7,1],[-2,-5,1,2],[-3,-6,2,-8]]).reshape((mpo_id.shape[0]**2*site_ip1_b.shape[0],mpo_id.shape[1]**2*site_ip1_b.shape[1], 3, 3)) # for odd chains
    mpo_end_n_even = ncon([site_ip1_if, mpo_id, site_ip1_if],[[-1,-4,-7,1],[-2,-5,1,2],[-3,-6,2,-8]]).reshape((site_ip1_if.shape[0]**2*mpo_id.shape[0],site_ip1_if.shape[1]**2*mpo_id.shape[1], 3, 3)) # for even chains
 
    mpo_step = []
    
    # left op
    mpo_step.append(mpo_start)
    
    # bulk ops
    for i in range(1,n-1):
        if (i%2) == 0:
            mpo_step.append(mpo_bulk_even)
        elif (i%2) == 1:
            mpo_step.append(mpo_bulk_odd)
    
    # right op
    if (n%2) == 0:
        mpo_step.append(mpo_end_n_even)
    elif (n%2) == 1:
        mpo_step.append(mpo_end_n_odd)
    
    return mpo_step

## helping functions
def U_i_ip2_tJ(tp_up, tp_down, delta):
    """
    U_i_ip2

    This function computes the exponential of the 2-site hamiltonian for the t-J model.
    It returns to versions: 
    1. one with a time step delta/2 to use at the initial and final step
    of the trotterization
    2. one with a time step delta to use in the bulk steps of the trotterization

    """

    # choose the hamiltonian parameters
    H_i_ip2 = (
    - (tp_up/8) * kron(T_up_h, T_h_up) 
    - (tp_up/8) * kron(T_h_up, T_up_h)
    - (tp_down/8) * kron(T_down_h, T_h_down) 
    - (tp_down/8) * kron(T_h_down, T_down_h)).toarray()

    # initial and final 2-site evolution operator
    op_ev_if = sp.linalg.expm(-1j * delta/2 * H_i_ip2)

    # bulk 2-site evolution operator
    op_ev_bulk = sp.linalg.expm(-1j * delta * H_i_ip2)
    
    return op_ev_if, op_ev_bulk

def evolution_mpo_svd_1_tJ(op_ev: np.ndarray, d: int=3, schmidt_tol: float=1e-15, trunc: bool=False):
    """
    evolution_mpo_svd

    This function takes the edges, and bulk 2-site evolution operators (of the t-J model) and performs an svd
    to separate the matrix into site i and site i+1. Reshaping the results of the svd
    we can obtain the mpo for those evolution operators (with bounded bond dimension D<=d^2)

    """
    op_ev = op_ev.reshape(d,d,d,d)
    op_ev = op_ev.transpose(0,2,1,3)
    op_ev = op_ev.reshape(d*d,d*d)

    u, s, v = svd(op_ev, full_matrices=False)

    if trunc:
        condition = s >= schmidt_tol
        s_trunc = np.extract(condition, s)
        s = s_trunc
        v = v[:len(s),:]

    site_i = u.reshape(d,d,u.shape[1])
    site_i = site_i[:, :, :len(s)]
    site_i = site_i.transpose(2,0,1)
    site_i = site_i.reshape(1,len(s),d,d)

    site_ip1 = ncon([np.diag(s), v],[[-1, 1],[1, -2]]).reshape(v.shape[0],d,d)
    site_ip1 = site_ip1.reshape(1,v.shape[0],d,d)
    site_ip1 = site_ip1.transpose(1,0,2,3)


    tol = 1e-15 * np.max(np.abs(site_i))
    site_i.real[np.abs(site_i.real) < tol] = 0
    site_i.imag[np.abs(site_i.imag) < tol] = 0
    
    tol = 1e-15 * np.max(np.abs(site_ip1))
    site_ip1.real[np.abs(site_ip1.real) < tol] = 0
    site_ip1.imag[np.abs(site_ip1.imag) < tol] = 0
    
    return site_i, site_ip1

## helping functions
def make_trott_mpo_eo_oe(n, op_l, op_r, parity: str="eo", d: int=3):
    if parity == "eo":
        mpo_even = ncon([op_l,op_r],[[-1,-3,-5,1],[-2,-4,1,-6]]).reshape((op_l.shape[0]*op_r.shape[0],op_l.shape[1]*op_r.shape[1],d,d))
        mpo_odd = ncon([op_r,op_l],[[-1,-3,-5,1],[-2,-4,1,-6]]).reshape((op_l.shape[0]*op_r.shape[0],op_l.shape[1]*op_r.shape[1],d,d))
    elif parity == "oe":
        mpo_even = ncon([op_r,op_l],[[-1,-3,-5,1],[-2,-4,1,-6]]).reshape((op_l.shape[0]*op_r.shape[0],op_l.shape[1]*op_r.shape[1],d,d))
        mpo_odd = ncon([op_l,op_r],[[-1,-3,-5,1],[-2,-4,1,-6]]).reshape((op_l.shape[0]*op_r.shape[0],op_l.shape[1]*op_r.shape[1],d,d))

    mpo_trott = []

    mpo_trott.append(op_l)
    for i in range(1,n-1):
        if (i%2) == 0:
            mpo_trott.append(mpo_even)
        elif (i%2) == 1:
            mpo_trott.append(mpo_odd)
    
    mpo_trott.append(op_r)
    return mpo_trott

def mpo_ev_trotter_i_ip1_pipeline(n, Jz, J_perp, t_up, t_down, V, delta, trunc=False):
    op_ev_delta_half, op_ev_delta = U_i_ip1_tJV(Jz, t_up, t_down, J_perp, V, delta)
    site_i_delta_half, site_ip1_delta_half = evolution_mpo_svd_1_tJ(op_ev_delta_half,trunc=trunc)
    mpo_eo = make_trott_mpo_eo_oe(n, site_i_delta_half, site_ip1_delta_half, parity="eo")
    mpo_oe = make_trott_mpo_eo_oe(n, site_i_delta_half, site_ip1_delta_half, parity="oe")
    return mpo_eo, mpo_oe

def mpo_ev_trotter_i_ip1_pipeline_alone(n, Jz, J_perp, t_up, t_down, V, delta, trunc=False):
    op_ev_delta_half, op_ev_delta = U_i_ip1_tJV(Jz, t_up, t_down, J_perp, V, delta)
    site_i_delta_half, site_ip1_delta_half = evolution_mpo_svd_1_tJ(op_ev_delta_half,trunc=trunc)
    site_i_delta, site_ip1_delta = evolution_mpo_svd_1_tJ(op_ev_delta,trunc=trunc)
    mpo_ev_trotter_i_ip1 = evolution_mpo_step_i_ip1_tJV(n, site_i_delta_half, site_ip1_delta_half, site_i_delta, site_ip1_delta)
    return mpo_ev_trotter_i_ip1


## helping functions
def make_mask_even(n):
    mask_id_A = [(i%4 == 1) if (i+1)<n else False for i in range(n)]
    mask_C_l = [(i%4 == 1) if (i+2)<n else False for i in range(n)]
    return mask_id_A, mask_C_l

def make_mask_odd(n):
    mask_id_B = [(i%4 == 3) if (i+1)<n else False for i in range(n)]
    mask_D_l = [(i%4 == 3) if (i+2)<n else False for i in range(n)]
    return mask_id_B, mask_D_l

def make_trott_mpo(n, op_l, i_en, op_r, id, parity: str="even", d: int=3):
    if parity == "even":
        mask_1, mask_2 = make_mask_even(n)
    elif parity == "odd":
        mask_1, mask_2 = make_mask_odd(n)

    mpo_trott = [id] * n

    idx = 0
    for mask_step_1, mask_step_2 in zip(mask_1, mask_2):
        if mask_step_1:
            mpo_trott[idx] = i_en.copy()
            mpo_trott[idx-1] = op_l.copy()
            mpo_trott[idx+1] = op_r.copy()
        if mask_step_2:
            mpo_trott[idx] = ncon([mpo_trott[idx],op_l],[[-1,-3,-5,1],[-2,-4,1,-6]]).reshape((mpo_trott[idx].shape[0],mpo_trott[idx].shape[1]*op_l.shape[1],d,d))
            mpo_trott[idx+1] = ncon([mpo_trott[idx+1],i_en],[[-1,-3,-5,1],[-2,-4,1,-6]]).reshape((mpo_trott[idx+1].shape[0]*i_en.shape[0],i_en.shape[1],d,d))
            mpo_trott[idx+2] = op_r.copy()
        idx += 1
    return mpo_trott

def mpo_ev_trotter_i_ip2_pipeline(n, tp_up, tp_down, delta, trunc=False):
    op_ev_half, op_ev_delta = U_i_ip2_tJ(tp_up, tp_down, delta)
    site_i_delta_half, site_ip2_delta_half = evolution_mpo_svd_1_tJ(op_ev_half,trunc=trunc)
    site_i_delta, site_ip2_delta = evolution_mpo_svd_1_tJ(op_ev_delta,trunc=trunc)
    id_enlarged_mpo = id_gate(3,site_i_delta_half.shape[1])
    id_mpo = sp.identity(n=3).toarray().reshape((1,1,3,3))
    trotter_step_delta_half = make_trott_mpo(n, site_i_delta_half, id_enlarged_mpo, site_ip2_delta_half, id_mpo, parity="even")
    trotter_step_delta = make_trott_mpo(n, site_i_delta, id_enlarged_mpo, site_ip2_delta, id_mpo, parity="odd")
    return trotter_step_delta_half, trotter_step_delta