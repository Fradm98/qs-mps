import numpy as np

from scipy.optimize import curve_fit

from qs_mps.applications.Z2.utils import get_cx, get_cy, arithmetic_average
from qs_mps.utils import von_neumann_entropy


def static_potential(
    g: float,
    R: int,
    l: int,
    L: int,
    chi: int,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
):
    """
    static potential

    This function computes the static potential as the difference between the ground state energies of the two-charge sector and vacuum sector.
    The potential indicates the energy of the static pair of charges separated by R for a lattice lxL at a specific value of the coupling g
    and a certain bond dimension chi.

    g: float - value of the electric field coupling
    R: int - string length formed by the separation of two charges
    l: int - number of ladders in the direct lattice
    L: int - number of plaquettes per ladder in the direct lattice (rungs-1)
    chi: int - bond dimension used to approximate DMRG computations of the ground state
    bc: str - boundary conditions of the lattice
    sector: str - sector of the ground state
    h_i: float - starting point for computations spanning the coupling phase space
    h_f: float - ending point for computations spanning the coupling phase space
    npoints: int - number of points for computations spanning the coupling phase space
    path_tensor: str - path name for retrieving the ground state energy values
    cx: list - list of charges x-coordinates
    cy: list - list of charges y-coordinates

    """
    if cx == None:
        cx = get_cx(L,R)
    if cy == None:
        cy = get_cy(l,bc=bc,R=R)
    interval = np.linspace(h_i,h_f,npoints)
    try:
        vac = None
        energy_charges = np.load(
            f"{path_tensor}/results/energy_data/energy_Z2_dual_direct_lattice_{l}x{L}_{sector}_bc_{bc}_{cx}-{cy}_h_{h_i}-{h_f}_delta_{npoints}_chi_{chi}.npy"
        )
        energy_vacuum = np.load(
            f"{path_tensor}/results/energy_data/energy_Z2_dual_direct_lattice_{l}x{L}_vacuum_sector_bc_{bc}_{vac}-{vac}_h_{h_i}-{h_f}_delta_{npoints}_chi_{chi}.npy"
        )
    except:
        vac = np.nan
        energy_charges = np.load(
            f"{path_tensor}/results/energy_data/energy_Z2_dual_direct_lattice_{l}x{L}_{sector}_bc_{bc}_{cx}-{cy}_h_{h_i}-{h_f}_delta_{npoints}_chi_{chi}.npy"
        )
        energy_vacuum = np.load(
            f"{path_tensor}/results/energy_data/energy_Z2_dual_direct_lattice_{l}x{L}_vacuum_sector_bc_{bc}_{vac}-{vac}_h_{h_i}-{h_f}_delta_{npoints}_chi_{chi}.npy"
        )

    energy_difference = energy_charges - energy_vacuum

    for i, val in enumerate(energy_difference):
        if round(g, 3) == round(interval[i], 3):
            return val


def static_potential_chis(
    g: float,
    R: int,
    l: int,
    L: int,
    chis: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
):
    """
    static potential

    This function collects the static potentials computed for different bond dimensions chis.

    g: float - value of the electric field coupling
    R: int - string length formed by the separation of two charges
    l: int - number of ladders in the direct lattice
    L: int - number of plaquettes per ladder in the direct lattice (rungs-1)
    chis: list - bond dimensions used to approximate DMRG computations of the ground state
    bc: str - boundary conditions of the lattice
    sector: str - sector of the ground state
    h_i: float - starting point for computations spanning the coupling phase space
    h_f: float - ending point for computations spanning the coupling phase space
    npoints: int - number of points for computations spanning the coupling phase space
    path_tensor: str - path name for retrieving the ground state energy values

    """
    st_pots = []
    for chi in chis:
        st_pot = static_potential(
            g, R, l, L, chi, bc, sector, h_i, h_f, npoints, path_tensor, cx, cy
        )
        st_pots.append(st_pot)
    return st_pots


def get_exact_potential_chis(chis, potentials):
    # Given data
    x_data = chis
    y_data = potentials
    x_inv_data = [1 / chi for chi in chis]

    # Define the model function with asymptotic behavior
    def asymptotic_model(x, y0, A, B):
        exponent = np.clip(-B * x, -700, 700)
        return y0 + A * np.exp(exponent)

    # Fit the model to the data
    popt, pcov = curve_fit(
        asymptotic_model, x_data, y_data, p0=(y_data[-1], 0.1, 0.1), maxfev=1000
    )

    # Extract fitted parameters and their errors
    y0_fit, A_fit, B_fit = popt
    y0_err, A_err, B_err = np.sqrt(np.diag(pcov))
    print(f"y0 (asymptotic value in 1/chi) = {y0_fit:.6f} ± {y0_err:.6f}")
    return y0_fit, y0_err


def static_potential_exact_chi(
    g: float,
    R: int,
    l: int,
    L: int,
    chis: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
    g_thr: float = 10,
):
    potentials = static_potential_chis(
        g, R, l, L, chis, bc, sector, h_i, h_f, npoints, path_tensor, cx, cy
    )
    if g > g_thr:
        pot_exact, err = get_exact_potential_chis(chis, potentials)
    else:
        pot_exact = potentials[-1]
        err = np.abs(potentials[-1] - potentials[-2])
    return pot_exact, err


def static_potential_Ls(
    g: float,
    R: int,
    l: int,
    Ls: int,
    chis: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
):
    potentials = []
    potentials_err = []
    for L in Ls:
        pot, err = static_potential_exact_chi(
            g, R, l, L, chis, bc, sector, h_i, h_f, npoints, path_tensor, cx, cy
        )
        potentials.append(pot)
        potentials_err.append(err)
    return potentials, potentials_err


def get_exact_potential_Ls(Ls, potentials, y_errs):
    # Given data
    x_data = Ls
    y_data = potentials
    x_inv_data = [1 / L for L in Ls]
    p0 = (1, y_data[-1])

    # Define the model function with asymptotic behavior
    def asymptotic_model(x, a, b):
        return b + (a * x)

    # Fit the model to the data
    popt, pcov = curve_fit(asymptotic_model, x_inv_data, y_data, sigma=y_errs, p0=p0)
    errs = np.sqrt(np.diag(pcov))

    # Extract fitted parameters and their errors
    y0_fit = popt[1]
    y0_err = errs[1]
    print(f"y0 (asymptotic value in 1/L) = {y0_fit:.6f} ± {y0_err:.6f}")
    return y0_fit, y0_err


def static_potential_exact_L(
    g: float,
    R: int,
    l: int,
    Ls: int,
    chis: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
    r_thr: float = 4 / 5,
):
    potentials, pot_errs = static_potential_Ls(
        g, R, l, Ls, chis, bc, sector, h_i, h_f, npoints, path_tensor, cx, cy
    )
    rs = [R / L for L in Ls]
    flag = 0
    for r in rs:
        if r > r_thr:
            flag = 1

    if flag == 1:
        print(f"The ratio R/L: {r} exceeds the threshold ratio: {r_thr}\n")
        print(f"Consider taking smaller Rs, computing the potential with linear fit")
        pot_exact, err = get_exact_potential_Ls(Ls, potentials, pot_errs)
    elif flag == 0:
        print(f"Negligible boundary effects in L\n")
        print(f"Computing the potential with arithmetic average")
        pot_exact, err = arithmetic_average(potentials, pot_errs)

    return pot_exact, err


def static_potential_varying_R(
    g, Rs, l, Ls, chis, bc, sector, h_i, h_f, npoints, path_tensor, cx=None, cy=None
):
    potentials = []
    err_potentials = []
    for R in Rs:
        print(f"R: {R}")
        pot, err = static_potential_exact_L(
            g, R, l, Ls, chis, bc, sector, h_i, h_f, npoints, path_tensor, cx, cy
        )
        potentials.append(pot)
        err_potentials.append(err)

    return potentials, err_potentials


def static_potential_varying_g(
    gs, R, l, Ls, chis, bc, sector, h_i, h_f, npoints, path_tensor
):
    potentials = []
    err_potentials = []
    for g in gs:
        print(f"g: {g}")
        pot, err = static_potential_exact_L(
            g, R, l, Ls, chis, bc, sector, h_i, h_f, npoints, path_tensor
        )
        potentials.append(pot)
        err_potentials.append(err)

    return potentials, err_potentials


def potential_fit_0(R, a, b):
    return a * R + b 

def potential_fit_1(R, a, b, c):
    return a * R - b * (1 / R) + c

def potential_fit_2(R, a, b, c, d):
    return a * R - b * (1 / R) - c * (1 / (R**3)) + d

def potential_fit_3(R, a, b, c, d, e):
    return a * R - b * (1 / R) - c * (1 / (R**3)) + d * (1 / (R**5)) + e

def fitting(Rs, potentials, errors, fit=1, guess=None):
    if fit == 0:
        popt, pcov = curve_fit(potential_fit_0, Rs, potentials, sigma=errors, p0=guess)
        print()
    if fit == 1:
        popt, pcov = curve_fit(potential_fit_0, Rs, potentials, sigma=errors, p0=guess)
        guess = np.append(popt,0)
        popt, pcov = curve_fit(potential_fit_1, Rs, potentials, sigma=errors, p0=guess)
    elif fit == 2:
        popt, pcov = curve_fit(potential_fit_0, Rs, potentials, sigma=errors, p0=guess)
        guess = np.append(popt,0)
        popt, pcov = curve_fit(potential_fit_1, Rs, potentials, sigma=errors, p0=guess)
        guess = np.append(popt,0)
        popt, pcov = curve_fit(potential_fit_2, Rs, potentials, sigma=errors, p0=guess)
    elif fit == 3:
        popt, pcov = curve_fit(potential_fit_0, Rs, potentials, sigma=errors, p0=guess)
        guess = np.append(popt,0)
        popt, pcov = curve_fit(potential_fit_1, Rs, potentials, sigma=errors, p0=guess)
        guess = np.append(popt,0)
        popt, pcov = curve_fit(potential_fit_2, Rs, potentials, sigma=errors, p0=guess)
        guess = np.append(popt,0)
        popt, pcov = curve_fit(potential_fit_3, Rs, potentials, sigma=errors, p0=guess)
    errs = np.sqrt(np.diag(pcov))
    return popt, errs


def fit_correction_term_g(g, Rs, l, Ls, chis, bc, sector, h_i, h_f, npoints, path_tensor, cx=None, cy=None, fit=1, param=2, guess=None, V=False):
    """
    - to get the string tension, param = 0 for all fits
    - to get the luscher term, param = 1 for fits=1,2,3
    - to get the second order correction, param = 2 for fits=2,3
    - to get the third order correction, param = 3 for fit=3
    - to get the constant, param = 1, fit = 0 ; param = 2, fit = 1 ; param = 3, fit = 2 ; param = 4, fit = 3
    """
    pot, err = static_potential_varying_R(
        g, Rs, l, Ls, chis, bc, sector, h_i, h_f, npoints, path_tensor, cx, cy
    )
    popt, errs = fitting(Rs, pot, err, fit, guess)
    term = popt[param]
    term_err = errs[param]
    if V:
        return term, term_err, pot
    else:
        return term, term_err


def get_fit_params(gs, Rs, l, Ls, chis, bc, sector, h_i, h_f, npoints, path_tensor, cx=None, cy=None, fit=1, param=2, guess=None, V=False):
    terms = []
    term_errs = []
    pots = []
    for g in gs:
        if V:
            term, term_err, pot = fit_correction_term_g(
                g, Rs, l, Ls, chis, bc, sector, h_i, h_f, npoints, path_tensor, cx, cy, fit, param, guess, V=V
            )
            terms.append(term)
            term_errs.append(term_err)
            pots.append(pot)
        else:
            term, term_err = fit_correction_term_g(
                g, Rs, l, Ls, chis, bc, sector, h_i, h_f, npoints, path_tensor, cx, cy, fit, param, guess
            )
            terms.append(term)
            term_errs.append(term_err)
    if V:
        return terms, term_errs, pots
    else:
        return terms, term_errs


def potential_first_discrete_derivative(
    g: float,
    R: int,
    l: int,
    Ls: int,
    chis: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
    r_thr: float = 4 / 5,
    a: int = 2,
):
    pot_ex, pot_ex_err = static_potential_exact_L(g,R,l,Ls,chis,bc,sector,h_i,h_f,npoints,path_tensor,cx,cy,r_thr)
    pot_ex_minus, pot_ex_minus_err = static_potential_exact_L(g,R-a,l,Ls,chis,bc,sector,h_i,h_f,npoints,path_tensor,cx,cy,r_thr)
    first_der = (pot_ex - pot_ex_minus) / a
    first_der_err = (1 / a) * np.sqrt(pot_ex_err**2 + pot_ex_minus_err**2)
    return first_der, first_der_err

def potential_first_discrete_derivative_varying_g(
    gs: np.ndarray,
    R: int,
    l: int,
    Ls: int,
    chis: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
    r_thr: float = 4 / 5,
    a: int = 2,
):
    sigmas, sigmas_err = [], []
    for g in gs:
        sigma, sigma_err = potential_first_discrete_derivative(g,R,l,Ls,chis,bc,sector,h_i,h_f,npoints,path_tensor,cx,cy,r_thr,a)
        sigmas.append(sigma)
        sigmas_err.append(sigma_err)
    return sigmas, sigmas_err

def potential_second_discrete_derivative(
    g: float,
    R: int,
    l: int,
    Ls: int,
    chis: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
    r_thr: float = 4 / 5,
    a: int = 2,
):
    pot_ex, pot_ex_err = static_potential_exact_L(g,R,l,Ls,chis,bc,sector,h_i,h_f,npoints,path_tensor,cx,cy,r_thr)
    pot_ex_plus, pot_ex_plus_err = static_potential_exact_L(g,R+a,l,Ls,chis,bc,sector,h_i,h_f,npoints,path_tensor,cx,cy,r_thr)
    pot_ex_minus, pot_ex_minus_err = static_potential_exact_L(g,R-a,l,Ls,chis,bc,sector,h_i,h_f,npoints,path_tensor,cx,cy,r_thr)
    sec_der = (R**3) / (2 * (a**2)) * (pot_ex_plus + pot_ex_minus - 2 * pot_ex)
    sec_der_err = (R**3) / (2 * (a**2)) * np.sqrt(pot_ex_plus_err**2 + pot_ex_minus_err**2 + 4 * pot_ex_err**2)
    return sec_der, sec_der_err

def potential_second_discrete_derivative_varying_g(
    gs: np.ndarray,
    R: int,
    l: int,
    Ls: int,
    chis: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
    r_thr: float = 4 / 5,
    a: int = 2,
):
    luschers, luschers_err = [], []
    for g in gs:
        luscher, luscher_err = potential_second_discrete_derivative(g,R,l,Ls,chis,bc,sector,h_i,h_f,npoints,path_tensor,cx,cy,r_thr,a)
        luschers.append(luscher)
        luschers_err.append(luscher_err)
    return luschers, luschers_err

def connected_electric_energy_density(
    g: float,
    R: int,
    l: int,
    L: int,
    chi: int,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
):
    """
    connected electric energy density

    This function computes the electric energy density as the difference between the electric energy densities of the two-charge sector and vacuum sector.
    The list of energy densities indicates the central ladder distribution of energy for a static pair of charges separated by R for a lattice lxL at a specific value of the coupling g
    and a certain bond dimension chi.

    g: float - value of the electric field coupling
    R: int - string length formed by the separation of two charges
    l: int - number of ladders in the direct lattice
    L: int - number of plaquettes per ladder in the direct lattice (rungs-1)
    chi: int - bond dimension used to approximate DMRG computations of the ground state
    bc: str - boundary conditions of the lattice
    sector: str - sector of the ground state
    h_i: float - starting point for computations spanning the coupling phase space
    h_f: float - ending point for computations spanning the coupling phase space
    npoints: int - number of points for computations spanning the coupling phase space
    path_tensor: str - path name for retrieving the energy densities values

    """
    if cx == None:
        cx = get_cx(L, R)
    if cy == None:
        cy = get_cy(l, R=R, bc=bc)
    interval = np.linspace(h_i, h_f, npoints)
    try:
        vac = None
        energy_densities_charges = np.load(
            f"{path_tensor}/results/energy_data/electric_energy_density_Z2_dual_direct_lattice_{l}x{L}_{sector}_bc_{bc}_{cx}-{cy}_h_{h_i}-{h_f}_delta_{npoints}_chi_{chi}.npy"
        )
        energy_densities_vacuum = np.load(
            f"{path_tensor}/results/energy_data/electric_energy_density_Z2_dual_direct_lattice_{l}x{L}_vacuum_sector_bc_{bc}_{vac}-{vac}_h_{h_i}-{h_f}_delta_{npoints}_chi_{chi}.npy"
        )
    except:
        vac = np.nan
        energy_densities_charges = np.load(
            f"{path_tensor}/results/energy_data/electric_energy_density_Z2_dual_direct_lattice_{l}x{L}_{sector}_bc_{bc}_{cx}-{cy}_h_{h_i}-{h_f}_delta_{npoints}_chi_{chi}.npy"
        )
        energy_densities_vacuum = np.load(
            f"{path_tensor}/results/energy_data/electric_energy_density_Z2_dual_direct_lattice_{l}x{L}_vacuum_sector_bc_{bc}_{vac}-{vac}_h_{h_i}-{h_f}_delta_{npoints}_chi_{chi}.npy"
        )

    energy_density_difference = energy_densities_charges - energy_densities_vacuum

    for i, val in enumerate(energy_density_difference):
        if round(g, 3) == round(interval[i], 3):
            return val


def string_width_electric_energy_density(
    g: float,
    R: int,
    l: int,
    L: int,
    chi: int,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
):
    """
    string width electric energy density

    This function computes the string width by taking the connected electric energy density of a plaquette and weighting it for the distance
    of the plaquette to the axis of the string (fluxtube). This value is relative to a specific electric coupling g.

    eed_conn_lad: numpy.ndarray - array of connected energy densities for a ladder in the middle of the string

    """
    if cx == None:
        cx = get_cx(L, R)
    if cy == None:
        cy = get_cy(l, R=R, bc=bc)

    eed_conn_lad = connected_electric_energy_density(
        g, R, l, L, chi, bc, sector, h_i, h_f, npoints, path_tensor, cx, cy
    )
    l = len(eed_conn_lad)
    if bc == "obc":
        x0 = cy[0]
        xs = [
            i
            for i in range(
                -x0,
                (l - x0) + 1,
            )
            if i != 0
        ]

    # correctly "translate" the coordinates
    elif bc == "pbc":
        x0 = l // 2
        xs = [i for i in range(-x0,(l - x0) + 1,)if i != 0]
        xs = [xs[i - (l - (l // 2))] for i in range(len(xs))]

    eed_sum_lad = 0
    for x, eed_x in zip(xs, eed_conn_lad):
        eed_sum_lad += eed_x * ((x) ** 2)
    eed_sum_lad = eed_sum_lad / sum(eed_conn_lad)
    return eed_sum_lad


def string_width_chis(
    g: float,
    R: int,
    l: int,
    L: int,
    chis: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    cx: list = None,
    cy: list = None,
):
    """
    static potential

    This function collects the electric string width computed for different bond dimensions chis.

    g: float - value of the electric field coupling
    R: int - string length formed by the separation of two charges
    l: int - number of ladders in the direct lattice
    L: int - number of plaquettes per ladder in the direct lattice (rungs-1)
    chis: list - bond dimensions used to approximate DMRG computations of the ground state
    bc: str - boundary conditions of the lattice
    sector: str - sector of the ground state
    h_i: float - starting point for computations spanning the coupling phase space
    h_f: float - ending point for computations spanning the coupling phase space
    npoints: int - number of points for computations spanning the coupling phase space
    path_tensor: str - path name for retrieving the energy density values

    """
    ws_chi = []
    for chi in chis:
        w = string_width_electric_energy_density(
            g, R, l, L, chi, bc, sector, h_i, h_f, npoints, path_tensor, cx, cy
        )
        ws_chi.append(w)
    return ws_chi


def get_exact_string_chis(chis, strings):
    # Given data
    x_data = chis
    y_data = strings
    x_inv_data = [1 / chi for chi in chis]

    # Define the model function with asymptotic behavior
    def asymptotic_model(x, a, b, c):
        return c + a * x**b

    # Fit the model to the data
    popt, pcov = curve_fit(
        asymptotic_model, x_inv_data, y_data, p0=(-1, 2, y_data[-1]), maxfev=1000
    )

    # Extract fitted parameters and their errors
    a_fit, b_fit, c_fit = popt
    a_err, b_err, c_err = np.sqrt(np.diag(pcov))
    print(f"y0 (asymptotic value in 1/chi) = {c_fit:.6f} ± {c_err:.6f}")
    return c_fit, c_err


def string_width_exact_chi(
    g: float,
    R: int,
    l: int,
    L: int,
    chis: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
    g_thr: float = 1,
):
    strings = string_width_chis(
        g, R, l, L, chis, bc, sector, h_i, h_f, npoints, path_tensor
    )
    if g > g_thr:
        str_exact, err = get_exact_string_chis(chis, strings)
    else:
        str_exact = strings[-1]
        err = np.abs(strings[-1] - strings[-2])
    return str_exact, err


def string_width_varying_g(
    gs, R, l, L, chis, bc, sector, h_i, h_f, npoints, path_tensor
):
    strings = []
    err_strings = []
    for g in gs:
        print(f"g: {g}")
        string, err = string_width_exact_chi(
            g, R, l, L, chis, bc, sector, h_i, h_f, npoints, path_tensor
        )
        strings.append(string)
        err_strings.append(err)

    return strings, err_strings


def entropy(
    R: int,
    l: int,
    L: int,
    chi: list,
    bc: str = None,
    sector: str = None,
    h_i: float = None,
    h_f: float = None,
    npoints: int = None,
    path_tensor: str = None,
):
    """
    entropy

    This function computes the entropies for some couplings gs.

    R: int - string length formed by the separation of two charges
    l: int - number of ladders in the direct lattice
    L: int - number of plaquettes per ladder in the direct lattice (rungs-1)
    chi: int - bond dimension used to approximate DMRG computations of the ground state
    bc: str - boundary conditions of the lattice
    sector: str - sector of the ground state
    h_i: float - starting point for computations spanning the coupling phase space
    h_f: float - ending point for computations spanning the coupling phase space
    npoints: int - number of points for computations spanning the coupling phase space
    path_tensor: str - path name for retrieving the energy density values

    """
    cx = get_cx(L, R)
    cy = get_cy(l, R=R, bc=bc)

    try:
        vac = None
        schmidt_values = np.load(
            f"{path_tensor}/results/entropy_data/{L//2}_schmidt_vals_Z2_dual_direct_lattice_{l}x{L}_{sector}_bc_{bc}_{cx}-{cy}_h_{h_i}-{h_f}_delta_{npoints}_chi_{chi}.npy"
        )
    except:
        vac = np.nan
        schmidt_values = np.load(
            f"{path_tensor}/results/entropy_data/{L//2}_schmidt_vals_Z2_dual_direct_lattice_{l}x{L}_{sector}_bc_{bc}_{cx}-{cy}_h_{h_i}-{h_f}_delta_{npoints}_chi_{chi}.npy"
        )

    entropies = []
    for sm in schmidt_values:
        entropies.append(von_neumann_entropy(sm))

    return entropies
