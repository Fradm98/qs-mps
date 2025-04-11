import numpy as np
from scipy.optimize import curve_fit

from typing import Literal, Union

import matplotlib.pyplot as plt
from qs_mps.mps_class import MPS as mps
import torch

def get_cx(L: int, R: int):
    assert 0 <= R < L, f"The fluxtube spans for {R} lattice links but the lattice length is {L}"
    if R == 0:
        return np.nan
    else:
        return [int(L/2-R/2),int(L/2+R/2)]

def get_cy(l: int, bc: str="obc", R: int=0):
    if R == 0:
        return np.nan
    else:
        if bc == "obc":
            return [l//2,l//2]
        elif bc == "pbc":
            return [0,0]

def find_closest_value(interval, g):
    """
    Finds the closest value in the interval to the g.
    If the g is between two values, returns the smaller one.

    Parameters:
    interval (list of numbers): The sorted discrete interval values.
    g (number): The value to find the closest match for.

    Returns:
    number: The closest value in the interval to the g.
    """
    # Sort the interval in ascending order to simplify comparisons
    interval = sorted(interval)

    # Edge cases for targets outside the interval bounds
    if g <= interval[0]:
        print(f"we search for g={interval[0]}")
        return interval[0], 0
    if g >= interval[-1]:
        print(f"we search for g={interval[-1]}")
        return interval[-1], len(interval)-1

    # Initialize closest variable to hold the result
    closest = interval[0]
    closest_index = 0
    # Traverse the interval to find the closest value
    for index, value in enumerate(interval):
        # If the current value is closer to the g, update closest
        if abs(value - g) < abs(closest - g):
            closest = value
            closest_index = index
        # If the distance is the same but g is between values, choose the smaller one
        elif abs(value - g) == abs(closest - g) and value < closest:
            closest = value
            closest_index = index

    print(f"we search for g={closest}")
    return closest, closest_index


def weighted_average(data: list, err_data: list):
    """
    weighted average

    This function takes the average of some data weighted for their standard deviations.
    In particular, we can use it when the r=R/L between the string and lattice length
    is below a certain r threshold of 4/5.

    data: list - e.g. list of static potential values
    err_data: list - e.g. list of static potential error values after getting rid of the bond dimension

    """
    weights = 1 / np.asarray(err_data) ** 2
    av = np.sum(np.asarray(data) * weights) / np.sum(weights)

    # Error in the weighted average
    av_err = np.sqrt(1 / np.sum(weights))

    print(f"Weighted Average: {av}")
    print(f"Error in the Average: {av_err}")
    return av, av_err


def arithmetic_average(data: list, err_data: list):
    """
    arithmetic average

    This function takes the average of some data and propagates the error for their standard deviations.
    In particular, we can use it when the r=R/L between the string and lattice length
    is below a certain r threshold of 4/5.

    data: list - e.g. list of static potential values
    err_data: list - e.g. list of static potential error values after getting rid of the bond dimension

    """
    weights = np.asarray(err_data) ** 2
    av = np.sum(data) / len(data)

    # Error propagation in the arithmetic average
    av_err = np.sqrt(np.sum(weights)) / len(weights)

    # print(f"Arithmetic Average: {av}")
    # print(f"Error propagation in the Average: {av_err}")
    return av, av_err


def asymptotic_fit(
    y_data: np.ndarray,
    x_data: Union[np.ndarray, list],
    x_label: str,
    y_err: np.ndarray = None,
    fit_func: Literal["exp", "lin"] = "exp",
    bounds: tuple = None,
):
    if fit_func == "exp":
        # Define the model function with asymptotic behavior
        def asymptotic_model(x, a, b, c):
            return c + a * np.exp(-b * x)

        p0 = (0.1, 0.1, y_data[-1])
        # bounds = (0, [np.inf, np.inf, y_data[-1]])
        x_inv_data = x_data
    elif fit_func == "lin":
        # Define the model function with asymptotic behavior
        def asymptotic_model(x, a, b):
            return b + (a * x)

        p0 = (1, y_data[-1])
        x_inv_data = [1 / x for x in x_data]
        # bounds = (0, [np.inf, y_data[-1]])
    else:
        raise TypeError(
            "The fit you chose is not available. 'exp' and 'lin' fits are implemented"
        )

    # Fit the model to the data
    popt, pcov = curve_fit(
        asymptotic_model, x_inv_data, y_data, sigma=y_err, p0=p0, maxfev=1000
    )
    errs = np.sqrt(np.diag(pcov))
    print(f"Fitted {fit_func} observable in function of {x_label}:")
    return popt, errs


def plot_asymptotic_fit(
    y_data: np.ndarray,
    x_data: Union[np.ndarray, list],
    x_label: str,
    popt: list,
    errs: np.ndarray,
    y_err: np.ndarray = None,
    fit_func: Literal["exp", "lin"] = "exp",
    fixed_params: list = None,
):
    g, R, l, L = fixed_params[0], fixed_params[1], fixed_params[2], fixed_params[3]
    x_inv_data = [1 / x for x in x_data]

    # Plot the data and the fit with respect to 1/x
    plt.figure(figsize=(8, 6))

    if x_label == "L":
        last_fixed_var = "$"
    elif x_label == "chi":
        x_label = "\\chi"
        last_fixed_var = f", l$x$L={l}$x${L}$"

    if fit_func == "exp":
        # Define the model function with asymptotic behavior
        def asymptotic_model(x, a, b, c):
            return c + a * np.exp(-b * x)

        # Assign label for the model function of the asymptotic behavior
        fit_label = (
            f"Fit: $y = {popt[2]:.2f} + {popt[0]:.2f} e^{{-{popt[1]:.2f} {x_label}}}$"
        )
        asymptotic_val = popt[2]
        err_val = errs[2]
        # Generate data for the fitted curve
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = asymptotic_model(x_fit, *popt)
        plt.plot(
            1 / x_fit, y_fit, linewidth=1, linestyle="--", color="red", label=fit_label
        )

    elif fit_func == "lin":
        # Define the model function with asymptotic behavior
        def asymptotic_model(x, a, b):
            return b + (a * x)

        # Assign label for the model function of the asymptotic behavior
        fit_label = f"Fit: $y = {popt[1]:.2f} + {popt[0]:.2f} {x_label}$"
        asymptotic_val = popt[1]
        err_val = errs[1]
        # Generate data for the fitted curve
        x_fit = np.linspace(0, max(x_inv_data), 100)
        y_fit = asymptotic_model(x_fit, *popt)
        plt.plot(
            x_fit, y_fit, linewidth=1, linestyle="--", color="red", label=fit_label
        )

    else:
        raise TypeError(
            "The fit you chose is not available. 'exp' and 'lin' fits are implemented"
        )

    plt.errorbar(
        x_inv_data,
        y_data,
        yerr=y_err,
        fmt="x",
        capsize=7,
        label=f"$y(g={round(g,2)}, R={R}{last_fixed_var})",
    )
    plt.text(
        x=x_inv_data[-1],
        y=y_data[-1] + (y_data[-2] - y_data[-1]) / 2,
        s=f"${x_label}: {x_data[-1]}$",
        bbox=dict(
            facecolor="lightblue",
            edgecolor="black",
            boxstyle="round,pad=0.5",
            alpha=0.7,
        ),
    )

    # plt.plot(x_fit, y_fit, linewidth=1, linestyle="--", color="red", label=fit_label)

    # Plot the asymptotic value at 1/x = 0 with error bar
    plt.errorbar(
        0,
        asymptotic_val,
        yerr=err_val,
        fmt="o",
        color="black",
        capsize=7,
        label="Asymptotic Value $y_0$",
    )

    # Customize the plot
    plt.xlabel(f"$1/{x_label}$")
    plt.ylabel("Relevan Observable $(y)$")
    plt.title(f"Asymptotic Fit $vs$ $1/{x_label}$")
    plt.legend()
    plt.grid(True)

    plt.show()


def get_rdm_qs_mps(psi: mps, *, sites, d=2) -> np.ndarray:
    """Obtain the reduced density matrix from the input MPS
    on the selected sites."""
    # get_rho_segment assumes the sites are sorted.
    sites = np.sort(sites)
    state = mps(L=len(psi), d=d, model="Z2_dual")
    state.sites = psi
    rdm = state.reduced_density_matrix_debug(sites)
    sz = int(np.sqrt(np.product(rdm.shape)))
    return np.reshape(rdm, (sz, sz))

def projh_psd(m) -> np.ndarray:
    """Project a given hermitian matrix into the cone
    of positive semidefinite and hermitian matrices."""
    d, u = np.linalg.eigh(m)
    d = np.where(d >= 0, d, 0)
    return u @ np.diag(d) @ np.conj(u).T

def gstates_to_rdms_matrix_qs_mps(gstates, *, sites=None, shape=None, proj_psd=False, d=2):
    """Given a list of ground states (qs-mps MPS) ordered corresponding
    to a flattened lattice of ground states (row-major ordering),
    obtain a matrix of RDMs."""
    if shape is None:
        shape = (int(np.sqrt(len(gstates))), ) * 2
    if sites is None:
        # Default to the middle site.
        sites = (len(gstates[0]) // 2, )
    rdms = [get_rdm_qs_mps(psi, sites=sites, d=d) for psi in gstates]
    if proj_psd:
        # Project on PSD cone to correct minor numerical errors
        # that induce small negative eigenvalues.
        rdms = [projh_psd(rdm) for rdm in rdms]
    rdms = np.array(rdms)
    rdms = rdms.reshape(shape + rdms.shape[1:])
    return rdms


###########################################


def mps_overlap(bs, *, squeeze_end=True):
    t = torch.tensor(bs[0])
    t = torch.einsum("ipa,jpb->ijab", t, torch.conj(t))
    t = torch.squeeze(torch.squeeze(torch.tensor(t), dim=0), dim=0)

    for b in bs[1:]:

        b = torch.tensor(b)
        t = torch.einsum("ij,ipa,jpb->ab", t, b, torch.conj(b))

    if squeeze_end:
        t = torch.squeeze(torch.squeeze(t, dim=0), dim=0)
    return t


def mps_overlap_b(bs, *, squeeze_end=True):
    t = torch.tensor(bs[0])
    t = torch.einsum("ipa,jpb->ijab", t, torch.conj(t))
    t = torch.squeeze(torch.squeeze(torch.tensor(t), dim=-1), dim=-1)

    for b in bs[1:]:

        b = torch.tensor(b)
        t = torch.einsum("ij,api,bpj->ab", t, b, torch.conj(b))
    if squeeze_end:
        t = torch.squeeze(torch.squeeze(t, dim=-1), dim=-1)
    return t


def mps_contract_open(bs, *, squeeze_ends=True):
    t = torch.tensor(bs[0])
    for b in bs[1:]:

        t = torch.einsum("ijk,kab->ijab", t, torch.tensor(b))
        # Merge the physical legs
        t = torch.flatten(t, start_dim=1, end_dim=2)
    if squeeze_ends:
        # Squeeze first and last legs since we have open boundary conditions.
        t = torch.squeeze(torch.squeeze(t, dim=0), dim=-1)
    return t


def single_rdm_n(mps, sites, L):
    bs = mps
    top = mps_overlap(bs[: sites[0]], squeeze_end=False)

    # print("Top")
    # for idx, tensor in enumerate(bs[: self.sites[0]]):
    #     print("Number: ", idx, "Shape: ", np.shape(tensor))

    # print("Bottom")
    # for idx, tensor in enumerate(bs[self.sites[-1] :]):
    #     print("Number: ", idx, "Shape: ", np.shape(tensor))

    bottom = mps_overlap_b(bs[sites[-1] :][::-1], squeeze_end=False)

    # RDM for the middle 2 sites
    t = mps_contract_open(bs[sites[0] : sites[-1]], squeeze_ends=False)
    t = np.einsum("apb,iqj->aipqbj", t, np.conj(t))

    # Put all together and obtain a RDM
    t = np.einsum("ai,aipqbj,bj->pq", top, t, bottom)
    return t


def partial_rdm_n(mps_list, sites, L) -> np.array:
    rdm_list = [single_rdm_n(mps_list[i], sites, L) for i in range(len(mps_list))]
    return np.array(rdm_list)