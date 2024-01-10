from qs_mps.mps_class import MPS
import time
import numpy as np

# simulate different chi
L = 50
d = 2
model = "Ising"
chis = [16, 32, 64, 128]  # ,8,16,32,64,128,256,512
h = 1
J = 1
eps = 0
path = "/data/fdimarca/projects/0_ISING/results/time_data/"

time_tot = []
for chi in chis:
    print(f"start DMRG for L={L}, chi={chi}")
    time_st = time.perf_counter()
    chain = MPS(L=L, d=d, model=model, chi=chi, h=h, eps=eps, J=J)
    chain._random_state(seed=3, chi=chi)
    energy, entropy = chain.DMRG(trunc_chi=True, trunc_tol=False, n_sweeps=2)
    time_end = abs(time.perf_counter() - time_st)
    print(f"Time of computation: {time_end} sec")
    time_tot.append(time_end)

np.savetxt(f"{path}times_dmrg_L_{L}_chis_2-4-8-16-32-64-128", time_tot)

# load time

time_tot = np.loadtxt(f"{path}times_dmrg_L_{L}_chis_2-4-8-16-32-64-128")

# default parameters of the plot layout
import matplotlib.pyplot as plt
import matplotlib as mpl
from qs_mps.utils import fitting

plt.rcParams["text.usetex"] = True  # use latex
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def exponential_function(x, a, b, c):
    """
    Exponential function: a * exp(b * x) + c
    """
    return a * np.exp(x * b) + c


def poly_function(x, a, b, c):
    """
    Poly function: a * x ** b + c
    """
    return a * x**b + c


def linear_function(x, a, b):
    """
    Linear function: a * x + b
    """
    return a * x + b


def fit(x_data, y_data, ftype):
    """
    Fit function to given data.

    Parameters:
    - x_data: Input data (independent variable)
    - y_data: Output data (dependent variable)
    - ftype: type of function we fit

    Returns:
    - Coefficients of the fitted function
    """
    # Use curve_fit to fit the data to the exponential function
    if ftype == "exp":
        params, covariance = curve_fit(exponential_function, x_data, y_data)
    if ftype == "poly":
        params, covariance = curve_fit(poly_function, x_data, y_data)
    if ftype == "lin":
        params, covariance = curve_fit(linear_function, x_data, y_data)

    # Extract the coefficients
    err = np.sqrt(np.diag(covariance))
    return params, err


# Example usage:
# Generate some example data
x_data = chis
# time_tot = np.array([3.23587389e+00,
#                     1.07527894e+01, 1.68719734e+02, 2.00160884e+03]) # 7.57573918e-02, 1.71739756e-01, 4.02117394e-01,
y_data = time_tot

# Fit the data to the exponential function
params_exp, err_exp = fit(x_data, y_data, "exp")
params_poly, err_poly = fit(x_data, y_data, "poly")

# Print the coefficients
print(f"Coefficients exp: {params_exp}")
print(f"Errors exp: {err_exp}")
print(f"Coefficients poly: {params_poly}")
print(f"Errors poly: {err_poly}")

# Plot the original data and the fitted curve
plt.title(f"DMRG Computational Time for $L={L}$ at $h=h_c$")
plt.scatter(x_data, y_data, label="Original Data")
x_fit = np.linspace(min(x_data), 1024, 1000)
# y_fit_exp = exponential_function(x_fit, params_exp[0], params_exp[1], params_exp[2])
y_fit_poly = poly_function(x_fit, params_poly[0], params_poly[1], params_poly[2])
# plt.plot(x_fit, y_fit_exp, 'r-', label='Fitted Exponential Curve')
plt.plot(x_fit, y_fit_poly, "g-", label="Fitted Polynomial Curve")
plt.fill_between(
    x_fit,
    poly_function(
        x_fit,
        params_poly[0] - err_poly[0],
        params_poly[1] - err_poly[1],
        params_poly[2] - err_poly[2],
    ),
    poly_function(
        x_fit,
        params_poly[0] + err_poly[0],
        params_poly[1] + err_poly[1],
        params_poly[2] + err_poly[2],
    ),
    color="green",
    alpha=0.2,
    label="Error Bounds",
)
plt.yscale("log")
# plt.xscale('log')
plt.xticks(ticks=chis + [1024], labels=chis + [1024])
plt.xlabel("bond dimension $(\chi)$")
plt.ylabel("time (t)")
y_last_plus = poly_function(
    x_fit[-1],
    params_poly[0] + err_poly[0],
    params_poly[1] + err_poly[1],
    params_poly[2] + err_poly[2],
)
print("last values")
print(y_fit_poly[-1])
print(y_last_plus)
y_err = abs(y_fit_poly[-1] - y_last_plus)

if y_fit_poly[-1] < 60:
    unit = "sec(s)"
elif y_fit_poly[-1] > 60 and y_fit_poly[-1] < 3600:
    unit = "min(s)"
    t = y_fit_poly[-1] / 60
    t_err = y_err / 60
elif y_fit_poly[-1] > 3600:
    unit = "hour(s)"
    t = y_fit_poly[-1] / 3600
    t_err = y_err / 3600

textstr = f"Expectation time for $\chi = 1024$ is {t:.2f} ± {t_err:.2f} {unit}"
plt.text(
    0.05,
    0.90,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
textstr = f"Fitted Parameters:\na = {params_poly[0]:.2e} ± {err_poly[0]:.2e}\nb = {params_poly[1]:.2f} ± {err_poly[1]:.2f}\nc = {params_poly[2]:.2f} ± {err_poly[2]:.2f}"
plt.text(
    0.05,
    0.80,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
textstr = f"Poly function:\n$a x^b + c$"
plt.text(
    0.33,
    0.80,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
plt.hlines(
    y=y_fit_poly[-1],
    xmin=chis[0],
    xmax=1024,
    linestyles="dashed",
    colors="black",
    label='time $``=" \langle t \\rangle_{\chi=1024}$',
)
plt.hlines(
    y=3600,
    xmin=chis[0],
    xmax=1024,
    linestyles="dashed",
    colors="black",
    linewidth=0.8,
    label="one hour",
)
plt.legend(loc="lower right", fontsize=10)
plt.savefig(f"times_dmrg_L_{L}_chis_16-32-64-128.png")
plt.show()


# time for different L

Ls = [20, 30, 40, 50, 60, 70, 80]
d = 2
model = "Ising"
chi = 64  # ,8,16,32,64,128,256,512
# hs = np.linspace(0,2,100)
h = 1
J = 1
eps = 0
# time_tot = np.array([
#     4.262944635422900319e+01,
#     1.045260993340052664e+02,
#     1.592617676099762321e+02,
#     2.302356971697881818e+02,
#     3.002348961471579969e+02,
#     3.662781152101233602e+02,
#     4.503631118210032582e+02,
#     ])

x_data = Ls
y_data = time_tot

params_lin, err_lin = fit(x_data, y_data, "lin")

# Plot the original data and the fitted curve
plt.title(f"DMRG Computational Time for $\chi={chi}$ at $h=h_c$")
plt.scatter(x_data, y_data, label="Original Data")
x_fit = np.linspace(min(x_data), 1000, 1000)
y_fit_lin = linear_function(x_fit, params_lin[0], params_lin[1])
plt.plot(x_fit, y_fit_lin, "g-", label="Fitted Linear Curve")
plt.fill_between(
    x_fit,
    linear_function(x_fit, params_lin[0] - err_lin[0], params_lin[1] - err_lin[1]),
    linear_function(x_fit, params_lin[0] + err_lin[0], params_lin[1] + err_lin[1]),
    color="green",
    alpha=0.2,
    label="Error Bounds",
)
# plt.xscale('log')
plt.xticks(ticks=Ls + [1000], labels=Ls + [1000])
plt.xlabel("chain length $(L)$")
plt.ylabel("time (t)")
y_last_plus = linear_function(
    x_fit[-1], params_lin[0] + err_lin[0], params_lin[1] + err_lin[1]
)
print("last values")
print(y_fit_lin[-1])
print(y_last_plus)
y_err = abs(y_fit_lin[-1] - y_last_plus)

if y_fit_lin[-1] < 60:
    unit = "sec(s)"
elif y_fit_lin[-1] > 60 and y_fit_lin[-1] < 3600:
    unit = "min(s)"
    t = y_fit_lin[-1] / 60
    t_err = y_err / 60
elif y_fit_lin[-1] > 3600:
    unit = "hour(s)"
    t = y_fit_lin[-1] / 3600
    t_err = y_err / 3600

textstr = f"Expectation time for $L = 1000$ is {t:.2f} ± {t_err:.2f} {unit}"
plt.text(
    0.05,
    0.90,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
textstr = f"Fitted Parameters:\na = {params_lin[0]:.2e} ± {err_lin[0]:.2e}\nb = {params_lin[1]:.2f} ± {err_lin[1]:.2f}"
plt.text(
    0.05,
    0.80,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
textstr = f"lin function:\n$a x + b$"
plt.text(
    0.35,
    0.80,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
plt.hlines(
    y=y_fit_lin[-1],
    xmin=chis[0],
    xmax=1024,
    linestyles="dashed",
    colors="black",
    label='time $``=" \langle t \\rangle_{L=1000}$',
)
plt.hlines(
    y=3600,
    xmin=chis[0],
    xmax=1024,
    linestyles="dashed",
    colors="black",
    linewidth=0.8,
    label="one hour",
)
plt.legend(loc="lower right", fontsize=10)
plt.savefig(f"times_dmrg_L_20-30-40-50-60-70-80_chi_{chi}.png")
plt.show()
