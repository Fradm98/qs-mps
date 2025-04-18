from qs_mps.mps_class import MPS
from qs_mps.utils import get_cx, get_cy, create_sequential_colors
from ncon import ncon
import numpy as np
import matplotlib.pyplot as plt

bc = "pbc"
model = "Z2_dual"
precision = 3
path_figures = "/Users/fradm/Google Drive/My Drive/projects/1_Z2/figures"
path_tensor = "/Users/fradm/Desktop/projects/1_Z2"
path_figures = "/Users/fradm/Desktop/projects/1_Z2/figures"

# default parameters of the plot layout
# plt.rcParams["text.usetex"] = True  # use latex
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

font = {'family': 'serif', 'size': 12}
plt.rcParams.update({'font.family': font['family'], 'font.size': font['size']})

def fidelity_susceptibility(l, L, chi, R, bc, model, h_i, h_f, npoints, log: bool = False, rdm: bool=False):
    gs = np.linspace(h_i,h_f,npoints)
    cx = get_cx(L, R)
    cy = get_cy(l, bc, R=R)
    sites = [L//2, L//2+1]
    if R == 0:
        sector = "vacuum_sector"
        cx = None
        cy = None
    else:
        sector = f"{len(cx)}_particle(s)_sector"
        cx = cx
        cy = cy

    fidelities = []
    for i in range(len(gs)-1):
        g = gs[i]
        g_dg = gs[i+1]
        mps_g = MPS(
            L=L, d=2**l, model=model, chi=chi, h=g, bc=bc
        )
        mps_g_dg = MPS(
            L=L, d=2**l, model=model, chi=chi, h=g_dg, bc=bc
        )

        if sector != "vacuum_sector":
            mps_g.Z2.add_charges(cx, cy)
            mps_g.charges = mps_g.Z2.charges
            mps_g.Z2._define_sector()
            mps_g_dg.Z2.add_charges(cx, cy)
            mps_g_dg.charges = mps_g_dg.Z2.charges
            mps_g_dg.Z2._define_sector()

        else:
            mps_g.Z2._define_sector()
            mps_g_dg.Z2._define_sector()
        mps_g.load_sites(
            path=path_tensor, precision=precision, cx=cx, cy=cy
        )
        mps_g_dg.load_sites(
            path=path_tensor, precision=precision, cx=cx, cy=cy
        )
        mps_g.ancilla_sites = mps_g_dg.sites.copy()

        if rdm:
            fid = ncon([mps_g.sites[sites[0]],
                        mps_g.sites[sites[1]], 
                        mps_g.sites[sites[0]].conjugate(),
                        mps_g.sites[sites[1]].conjugate(), 
                        mps_g_dg.sites[sites[0]],
                        mps_g_dg.sites[sites[1]],
                        mps_g_dg.sites[sites[0]].conjugate(),
                        mps_g_dg.sites[sites[1]].conjugate()],
                        [[1,2,3],[3,4,5],
                         [1,7,6],[6,8,5],
                         [9,7,10],[10,8,11],
                         [9,2,12],[12,4,11]])
            fidelities.append(np.sqrt(fid))
            
        else:
            fid = mps_g._compute_norm(site=1, mixed=True)
            if log:
                fidelities.append(np.log(np.sqrt(fid.real**2 + fid.imag**2)))
            else:
                fidelities.append(np.sqrt(fid.real**2 + fid.imag**2))
    return np.gradient(np.gradient(fidelities))

def plot_fidelity_susceptibility(fidelities, R, h_i, h_f, npoints, color):
    gs = np.linspace(h_i, h_f, npoints-1)
    plt.plot(gs, fidelities, color=color, label=f"$R: {R}$")

L = 30
R = 20


Rs = [19,21]
Rs = [0,11,12,13,14,15,16,17,18,19,20]
colors = create_sequential_colors(len(Rs))
l = 6
chi = 256
chi = 32
log = True
rdm = False
h_i, h_f, npoints = 0.6, 0.95, 15
h_i, h_f, npoints = 0.4, 0.6, 21
plt.title(f"$\\chi_{{\\mathcal{{F}}}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2$ for $l \\times L: {l} \\times {L}$, $D:{chi}$, $log: {log}$")
plt.xlabel("electric coupling $(g)$")
plt.ylabel("fidelity susceptibility $(\\chi_{\\mathcal{F}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2)$")
for i, R in enumerate(Rs):
    fidelities = fidelity_susceptibility(l, L, chi, R, bc, model, h_i, h_f, npoints, log=log, rdm=rdm)
    plot_fidelity_susceptibility(fidelities, R, h_i, h_f, npoints, colors[i])
plt.legend()
plt.savefig(f"{path_figures}/fluxtube/fidelity_susceptibility_log_{log}_rdm_{rdm}_{model}_{l}x{L}_bc_{bc}_Rs_{Rs}_npoints_{npoints}_h_{h_i}-{h_f}_chi_{chi}.png")
plt.close()

# Rs = [18,20]
# colors = create_sequential_colors(len(Rs))
l = 5
chi = 128
# log = True
# h_i, h_f, npoints = 0.4, 1.0, 61
# plt.title(f"$\\chi_{{\\mathcal{{F}}}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2$ for $l \\times L: {l} \\times {L}$, $D:{chi}$, $log: {log}$")
# plt.xlabel("electric coupling $(g)$")
# plt.ylabel("fidelity susceptibility $(\\chi_{\\mathcal{F}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2)$")
# for i, R in enumerate(Rs):
#     fidelities = fidelity_susceptibility(l, L, chi, R, bc, model, h_i, h_f, npoints, log=log)
#     plot_fidelity_susceptibility(fidelities, l, L, R, chi, h_i, h_f, npoints, colors[i])
# plt.legend()
# plt.savefig(f"{path_figures}/fluxtube/fidelity_susceptibility_log_{log}_{model}_{l}x{L}_bc_{bc}_Rs_{Rs}_npoints_{npoints}_h_{h_i}-{h_f}_chi_{chi}.png")
# plt.close()

# Rs = [10,12,14,16,18,20]
# colors = create_sequential_colors(len(Rs))
# log = False
# h_i, h_f, npoints = 0.8, 1.0, 41
# plt.title(f"$\\chi_{{\\mathcal{{F}}}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2$ for $l \\times L: {l} \\times {L}$, $D:{chi}$, $log: {log}$")
# plt.xlabel("electric coupling $(g)$")
# plt.ylabel("fidelity susceptibility $(\\chi_{\\mathcal{F}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2)$")
# for i, R in enumerate(Rs):
#     fidelities = fidelity_susceptibility(l, L, chi, R, bc, model, h_i, h_f, npoints, log=log)
#     plot_fidelity_susceptibility(fidelities, l, L, R, chi, h_i, h_f, npoints, colors[i])
# plt.legend()
# plt.savefig(f"{path_figures}/fluxtube/fidelity_susceptibility_log_{log}_{model}_{l}x{L}_bc_{bc}_Rs_{Rs}_npoints_{npoints}_h_{h_i}-{h_f}_chi_{chi}.png")
# plt.close()

# Rs = [0,11,13,15,17,19]
# Rs = [0,11]
# colors = create_sequential_colors(len(Rs))
# log = False
# rdm = False
# h_i, h_f, npoints = 0.8, 1.0, 41
# plt.title(f"$\\chi_{{\\mathcal{{F}}}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2$ for $l \\times L: {l} \\times {L}$, $D:{chi}$, $log: {log}$")
# plt.xlabel("electric coupling $(g)$")
# plt.ylabel("fidelity susceptibility $(\\chi_{\\mathcal{F}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2)$")
# for i, R in enumerate(Rs):
#     if R != 0:
#         cy = None
#     fidelities = fidelity_susceptibility(l, L, chi, R, bc, model, h_i, h_f, npoints, log=log, rdm=rdm)
#     plot_fidelity_susceptibility(fidelities, l, L, R, chi, h_i, h_f, npoints, colors[i])
# plt.legend()
# plt.savefig(f"{path_figures}/fluxtube/fidelity_susceptibility_log_{log}_{model}_{l}x{L}_bc_{bc}_Rs_{Rs}_npoints_{npoints}_h_{h_i}-{h_f}_chi_{chi}.png")
# plt.close()
