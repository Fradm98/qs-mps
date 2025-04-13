from qs_mps.mps_class import MPS
from qs_mps.utils import get_cx, get_cy, create_sequential_colors
import numpy as np
import matplotlib.pyplot as plt

bc = "pbc"
model = "Z2_dual"
precision = 3
path_figures = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
path_tensor = "/Users/fradm/Desktop/projects/1_Z2"

# default parameters of the plot layout
plt.rcParams["text.usetex"] = True  # use latex
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

font = {'family': 'serif', 'size': 12}
plt.rcParams.update({'font.family': font['family'], 'font.size': font['size']})

def fidelity_susceptibility(l, L, chi, R, bc, model, h_i, h_f, npoints):
    gs = np.linspace(h_i,h_f,npoints)
    cx = get_cx(L, R)
    cy = get_cy(l, bc)
    
    if len(cx) == 0 and R == 0:
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
        fid = mps_g._compute_norm(site=1, mixed=True)
        fidelities.append(fid)
    return np.gradient(np.gradient(fidelities))

def plot_fidelity_susceptibility(fidelities, l, L, R, chi, h_i, h_f, npoints, color):
    gs = np.linspace(h_i, h_f, npoints)
    plt.plot(gs, fidelities, color=color, label=f"$R: {R}$")

L = 30
R = 20


Rs = [19,21]
colors = create_sequential_colors(len(Rs))
l = 6
chi = 256
h_i, h_f, npoints = 0.6, 0.95, 15
plt.title(f"$\\chi_{{\\mathcal{{F}}}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2$ for $l \\times L: {l} \\times {L}$, $D:{chi}$")
plt.xlabel("electric coupling $(g)$")
plt.ylabel("fidelity susceptibility $(\\chi_{\\mathcal{F}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2)$")
for i, R in enumerate(Rs):
    fidelities = fidelity_susceptibility(l, L, chi, R, bc, model, h_i, h_f, npoints)
    plot_fidelity_susceptibility(fidelities, l, L, R, chi, h_i, h_f, npoints, colors[i])
plt.savefig(f"{path_figures}/fluxtube/fidelity_susceptibility_{model}_{l}x{L}_bc_{bc}_Rs_{Rs}_npoints_{npoints}_h_{h_i}-{h_f}_chi_{chi}.png")
plt.legend()
plt.show()
plt.close()

Rs = [10,11,12,13,14,15,16,17,18,19,20]
colors = create_sequential_colors(len(Rs))
l = 5
chi = 128
h_i, h_f, npoints = 0.4, 1.0, 61
plt.title(f"$\\chi_{{\\mathcal{{F}}}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2$ for $l \\times L: {l} \\times {L}$, $D:{chi}$")
plt.xlabel("electric coupling $(g)$")
plt.ylabel("fidelity susceptibility $(\\chi_{\\mathcal{F}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2)$")
for i, R in enumerate(Rs):
    fidelities = fidelity_susceptibility(l, L, chi, R, bc, model, h_i, h_f, npoints)
    plot_fidelity_susceptibility(fidelities, l, L, R, chi, h_i, h_f, npoints, colors[i])
plt.savefig(f"{path_figures}/fluxtube/fidelity_susceptibility_{model}_{l}x{L}_bc_{bc}_Rs_{Rs}_npoints_{npoints}_h_{h_i}-{h_f}_chi_{chi}.png")
plt.legend()
plt.show()
plt.close()

h_i, h_f, npoints = 0.8, 1.0, 41
plt.title(f"$\\chi_{{\\mathcal{{F}}}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2$ for $l \\times L: {l} \\times {L}$, $D:{chi}$")
plt.xlabel("electric coupling $(g)$")
plt.ylabel("fidelity susceptibility $(\\chi_{\\mathcal{F}} = d^2 \\langle \\psi (g) | \\psi(g+dg) \\rangle / dg^2)$")
for i, R in enumerate(Rs):
    fidelities = fidelity_susceptibility(l, L, chi, R, bc, model, h_i, h_f, npoints)
    plot_fidelity_susceptibility(fidelities, l, L, R, chi, h_i, h_f, npoints, colors[i])
plt.savefig(f"{path_figures}/fluxtube/fidelity_susceptibility_{model}_{l}x{L}_bc_{bc}_Rs_{Rs}_npoints_{npoints}_h_{h_i}-{h_f}_chi_{chi}.png")
plt.legend()
plt.show()
plt.close()