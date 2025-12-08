from qs_mps.utils import *
import h5py

# default parameters of the plot layout
plt.rcParams["text.usetex"] = True  # use latex
plt.rcParams["font.size"] = 13
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.constrained_layout.use"] = True

path = "marcos"
if path == "pc":
    path_tensor = f"C:/Users/HP/Desktop/projects/1_Z2"
    parent_path = path_tensor
    path_figures = "G:/My Drive/projects/1_Z2"

    # parent_path = "G:/My Drive/projects/1_Z2"
    # path_tensor = "D:/code/projects/1_Z2"
elif path == "mac":
    # parent_path = "/Users/fradm98/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/Users/fradm98/Desktop/projects/1_Z2"
    parent_path = path_tensor
elif path == "marcos":
    path_figures = "/Users/fradm/Google Drive/My Drive/projects/1_Z2"
    path_tensor = "/Users/fradm/Desktop/projects/1_Z2"
    parent_path = path_tensor

save_gif = True
target = {
    'L': 40, 'N': 5, 'R': 20, 'T': 200,
    'bc': 'pbc',
    'chis': np.array([128]),
    'cx': np.array([10, 30]),
    'cy': np.array([0, 0]),
    'delta': 0.05, 'h_ev': 0.8,
    'h_i': 0.8, 'of': 0.3,
}
chi_t = 128

efields = np.asarray(get_el_field_in_time(f"{parent_path}/results/time_data/results_time_test.hdf5", target, bond_dim=chi_t))

steps = len(efields)
if save_gif:
    movie = anim(frames=steps, interval=200, data=efields, params=np.linspace(0,steps*target['delta'],steps+1), show=False, charges_x=target['cx'], charges_y=target['cy'], precision=2, time=True)
    movie.save(filename=f"{path_figures}/figures/animations/quench_Z2_{target['L']}x{target['N']}_g_i_{target['h_i']}_g_ev_{target['h_ev']}_trott_steps_{steps}_delta_{target['delta']}_chi_{chi_t}.gif")