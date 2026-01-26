from qs_mps.utils import *
import h5py

# default parameters of the plot layout
plt.rcParams["text.usetex"] = False  # use latex
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
    'bc': 'pbc', 'chis': np.array([128, 192, 256, 384, 512]), 
    'cx': np.array([10, 30]), 'cy': np.array([0, 0]), 
    'delta': 0.05, 'h_ev': 0.8, 'h_i': 0.8, 'of': 0.3,
}

target = {
    'L': 40, 'N': 7, 'R': 20, 'T': 200, 
    'bc': 'pbc', 'chis': [128, 192, 256, 384, 512], 
    'cx': [10, 30], 'cy': [0, 0], 
    'delta': 0.05, 'h_ev': 0.8, 'h_i': 0.8, 'of': 0.3,
}

chi_t = 128

lattice = False

efields = np.asarray(get_el_field_in_time(f"{parent_path}/results/time_data/results_time_cluster.hdf5", target, bond_dim=chi_t))

steps = len(efields)

if lattice:
    if save_gif:
        movie = anim(frames=steps, interval=200, data=efields, params=np.linspace(0,steps/target['of']*target['delta'],steps+1), show=False, charges_x=target['cx'], charges_y=target['cy'], precision=2, time=True)
        movie.save(filename=f"{path_figures}/figures/animations/quench_Z2_{target['L']}x{target['N']}_g_i_{target['h_i']}_g_ev_{target['h_ev']}_trott_steps_{steps}_delta_{target['delta']}_chi_{chi_t}.gif")

else:
    data = []
    for field in efields:
        hor_field = np.roll(field, shift=target['N']-1, axis=0)
        hor_field = hor_field[::2,1::2]
        hor_occup = (1 - hor_field) / 2
        data.append(hor_occup)
    data = np.array(data)
    cy = [target['N']//2, target['N']//2]

if save_gif:
    movie = anim_no_lattice(frames=steps, interval=200, data=data, params=np.linspace(0,steps/target['of']*target['delta'],steps+1), show=False, charges_x=target['cx'], charges_y=cy, precision=2, time=True)
    movie.save(filename=f"{path_figures}/figures/animations/quench_occupation_{target['L']}x{target['N']}_g_start_{target['h_i']}_g_ev_{target['h_ev']}_trott_steps_{target['T']}_delta_{target['delta']}_chi_{chi_t}.gif")