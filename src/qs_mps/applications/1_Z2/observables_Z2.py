#%%
# import packages
from qs_mps.mps_class import MPS
from qs_mps.utils import *
import matplotlib.pyplot as plt
from ncon import ncon
from scipy.sparse import csr_array
# %%
L = 4
d = 4
chi = 16 # this is interpreted as d**(int(log2(chi))) --> e.g. chi=8 == 4**3=64
# array = np.linspace(1e-3,0.1,20)
# array = np.linspace(0.4,2,20)
array = np.linspace(100,1000,10)

hs = [h for h in array]
model = "Z2_two_ladder"
charges = [1,1,-1,-1,1,1]

# %%
for h in hs:
    ladder = MPS(L=L, d=4, model=model, chi=chi, charges=charges, h=h)
    ladder.load_sites("/Users/fradm98/Desktop/mps/tests/results/tensor_data")
    mag_ladders = []
    for l in range(1,3):
        mag_sites = []
        for i in range(1,L+1):
            ladder.sigma_x_Z2_two_ladder(site=i, ladder=l)
            mag_loc = ladder.mpo_first_moment()
            mag_sites.append(mag_loc)
        mag_ladders.append(mag_sites)
    plt.imshow(mag_ladders, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()

# %%
