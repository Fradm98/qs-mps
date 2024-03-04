import matplotlib.pyplot as plt
import numpy as np
from qs_mps.utils import load_list_of_lists, tensor_shapes
from qs_mps.mps_class import MPS
from scipy.optimize import curve_fit

path_tensor = "/Users/fradm98/Desktop/projects/1_Z2"
charges_x = [1,3]
charges_y = [2,2]
precision = 2
lattice_mps = MPS(L=5,d=16, model="Z2_dual", chi=50, h=0.1)
lattice_mps.L = lattice_mps.L - 1
lattice_mps.Z2.add_charges(charges_x,charges_y)
lattice_mps.load_sites(path=path_tensor, precision=precision, cx=charges_x, cy=charges_y)

lattice_mps.TEBD_variational_Z2(trotter_steps=10, delta=0.1, h_ev=0.5, quench="global", flip=False)
